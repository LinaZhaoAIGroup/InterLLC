import warnings
warnings.filterwarnings("ignore")

import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any
import warnings

from configurations import xafs_group, xanes_group, exafs_group, simple_feature_names


class XAFSFeatureProcessor(nn.Module):
    """
    XAFS处理器
    """
    def __init__(self, descriptor_dims: Dict[str, int], hidden_dim: int = 256,
                 dropout_rate: float = 0.1, fusion_dropout: float = 0.2, 
                 fusion_type: str = 'hierarchical', device: str = 'cpu'):
        """
        Args:
            descriptor_dims: 描述符名称 -> 实际维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.descriptor_dims = descriptor_dims
        self.hidden_dim = hidden_dim
        
        self.processor = HierarchicalDescriptorFusion(
            descriptor_dims=descriptor_dims,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            fusion_dropout=fusion_dropout,
            fusion_type=fusion_type,
            device=device
        )
        
        self.config = self.processor.get_config_info()
    
    def forward(self, xafs_input):
        """
        Args:
            xafs_input: 字典 {descriptor_name: tensor}
        """
        if isinstance(xafs_input, dict):
            return self.processor(xafs_input)
        else:
            raise ValueError("XAFSFeatureProcessor需要字典格式的输入")
    
    def get_config(self):
        return self.config


class HierarchicalDescriptorFusion(nn.Module):
    """
    层次融合策略：
    1. 组内融合：相似物理意义的描述符先融合
    2. 组间交叉注意力：电子结构与结构信息交互
    3. 全局门控：动态权重分配
    """
    def __init__(self, descriptor_dims: Dict[str, int], hidden_dim: int = 256,
                 dropout_rate: float = 0.1, fusion_dropout: float = 0.2, 
                 fusion_type: str = 'hierarchical', device: str = 'cpu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.descriptor_dims = descriptor_dims
        self.dropout_rate = dropout_rate        
        self.fusion_dropout = fusion_dropout   
        init_device = torch.device(device)    
        self.fusion_type = fusion_type
        
        self.xafs_group = xafs_group
        self.xanes_group = xanes_group  
        self.exafs_group = exafs_group  
        self.global_group = simple_feature_names
        self.xafs_group = [name for name in self.xafs_group if name in descriptor_dims]
        self.xanes_group = [name for name in self.xanes_group if name in descriptor_dims]
        self.exafs_group = [name for name in self.exafs_group if name in descriptor_dims]
        self.global_group = [name for name in self.global_group if name in descriptor_dims]
        print(f"Signal组特征{len(self.xafs_group)}个: {self.xafs_group}")
        print(f"XANES组特征{len(self.xanes_group)}个: {self.xanes_group}")
        print(f"EXAFS组特征{len(self.exafs_group)}个: {self.exafs_group}")
        print(f"Addition组特征{len(self.global_group)}个: {self.global_group}")
        
        # 组内编码器
        self.xafs_encoder = self._create_group_encoder(self.xafs_group, 'xafs', 
                            descriptor_dims, hidden_dim, self.dropout_rate, self.fusion_dropout, init_device)
        self.xanes_encoder = self._create_group_encoder(self.xanes_group, 'xanes', 
                            descriptor_dims, hidden_dim, self.dropout_rate, self.fusion_dropout, init_device)
        self.exafs_encoder = self._create_group_encoder(self.exafs_group, 'exafs', 
                            descriptor_dims, hidden_dim, self.dropout_rate, self.fusion_dropout, init_device)
        if self.global_group:
            self.global_encoder = nn.Sequential(
                        nn.Linear(len(self.global_group), 32),
                        nn.BatchNorm1d(32),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(32, 16),
                        nn.BatchNorm1d(16),
                        nn.ReLU()
                    ).to(init_device)
            self.global_encoder.output_dim = 16
        else:
            self.global_encoder = None

        self.actual_groups = []
        self.group_encoders = {}
        if self.xafs_encoder is not None:
            self.actual_groups.append('xafs')
            self.group_encoders['xafs'] = self.xafs_encoder
        if self.xanes_encoder is not None:
            self.actual_groups.append('xanes')
            self.group_encoders['xanes'] = self.xanes_encoder
        if self.exafs_encoder is not None:
            self.actual_groups.append('exafs')
            self.group_encoders['exafs'] = self.exafs_encoder
        if self.global_encoder is not None:
            self.actual_groups.append('global')
            self.group_encoders['global'] = self.global_encoder

        if self.fusion_type == 'hierarchical':
            self._init_hierarchical_fusion(init_device)
        else:   
            self._init_concat_fusion(init_device)
        
        print(f"最终隐藏维度: {hidden_dim}")
        print("=" * 50)

    def _init_hierarchical_fusion(self, device):
        self.num_groups = len(self.actual_groups)
        if self.num_groups > 1:
            self.attention_dim = self._get_encoder_output_dim()
            if self.attention_dim % 4 != 0:
                self.attention_dim = ((self.attention_dim + 3) // 4) * 4
                print(f"调整注意力维度为: {self.attention_dim}")
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.attention_dim, 
                num_heads=4, 
                dropout=0.1, 
                batch_first=True
            ).to(device)
            self.gate = nn.Sequential(
                nn.Linear(self.attention_dim * self.num_groups, self.attention_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.attention_dim, self.num_groups),
                nn.Softmax(dim=-1)
            ).to(device)
        else:
            self.cross_attention = None
            self.gate = None
            self.attention_dim = self._get_encoder_output_dim()

        self.projection = nn.Sequential(
            nn.Linear(self.attention_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        ).to(device)

    def _init_concat_fusion(self, device):
        self.num_groups = len(self.actual_groups)
        self.concat_total_dim = 0
        for group_name in self.actual_groups:
            encoder = self.group_encoders[group_name]
            if hasattr(encoder, 'output_dim'):
                dim = encoder.output_dim
            elif isinstance(encoder, nn.Sequential) and len(encoder) > 0:
                last_layer = encoder[-1]
                if isinstance(last_layer, nn.Linear):
                    dim = last_layer.out_features
                else:
                    with torch.no_grad():
                        if group_name == 'global':
                            input_dim = len(self.global_group)
                        else:
                            first_desc = getattr(self, f'{group_name}_group')[0]
                            input_dim = self.descriptor_dims[first_desc]
                        dummy = torch.randn(1, input_dim).to(device)
                        out = encoder(dummy)
                        dim = out.shape[-1]
            else:
                raise AttributeError(f"无法获取编码器 {group_name} 的输出维度")
            self.concat_total_dim += dim
            print(f"  组 {group_name} 输出维度: {dim}")

        self.projection = nn.Sequential(
            nn.Linear(self.concat_total_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        ).to(device)
        self.cross_attention = None
        self.gate = None
        self.attention_dim = self.concat_total_dim
 

    def _get_encoder_output_dim(self) -> int:
        for encoder in [self.xafs_encoder, self.xanes_encoder, self.exafs_encoder, self.global_encoder]:
            if encoder is not None:
                return encoder.output_dim
        return self.hidden_dim // 2  
    
    def _create_group_encoder(self, group_names: List[str], group_type: str,
                         descriptor_dims: Dict[str, int], hidden_dim: int,
                         dropout_rate: float, fusion_dropout: float, device: str) -> Optional[nn.Module]:
        if not group_names:
            return None
        
        class GroupEncoder(nn.Module):
            def __init__(self, descriptor_dims: Dict[str, int], 
                     group_names: List[str], group_type: str, hidden_dim: int, 
                     dropout_rate: float, fusion_dropout: float, device: str):
                
                super().__init__()
                init_device = torch.device(device)
                self.group_names = group_names
                self.group_type = group_type
                self.descriptor_dims = descriptor_dims
                
                print(f"\n创建 {group_type} 组编码器:")
                self.encoders = nn.ModuleDict()
                self.output_dims = {}  
                
                for name in group_names:
                    input_dim = descriptor_dims[name]
                    
                    if group_type == 'global':
                        output_dim = 8
                    else:
                        if name.startswith(('pdos', 'xtb', 'cwt', 'wacsf', 'soap', 'rdf', 'rdc', 'mbtr', 'lmbtr', 'msr')):
                            output_dim = max(32, input_dim // 6)
                        else:
                            output_dim = max(16, input_dim // 4)
                    
                    output_dim = min(output_dim, 64)
                    self.output_dims[name] = output_dim 
                    
                    encoder = nn.Sequential(
                        nn.Linear(input_dim, output_dim * 2),
                        nn.BatchNorm1d(output_dim * 2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.Linear(output_dim * 2, output_dim),
                        nn.BatchNorm1d(output_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate)
                    ).to(init_device)
                    self.encoders[name] = encoder
                
                self.total_intermediate_dim = sum(
                    self.output_dims[name]  
                    for name in group_names
                )
                
                self.output_dim = 64
                self.group_fusion = nn.Sequential(
                    nn.Linear(self.total_intermediate_dim, self.output_dim * 2),
                    nn.BatchNorm1d(self.output_dim * 2),
                    nn.ReLU(),
                    nn.Dropout(fusion_dropout),
                    nn.Linear(self.output_dim * 2, self.output_dim),
                    nn.BatchNorm1d(self.output_dim),
                    nn.ReLU()
                ).to(init_device)
                
                print(f"  组输出维度: {self.output_dim}")
                
            def forward(self, descriptors: Dict[str, torch.Tensor]) -> torch.Tensor:
                encoded_features = []
                current_device = None
                for name in self.group_names:
                    if name in descriptors:
                        current_device = descriptors[name].device
                        break
                if current_device is None:
                    current_device = torch.device('cpu')
                    
                for name in self.group_names:
                    if name in descriptors:
                        desc_tensor = descriptors[name].to(current_device)
                        if desc_tensor.dim() == 1:
                            desc_tensor = desc_tensor.unsqueeze(1)
                        elif desc_tensor.dim() != 2:
                            raise ValueError(f"Unexpected shape for {name}: {desc_tensor.shape}")
                        expected_dim = self.descriptor_dims[name]
                        if desc_tensor.shape[1] != expected_dim:
                            if desc_tensor.shape[1] > expected_dim:
                                desc_tensor = desc_tensor[:, :expected_dim]
                            else:
                                padding = expected_dim - desc_tensor.shape[1]
                                desc_tensor = F.pad(desc_tensor, (0, padding))
                        
                        encoder = self.encoders[name]
                        if next(encoder.parameters()).device != desc_tensor.device:
                            encoder = encoder.to(desc_tensor.device)
                            self.encoders[name] = encoder
                        encoded = encoder(desc_tensor)
                        encoded_features.append(encoded)
                    else:
                        batch_size = next(iter(descriptors.values())).shape[0]
                        output_dim = self.output_dims[name]
                        encoded_features.append(torch.zeros(batch_size, output_dim, device=current_device))
                concat_features = torch.cat(encoded_features, dim=1)
                
                if next(self.group_fusion.parameters()).device != current_device:
                    self.group_fusion = self.group_fusion.to(current_device)
                group_features = self.group_fusion(concat_features)
                return group_features
        
        return GroupEncoder(descriptor_dims, group_names, group_type, hidden_dim, dropout_rate, fusion_dropout, device)
    
    def _ensure_dim_adjust_layers(self, group_features: List[torch.Tensor]):
        if not hasattr(self, 'dim_adjust_layers') or self.dim_adjust_layers is None:
            self.dim_adjust_layers = nn.ModuleList()
        if len(self.dim_adjust_layers) != len(group_features):
            self.dim_adjust_layers = nn.ModuleList()
            for feat in group_features:
                layer = nn.Linear(feat.shape[1], self.attention_dim)
                self.dim_adjust_layers.append(layer)

    def _adjust_features(self, group_features):
        self._ensure_dim_adjust_layers(group_features)
        adjusted = []
        for idx, feat in enumerate(group_features):
            target_device = feat.device
            if self.dim_adjust_layers[idx].weight.device != target_device:
                self.dim_adjust_layers[idx] = self.dim_adjust_layers[idx].to(target_device)
            if feat.shape[1] != self.attention_dim:
                adjusted.append(self.dim_adjust_layers[idx](feat))
            else:
                adjusted.append(feat)
        return adjusted
    
    def get_descriptor_importance(self, descriptors: Dict[str, torch.Tensor]) -> Dict[str, float]:
        importance_scores = {}
        
        with torch.no_grad():
            for name, tensor in descriptors.items():
                
                if tensor.numel() > 0:
                    variance = torch.var(tensor).item()
                    importance = variance + 1e-6
                else:
                    importance = 1e-6
                
                importance_scores[name] = importance
        
        return importance_scores
    
    def forward(self, descriptors: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        descriptors: 字典，描述符名称 -> 张量 [batch, descriptor_dim]
        """
        if not descriptors:
            sample_device = next(iter(descriptors.values())).device if descriptors else torch.device('cpu')
            return torch.zeros(1, self.hidden_dim // 2, device=sample_device)
        
        # 1. 组内融合
        group_features = []
        if self.xafs_encoder is not None:
            xafs_features = self.xafs_encoder(descriptors)
            group_features.append(xafs_features)
        if self.xanes_encoder is not None:
            xanes_features = self.xanes_encoder(descriptors)
            group_features.append(xanes_features)
        if self.exafs_encoder is not None:
            exafs_features = self.exafs_encoder(descriptors)
            group_features.append(exafs_features)
        
        if self.global_encoder:
            global_tensors = [descriptors[name] for name in self.global_group]
            global_concat = torch.cat(global_tensors, dim=1)
            if next(self.global_encoder.parameters()).device != global_concat.device:
                self.global_encoder = self.global_encoder.to(global_concat.device)
            global_features = self.global_encoder(global_concat)
            group_features.append(global_features)
        
        # 2. 根据 fusion_type 选择融合路径
        if self.fusion_type == 'hierarchical':
            if len(group_features) == 1:
                return self.projection(group_features[0])
            adjusted_features = self._adjust_features(group_features)
            group_features_tensor = torch.stack(adjusted_features, dim=1)
            if self.cross_attention is not None:
                if next(self.cross_attention.parameters()).device != group_features_tensor.device:
                    self.cross_attention = self.cross_attention.to(group_features_tensor.device)
                attended, _ = self.cross_attention(
                    group_features_tensor, group_features_tensor, group_features_tensor)
            else:
                attended = group_features_tensor

            if self.gate is not None:
                concat_all = torch.cat(adjusted_features, dim=1)
                if next(self.gate.parameters()).device != concat_all.device:
                    self.gate = self.gate.to(concat_all.device)
                gate_weights = self.gate(concat_all).unsqueeze(2)
                final_features = (attended * gate_weights).sum(dim=1) 
            else:
                final_features = attended.mean(dim=1) 
            
            if next(self.projection.parameters()).device != final_features.device:
                self.projection = self.projection.to(final_features.device)
            output = self.projection(final_features)
            return output
        
        else:   
            concat_features = torch.cat(group_features, dim=1)   # [batch, total_dim]
            if next(self.projection.parameters()).device != concat_features.device:
                self.projection = self.projection.to(concat_features.device)
            output = self.projection(concat_features)
            return output

    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息"""
        info = {
            'groups': {
                'xafs': {
                    'names': self.xafs_group,
                    'size': len(self.xafs_group),
                    'encoder': 'exists' if self.xafs_encoder else 'none'
                },
                'xanes': {
                    'names': self.xanes_group,
                    'size': len(self.xanes_group),
                    'encoder': 'exists' if self.xanes_encoder else 'none'
                },
                'exafs': {
                    'names': self.exafs_group,
                    'size': len(self.exafs_group),
                    'encoder': 'exists' if self.exafs_encoder else 'none'
                },
                'global': {
                    'names': self.global_group,
                    'size': len(self.global_group),
                    'encoder': 'exists' if self.global_encoder else 'none'
                }
            },
            'num_groups': self.num_groups,
            'attention_dim': self.attention_dim,
            'hidden_dim': self.hidden_dim,
            'descriptor_dims': self.descriptor_dims,
            'fusion_type': self.fusion_type
        }
        if self.fusion_type == 'hierarchical':
            info['attention_dim'] = self.attention_dim
        else:
            info['concat_total_dim'] = self.concat_total_dim
        return info
    
    def get_attention_weights(self, descriptors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        if self.fusion_type == 'concat':
            return {}
        
        if self.num_groups <= 1:
            return {}
        
        with torch.no_grad():
            group_features = []
            group_names = []
            
            if self.xafs_encoder is not None:
                xafs_features = self.xafs_encoder(descriptors)
                group_features.append(xafs_features)
                group_names.append('xafs')
            
            if self.xanes_encoder is not None:
                xanes_features = self.xanes_encoder(descriptors)
                group_features.append(xanes_features)
                group_names.append('xanes')
            
            if self.exafs_encoder is not None:
                exafs_features = self.exafs_encoder(descriptors)
                group_features.append(exafs_features)
                group_names.append('exafs')
            
            if self.global_encoder is not None:
                global_features = self.global_encoder(descriptors)
                group_features.append(global_features)
                group_names.append('global')
            
            adjusted_features = self._adjust_features(group_features)
            group_tensor = torch.stack(adjusted_features, dim=1)          # [B, G, D]
            if self.cross_attention is not None:
                _, attention_weights = self.cross_attention(
                    group_tensor, group_tensor, group_tensor,
                    need_weights=True)   # [B, G, G]
            else:
                device = group_tensor.device
                attention_weights = torch.eye(len(group_names), device=device).unsqueeze(0).repeat(group_tensor.size(0), 1, 1)
            
            concat_all = torch.cat(adjusted_features, dim=1)              # [B, G*D]
            if self.gate is not None:
                gate_weights = self.gate(concat_all)                      # [B, G]
            else:
                gate_weights = torch.full((group_tensor.size(0), len(group_names)),
                                          1.0 / len(group_names), device=concat_all.device)
            
            descriptor_importance = {}
            for group_name, encoder in zip(['xafs', 'xanes', 'exafs', 'global'],
                                          [self.xafs_encoder, self.xanes_encoder, 
                                           self.exafs_encoder, self.global_encoder]):
                if encoder is not None:
                    for desc_name in getattr(self, f'{group_name}_group', []):
                        if desc_name in descriptors:
                            desc_tensor = descriptors[desc_name]
                            importance = torch.norm(desc_tensor, dim=1).mean().item()
                            descriptor_importance[desc_name] = importance
            
            return {
                'group_attention': attention_weights.mean(dim=0),  # [num_groups, num_groups]
                'gate_weights': gate_weights.mean(dim=0),  # [num_groups]
                'descriptor_importance': descriptor_importance,
                'group_names': group_names
            }
        
class XRDCNN(nn.Module):
    def __init__(self, input_length: int = 1024, output_dim: int = 256, dropout_fc: float = 0.2):
        super().__init__()
        
        self.conv1 = nn.Conv1d(1, 27, kernel_size=11, stride=1, padding=5)
        self.conv2 = nn.Conv1d(27, 27, kernel_size=11, stride=1, padding=5)
        self.conv3 = nn.Conv1d(27, 27, kernel_size=11, stride=1, padding=5)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.dropout_conv = nn.Dropout(0.21)
        self.dropout_fc1 = nn.Dropout(0.4)
        self.dropout_fc2 = nn.Dropout(0.3)
        self.dropout_fc3 = nn.Dropout(dropout_fc)
        
        conv_output_length = input_length // 8  
        
        self.fc1 = nn.Linear(27 * conv_output_length, 2000)
        self.fc2 = nn.Linear(2000, 512)
        self.fc3 = nn.Linear(512, output_dim)  
        
        self.output_dim = output_dim
        
        self._initialize_weights()

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3:
            if x.shape[1] != 1:
                x = x.unsqueeze(1)
        
        batch_size = x.shape[0]
        
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout_conv(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout_conv(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout_conv(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        
        x = self.fc3(x)
        x = self.dropout_fc3(x)
        
        if x.shape[1] != self.output_dim:
            print(f"警告: XRDCNN输出维度为{x.shape[1]}，期望{self.output_dim}")
            if not hasattr(self, 'dim_adjust'):
                self.dim_adjust = nn.Linear(x.shape[1], self.output_dim).to(x.device)
            x = self.dim_adjust(x)
        
        return x
    
class TaskGate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)

class MultiModalFusionModel(nn.Module):
    def __init__(self, 
                 xrd_seq_len: int = 1024,
                 xafs_total_dim: int = 500,
                 descriptor_split_dims: dict = None,
                 hidden_dim: int = 256,
                 output_activation: str = 'none',
                 xrd_dropout_fc: float = 0.2,
                 xafs_dropout_rate: float = 0.1,
                 xafs_fusion_dropout: float = 0.2,
                 modal_attention_dropout: float = 0.2,
                 fusion_layer_dropout: float = 0.3,
                 task_head_dropout: float = 0.2,
                 fusion_type: str = 'concat',        
                 cross_attn_heads: int = 4,          
                 cross_attn_dropout: float = 0.1,       
                 use_modal_attention: bool = False,
                 xafs_fusion_type: str = 'hierarchical',
                 device: str = 'cpu',
                 task_names: List[str] = ['formation_energy', 'fermi_energy', 'band_gap'],
                 use_xrd: bool = True,      
                 use_xafs: bool = True,
                 ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_activation = output_activation
        self.fusion_type = fusion_type
        self.use_modal_attention = use_modal_attention
        self.device = torch.device(device)
        self.task_names = task_names
        self.num_tasks = len(task_names)
        self.use_xrd = use_xrd          
        self.use_xafs = use_xafs

        self.xrd_dim_adjust_linear = None
        self.xafs_dim_adjust_linear = None
        
        # XRD分支
        if use_xrd:
            self.xrd_branch = XRDCNN(input_length=xrd_seq_len, 
                                    output_dim=hidden_dim, 
                                    dropout_fc=xrd_dropout_fc
                                ).to(self.device)
        else:
            self.xrd_branch = None
        
        # XAFS分支
        if use_xafs:
            self.xafs_processor = XAFSFeatureProcessor(descriptor_dims=descriptor_split_dims,
                                        hidden_dim=hidden_dim,
                                        dropout_rate=xafs_dropout_rate,
                                        fusion_dropout=xafs_fusion_dropout,
                                        fusion_type=xafs_fusion_type
                                    ).to(self.device)
        else:
            self.xafs_processor = None

        if fusion_type == 'cross_attention' and use_xrd and use_xafs:
            self.xrd_proj = nn.Linear(hidden_dim, hidden_dim).to(self.device)
            self.xafs_proj = nn.Linear(hidden_dim, hidden_dim).to(self.device)
            self.cross_attn_xrd2xafs = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=cross_attn_heads,
                dropout=cross_attn_dropout, batch_first=True
            ).to(self.device)
            self.cross_attn_xafs2xrd = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=cross_attn_heads,
                dropout=cross_attn_dropout, batch_first=True
            ).to(self.device)
            self.cross_attn_norm = nn.LayerNorm(hidden_dim).to(self.device)
            self.cross_attn_dropout = nn.Dropout(cross_attn_dropout)
        else:
            self.xrd_proj = None
            self.xafs_proj = None
            self.cross_attn_xrd2xafs = None
            self.cross_attn_xafs2xrd = None
            self.cross_attn_norm = None
            self.cross_attn_dropout = None
        
        fusion_input_dim = 0
        if use_xrd:
            fusion_input_dim += hidden_dim
        if use_xafs:
            fusion_input_dim += hidden_dim

        if (fusion_type == 'concat' or use_modal_attention) and use_xrd and use_xafs:
            self.modal_attention = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(modal_attention_dropout),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            ).to(self.device)
        else:
            self.modal_attention = None
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(fusion_layer_dropout),
            nn.Linear(hidden_dim, hidden_dim // 4)
        ).to(self.device)
        
        self.task_heads = nn.ModuleDict()
        for task_name in self.task_names:
            self.task_heads[task_name] = self._create_task_head(
                        hidden_dim // 4, task_head_dropout, task_name
                    ).to(self.device)
        
        self.band_gap_cls_head = None

        self.attention_weights_history = {
            'xrd_top_features': [],
            'xafs_top_descriptors': [],
            'modal_attention': [],
            'cross_attention_xrd2xafs': [],   
            'cross_attention_xafs2xrd': []      
        }
        
        print(f"模型初始化完成，融合方式: {fusion_type}")
        print(f"  使用 XRD: {use_xrd}, 使用 XAFS: {use_xafs}")
        print(f"  融合层输入维度: {fusion_input_dim}")
        print(f"  预测任务: {self.task_names}")
        print(f"  XAFS内部融合方式: {xafs_fusion_type}")
        self.to(self.device)
    
    def _create_task_head(self, input_dim: int, dropout_rate: float,
                          task_name: str = None) -> nn.Module:
        layers = [
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        ]

        return nn.Sequential(*layers)
    
    def forward(self, xrd_input, xafs_input):
        if xrd_input is not None:
            input_device = xrd_input.device
        elif xafs_input is not None:
            first_desc = next(iter(xafs_input.values()))
            input_device = first_desc.device
        else:
            raise ValueError("至少需要提供一个模态的输入")
        
        if next(self.parameters()).device != input_device:
            self.to(input_device)

        xrd_features = None
        xafs_features = None

        if self.use_xrd and xrd_input is not None:                     
            xrd_features = self.xrd_branch(xrd_input)
            if xrd_features.shape[1] != self.hidden_dim:
                if self.xrd_dim_adjust_linear is None:
                    self.xrd_dim_adjust_linear = nn.Linear(xrd_features.shape[1], self.hidden_dim).to(xrd_features.device)
                elif self.xrd_dim_adjust_linear.weight.device != xrd_features.device:
                    self.xrd_dim_adjust_linear = self.xrd_dim_adjust_linear.to(xrd_features.device)
                xrd_features = self.xrd_dim_adjust_linear(xrd_features)

        if self.use_xafs and xafs_input is not None:                   
            xafs_features = self.xafs_processor(xafs_input)
            if xafs_features.shape[1] != self.hidden_dim:
                if self.xafs_dim_adjust_linear is None:
                    self.xafs_dim_adjust_linear = nn.Linear(xafs_features.shape[1], self.hidden_dim).to(xafs_features.device)
                elif self.xafs_dim_adjust_linear.weight.device != xafs_features.device:
                    self.xafs_dim_adjust_linear = self.xafs_dim_adjust_linear.to(xafs_features.device)
                xafs_features = self.xafs_dim_adjust_linear(xafs_features)

        if xrd_features is None and xafs_features is None:
            raise RuntimeError("未能提取任何模态的特征，请检查输入数据或 use_xrd/use_xafs 设置")

        if self.fusion_type == 'cross_attention':
            if xrd_features is not None and xafs_features is not None:
                if self.xrd_proj.weight.device != xrd_features.device:
                    self.xrd_proj = self.xrd_proj.to(xrd_features.device)
                if self.xafs_proj.weight.device != xafs_features.device:
                    self.xafs_proj = self.xafs_proj.to(xafs_features.device)

                xrd_q = self.xrd_proj(xrd_features).unsqueeze(1)
                xafs_kv = self.xafs_proj(xafs_features).unsqueeze(1)
                xafs_q = self.xafs_proj(xafs_features).unsqueeze(1)
                xrd_kv = self.xrd_proj(xrd_features).unsqueeze(1)

                if next(self.cross_attn_xrd2xafs.parameters()).device != xrd_q.device:
                    self.cross_attn_xrd2xafs = self.cross_attn_xrd2xafs.to(xrd_q.device)
                if next(self.cross_attn_xafs2xrd.parameters()).device != xafs_q.device:
                    self.cross_attn_xafs2xrd = self.cross_attn_xafs2xrd.to(xafs_q.device)

                attn_output_xrd2xafs, attn_weights_xrd2xafs = self.cross_attn_xrd2xafs(
                    query=xrd_q, key=xafs_kv, value=xafs_kv)
                attn_output_xafs2xrd, attn_weights_xafs2xrd = self.cross_attn_xafs2xrd(
                    query=xafs_q, key=xrd_kv, value=xrd_kv)

                self.attention_weights_history['cross_attention_xrd2xafs'].append(attn_weights_xrd2xafs.detach().cpu())
                self.attention_weights_history['cross_attention_xafs2xrd'].append(attn_weights_xafs2xrd.detach().cpu())

                xrd_enhanced = xrd_features + self.cross_attn_dropout(attn_output_xrd2xafs.squeeze(1))
                xrd_enhanced = self.cross_attn_norm(xrd_enhanced)
                xafs_enhanced = xafs_features + self.cross_attn_dropout(attn_output_xafs2xrd.squeeze(1))
                xafs_enhanced = self.cross_attn_norm(xafs_enhanced)

                fused_concat = torch.cat([xrd_enhanced, xafs_enhanced], dim=1)

                if self.modal_attention is not None:
                    if next(self.modal_attention.parameters()).device != fused_concat.device:
                        self.modal_attention = self.modal_attention.to(fused_concat.device)
                    attention_weights = self.modal_attention(fused_concat)
                    weighted_xrd = xrd_enhanced * attention_weights[:, 0:1]
                    weighted_xafs = xafs_enhanced * attention_weights[:, 1:2]
                    fused_concat = torch.cat([weighted_xrd, weighted_xafs], dim=1)
                else:
                    attention_weights = torch.full((xrd_features.size(0), 2), 0.5, device=self.device)

                if next(self.fusion_layer.parameters()).device != fused_concat.device:
                    self.fusion_layer = self.fusion_layer.to(fused_concat.device)
                fused_features = self.fusion_layer(fused_concat)
                self.attention_weights_history['modal_attention'].append(attention_weights.detach().cpu())
            else:
                exist_features = []
                if xrd_features is not None:
                    exist_features.append(xrd_features)
                if xafs_features is not None:
                    exist_features.append(xafs_features)
                concat_features = torch.cat(exist_features, dim=1)
                if next(self.fusion_layer.parameters()).device != concat_features.device:
                    self.fusion_layer = self.fusion_layer.to(concat_features.device)
                fused_features = self.fusion_layer(concat_features)
                self.attention_weights_history['cross_attention_xrd2xafs'].append(None)
                self.attention_weights_history['cross_attention_xafs2xrd'].append(None)
                self.attention_weights_history['modal_attention'].append(None)

        else: 
            exist_features = []
            if xrd_features is not None:
                exist_features.append(xrd_features)
            if xafs_features is not None:
                exist_features.append(xafs_features)
            concat_features = torch.cat(exist_features, dim=1)

            expected_dim = self.hidden_dim * len(exist_features)
            if concat_features.shape[1] != expected_dim:
                if concat_features.shape[1] < expected_dim:
                    padding = expected_dim - concat_features.shape[1]
                    concat_features = F.pad(concat_features, (0, padding))
                else:
                    concat_features = concat_features[:, :expected_dim]

            if self.modal_attention is not None and xrd_features is not None and xafs_features is not None:  
                if next(self.modal_attention.parameters()).device != concat_features.device:
                    self.modal_attention = self.modal_attention.to(concat_features.device)
                attention_weights = self.modal_attention(concat_features)
                self.attention_weights_history['modal_attention'].append(attention_weights.detach().cpu())

                weighted_xrd = xrd_features * attention_weights[:, 0:1]
                weighted_xafs = xafs_features * attention_weights[:, 1:2]
                weighted_concat = torch.cat([weighted_xrd, weighted_xafs], dim=1)
            else:
                attention_weights = torch.full((concat_features.size(0), 2), 0.5, device=self.device) if xrd_features is not None and xafs_features is not None else torch.full((concat_features.size(0), 1), 1.0, device=self.device)
                weighted_concat = concat_features
                self.attention_weights_history['modal_attention'].append(None)

            if next(self.fusion_layer.parameters()).device != weighted_concat.device:
                self.fusion_layer = self.fusion_layer.to(weighted_concat.device)
            fused_features = self.fusion_layer(weighted_concat)

        outputs = {}
        for task_name in self.task_names:
            task_specific_features = fused_features

            pred = self.task_heads[task_name](task_specific_features)
            if self.output_activation == 'sigmoid':
                pred = torch.sigmoid(pred)
            elif self.output_activation == 'tanh':
                pred = torch.tanh(pred)
            outputs[task_name] = pred.squeeze(-1)

        outputs['modal_attention'] = attention_weights if 'attention_weights' in locals() else None
        outputs['xrd_features'] = xrd_features
        outputs['xafs_features'] = xafs_features
        outputs['fused_features'] = fused_features
        if self.fusion_type == 'cross_attention' and xrd_features is not None and xafs_features is not None:
            outputs['cross_attention_xrd2xafs'] = attn_weights_xrd2xafs if 'attn_weights_xrd2xafs' in locals() else None
            outputs['cross_attention_xafs2xrd'] = attn_weights_xafs2xrd if 'attn_weights_xafs2xrd' in locals() else None
        else:
            outputs['cross_attention_xrd2xafs'] = None
            outputs['cross_attention_xafs2xrd'] = None

        return outputs
    
    def analyze_attention_weights(self, xrd_input, xafs_input):
        self.eval()
        with torch.no_grad():
            outputs = self(xrd_input, xafs_input)
            modal_attention = outputs['modal_attention']  # [batch, 2]
            
            batch_size = modal_attention.shape[0]
            xrd_modal_mean = modal_attention[:, 0].mean().item()
            xafs_modal_mean = modal_attention[:, 1].mean().item()
            
            xrd_features = outputs['xrd_features']  # [batch, hidden_dim]
            xrd_feature_importance = torch.abs(xrd_features).mean(dim=0)  # [hidden_dim]
            xrd_total = xrd_feature_importance.sum()
            if xrd_total > 0:
                xrd_feature_importance_normalized = xrd_feature_importance / xrd_total
            else:
                xrd_feature_importance_normalized = xrd_feature_importance
            
            xrd_feature_importance_weighted = xrd_feature_importance_normalized * xrd_modal_mean
            xrd_features_dict = {}
            for idx in range(len(xrd_feature_importance_weighted)):
                weight = xrd_feature_importance_weighted[idx].item()
                if weight > 1e-6: 
                    xrd_features_dict[f'xrd_feature_{idx}'] = weight
            
            # 分析XAFS特征重要性
            xafs_features_dict = {}
            if hasattr(self.xafs_processor.processor, 'get_attention_weights'):
                attention_info = self.xafs_processor.processor.get_attention_weights(xafs_input)
                descriptor_importance = attention_info.get('descriptor_importance', {})
                if descriptor_importance:
                    xafs_descriptors_total = sum(descriptor_importance.values())
                    if xafs_descriptors_total > 0:
                        for desc_name, importance in descriptor_importance.items():
                            normalized = importance / xafs_descriptors_total
                            weighted = normalized * xafs_modal_mean
                            xafs_features_dict[desc_name] = weighted
            
            cross_attn_weights = None
            if self.fusion_type == 'cross_attention' and 'cross_attention_weights' in outputs:
                attn_xrd2xafs = outputs['cross_attention_xrd2xafs']
                attn_xafs2xrd = outputs['cross_attention_xafs2xrd']
                print(f"XRD->XAFS 注意力权重均值: {attn_xrd2xafs.mean().item():.4f}, 标准差: {attn_xrd2xafs.std().item():.4f}")
                print(f"XAFS->XRD 注意力权重均值: {attn_xafs2xrd.mean().item():.4f}, 标准差: {attn_xafs2xrd.std().item():.4f}")

            all_features = {**xrd_features_dict, **xafs_features_dict}
            sorted_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)
            top_10_features = sorted_features[:10]
            
            total_xrd_weight = sum(weight for name, weight in all_features.items() if name.startswith('xrd_feature_'))
            total_xafs_weight = sum(weight for name, weight in all_features.items() if not name.startswith('xrd_feature_'))
            total_weight = total_xrd_weight + total_xafs_weight
            if total_weight > 0 and abs(total_weight - (xrd_modal_mean + xafs_modal_mean)) > 0.01:
                scale_factor = (xrd_modal_mean + xafs_modal_mean) / total_weight
                top_10_features = [(name, weight * scale_factor) for name, weight in top_10_features]
            
            return {
                'top_10_features': top_10_features,  
                'modal_attention': {
                    'xrd_mean': xrd_modal_mean,
                    'xafs_mean': xafs_modal_mean,
                    'xrd_sum': xrd_modal_mean,  
                    'xafs_sum': xafs_modal_mean,
                    'total_sum': xrd_modal_mean + xafs_modal_mean},
                'weight_analysis': {
                    'total_xrd_weight': total_xrd_weight,
                    'total_xafs_weight': total_xafs_weight,
                    'total_weight': total_weight,
                    'xrd_modal_weight': xrd_modal_mean,
                    'xafs_modal_weight': xafs_modal_mean},
                'xrd_features_count': len(xrd_features_dict),
                'xafs_features_count': len(xafs_features_dict),
                'cross_attention_weights': cross_attn_weights
            }
        
    def get_detailed_feature_importance(self, xrd_input, xafs_input, task_name='formation_energy'):
        """
        Args:
            xrd_input: [1, seq_len]
            xafs_input: dict of [1, dim]
            task_name: 任务名称
        Returns:
            xrd_importance: dict {f'xrd_{idx}': weight}
            xafs_importance: dict {f'{desc_name}_{idx}': weight}
        """
        self.eval()
        xrd_importance = {}
        xafs_importance = {}

        if xrd_input is None and xafs_input is None:
            return xrd_importance, xafs_importance
        
        if xrd_input is not None:
            if xrd_input.requires_grad is False:
                xrd_input.requires_grad_(True)
            if xrd_input.dim() == 1:
                xrd_input = xrd_input.unsqueeze(0)
        
        if xafs_input is not None:
            for k in xafs_input:
                if xafs_input[k].requires_grad is False:
                    xafs_input[k].requires_grad_(True)
                if xafs_input[k].dim() == 1:
                    xafs_input[k] = xafs_input[k].unsqueeze(0)
        
        processor = self.xafs_processor.processor if self.xafs_processor is not None else None
        handles = []
        captured = {}
        
        def hook_fn(name):
            def hook(module, inp, out):
                captured[name] = out
            return hook
        
        if self.use_xafs and xafs_input is not None and processor is not None:
            for group_name in ['xafs_encoder', 'xanes_encoder', 'exafs_encoder']:
                encoder = getattr(processor, group_name, None)
                if encoder is not None and hasattr(encoder, 'encoders'):
                    for desc_name, enc in encoder.encoders.items():
                        last_linear = None
                        for layer in reversed(enc):
                            if isinstance(layer, nn.Linear):
                                last_linear = layer
                                break
                        if last_linear is not None:
                            handles.append(last_linear.register_forward_hook(hook_fn(desc_name)))
        
        outputs = self(xrd_input, xafs_input)
        pred = outputs[task_name].squeeze()   
        
        if xrd_input is not None and 'xrd_features' in outputs and outputs['xrd_features'] is not None:
            xrd_feat = outputs['xrd_features']    # [1, hidden_dim]
            if xrd_feat.requires_grad is False:
                xrd_feat.requires_grad_(True)
            try:
                grad_xrd = autograd.grad(pred, xrd_feat, retain_graph=True, allow_unused=True)[0]
                if grad_xrd is None:
                    raise RuntimeError("grad_xrd is None")
            except Exception as e:
                print(f"警告: 计算 XRD 梯度失败 ({e})，使用特征绝对值")
                grad_xrd = xrd_feat
            xrd_imp = grad_xrd.abs().squeeze(0).detach().cpu().numpy()
            if xrd_imp.sum() > 0:
                xrd_imp = xrd_imp / xrd_imp.sum()
            for idx, weight in enumerate(xrd_imp):
                if weight > 1e-6:
                    xrd_importance[f'xrd_{idx}'] = float(weight)

        if xafs_input is not None and captured:
            for desc_name, feat in captured.items():
                # feat shape: [1, output_dim]
                if not isinstance(feat, torch.Tensor):
                    continue
                if feat.requires_grad is False:
                    feat.requires_grad_(True)
                try:
                    grad = autograd.grad(pred, feat, retain_graph=True, allow_unused=True)[0]
                    if grad is None:
                        raise RuntimeError(f"grad for {desc_name} is None")
                except Exception as e:
                    print(f"警告: 计算 {desc_name} 梯度失败 ({e})，使用特征绝对值")
                    grad = feat
                imp = grad.abs().squeeze(0).detach().cpu().numpy()
                if imp.sum() > 0:
                    imp = imp / imp.sum()
                for dim_idx, weight in enumerate(imp):
                    if weight > 1e-6:
                        xafs_importance[f'{desc_name}_{dim_idx}'] = float(weight)
        for h in handles:
            h.remove()

        return xrd_importance, xafs_importance