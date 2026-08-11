import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from configurations import activation_map

class MLP(nn.Module):
    def __init__(self, input_size=3501):
        super(MLP, self).__init__()
        self.input_size = input_size
        
        print(f"MLP模型输入维度: {self.input_size}")
        mlp_hidden1 = 7000  
        mlp_hidden2 = 7000
        if self.input_size < 50:
            mlp_hidden1 = 512
            mlp_hidden2 = 256
        self.mlp_layers = nn.Sequential(
            nn.Linear(self.input_size, mlp_hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden1, mlp_hidden2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.flatten = nn.Flatten()
        fc_hidden = mlp_hidden2 // 2
        
        self.fc_layers = nn.Sequential(
            nn.Linear(mlp_hidden2, fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fc_hidden, 3), 
        )
        
        print(f"MLP模型结构:")
        print(f"  输入层: {self.input_size} -> {mlp_hidden1}")
        print(f"  隐藏层: {mlp_hidden1} -> {mlp_hidden2}")
        print(f"  输出层: {mlp_hidden2} -> {fc_hidden} -> 3")

    def forward(self, x):
        original_shape = x.shape
        
        if len(x.shape) == 3:
            if x.shape[1] == 1: 
                x = x.squeeze(1) 
            else:
                x = x.reshape(x.shape[0], -1)  
        elif len(x.shape) == 2:
            pass
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        if x.shape[1] != self.input_size:
            current_size = x.shape[1]
            
            if abs(current_size - self.input_size) / self.input_size < 0.5:  
                print(f"调整输入维度: {current_size} -> {self.input_size}")
                x = x.unsqueeze(1)  
                x = F.interpolate(
                    x, 
                    size=self.input_size, 
                    mode='linear', 
                    align_corners=False
                ).squeeze(1) 
            else:
                if current_size < self.input_size:
                    padding = self.input_size - current_size
                    x = F.pad(x, (0, padding), 'constant', 0)
                else:
                    x = x[:, :self.input_size]
        
        x = self.mlp_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x


class XASMLP(nn.Module):
    def __init__(
        self,
        input_size: int,      
        hidden_size: int = 256,    
        dropout: float = 0.1,       
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.activation_fn = nn.ReLU()

        if input_size < 50:
            hidden_size = min(hidden_size, 512)
            hidden_size2 = hidden_size // 2
        else:
            hidden_size2 = hidden_size

        self.mlp1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.mlp2 = nn.Linear(hidden_size, hidden_size2)
        self.dropout2 = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size2, 3)  
        self.relu = nn.ReLU()

        print(f"MLP模型结构:")
        print(f"  输入层: {input_size} -> {hidden_size}")
        print(f"  隐藏层: {hidden_size} → {hidden_size2}")
        print(f"  输出层: {hidden_size2} → 3")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            if x.shape[1] == 1:  
                x = x.squeeze(1)  
            else:
                x = x.view(x.shape[0], -1)
        
        if x.shape[1] != self.input_size:
            x = self._adjust_input_dimension(x)
        
        x = self.mlp1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.mlp2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.output(x)
        return x
    
    def _adjust_input_dimension(self, x: torch.Tensor) -> torch.Tensor:
        current_size = x.shape[1]
        
        if current_size == self.input_size:
            return x
        
        if abs(current_size - self.input_size) / self.input_size < 0.5:
            print(f"调整输入维度: {current_size} → {self.input_size}")
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
            x = F.interpolate(
                x, 
                size=self.input_size, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)  # [batch, seq_len]
        else:
            if current_size < self.input_size:
                padding = self.input_size - current_size
                x = F.pad(x, (0, padding), 'constant', 0)
            else:
                x = x[:, :self.input_size]
        
        return x
    

class XASCNN(nn.Module):
    def __init__(
        self,
        input_size: int,           
        hidden_size: int = 256,    
        dropout: float = 0.2,
        num_conv_layers: int = 3,   
        out_channel: int = 32,      
        channel_mul: int = 2,       
        kernel_size: int = 3,       
        stride: int = 1             
    ) -> None:
        super().__init__()
        self.input_size = input_size
        conv_outputs = []
        current_channels = out_channel
        seq_length = input_size
        for i in range(num_conv_layers):
            seq_length = (seq_length - kernel_size) // stride + 1
            conv_outputs.append((seq_length, current_channels))
            current_channels *= channel_mul
        
        final_seq_length, final_channels = conv_outputs[-1]
        conv_output_size = final_seq_length * final_channels
        
        fc_hidden1 = conv_output_size // 2
        if fc_hidden1 > hidden_size * 2:
            fc_hidden1 = hidden_size * 2
        fc_hidden2 = hidden_size
        
        conv_layers = []
        in_channels = 1  
        current_channels = out_channel
        
        for i in range(num_conv_layers):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels, current_channels, kernel_size, stride),
                nn.BatchNorm1d(current_channels),
                nn.PReLU(),
                nn.Dropout(dropout)
            )
            conv_layers.append(conv_block)
            
            in_channels = current_channels
            current_channels *= channel_mul
        
        self.conv_encoder = nn.Sequential(*conv_layers)
        self.dense_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_output_size, fc_hidden1),
            nn.PReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.PReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(fc_hidden2, 3)  
        )
        print(f"XASCNN模型结构:")
        print(f"  输入层: [batch, 1, {input_size}]")
        for i, (seq_len, channels) in enumerate(conv_outputs):
            if i == 0:
                print(f"  卷积层{i+1}: 1 → {channels}通道, 序列长度: {input_size} → {seq_len}")
            else:
                prev_channels = out_channel * (channel_mul ** (i-1))
                print(f"  卷积层{i+1}: {prev_channels} → {channels}通道, 序列长度: {conv_outputs[i-1][0]} → {seq_len}")
        print(f"  全连接层1: {conv_output_size:,} → {fc_hidden1}")
        print(f"  全连接层2: {fc_hidden1} → {fc_hidden2}")
        print(f"  输出层: {fc_hidden2} → 3")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  总参数: {total_params:,}")

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:  
            x = x.unsqueeze(1) 
        elif len(x.shape) == 3 and x.shape[1] != 1:  
            if x.shape[1] > 1:
                x = x[:, 0:1, :]
        
        seq_len = x.shape[2]
        if seq_len != self.input_size:
            x = self._adjust_sequence_length(x, seq_len)

        x = self.conv_encoder(x)
        x = self.dense_predictor(x)
        
        return x
    
    def _adjust_sequence_length(self, x: torch.Tensor, current_len: int) -> torch.Tensor:
        target_len = self.input_size
        
        if current_len == target_len:
            return x
        
        if abs(current_len - target_len) / target_len < 0.3:
            x = F.interpolate(
                x, 
                size=target_len, 
                mode='linear', 
                align_corners=False
            )
        else:
            if current_len < target_len:
                padding = target_len - current_len
                x = F.pad(x, (0, padding), 'constant', 0)
            else:
                x = x[:, :, :target_len]
        
        return x
    

class XASLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,               
        hidden_size: int = 256,        
        lstm_layers: int = 3,          
        dropout: float = 0.2,
        bidirectional: bool = True,    
        dense_hidden_size: int = 128   
    ) -> None:
        
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=lstm_layers,
                        bidirectional=bidirectional,
                        batch_first=True,
                        dropout=dropout if lstm_layers > 1 else 0
                    )
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.PReLU(),
            nn.Linear(lstm_output_size // 2, 1)
        )
    
        self.dense_predictor = nn.Sequential(
            nn.Linear(lstm_output_size, dense_hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(dense_hidden_size, dense_hidden_size // 2),
            nn.PReLU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(dense_hidden_size // 2, 3)
        )
        
        print("LSTM模型结构:")
        print(f"  输入层: 序列长度 × {input_size}")
        print(f"  注意力层: {lstm_output_size} → {lstm_output_size//2} → 1")
        print(f"  全连接层: {lstm_output_size} → {dense_hidden_size} → {dense_hidden_size//2}")
        print(f"  输出层: {dense_hidden_size//2} → 3")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2: 
            x = x.unsqueeze(1) 
        elif len(x.shape) == 3 and x.shape[1] == 1: 
            x = x.transpose(1, 2)
        
        if x.shape[2] != self.input_size:
            x = self._adjust_feature_dimension(x)
        lstm_out, (hidden, cell) = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(lstm_out * attention_weights, dim=1)
        output = self.dense_predictor(context)
        
        return output
    
    def _adjust_feature_dimension(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, current_features = x.shape
        target_features = self.input_size
        
        if current_features == target_features:
            return x
        
        if abs(current_features - target_features) / target_features < 0.3:
            x = x.transpose(1, 2)
            x = nn.functional.interpolate(
                x, 
                size=target_features, 
                mode='linear', 
                align_corners=False
            )
            x = x.transpose(1, 2)
        else:
            if current_features < target_features:
                padding = target_features - current_features
                x = nn.functional.pad(x, (0, padding), 'constant', 0)
            else:
                x = x[:, :, :target_features]
        
        return x


class XASAECNN(nn.Module):
    def __init__(
        self,
        input_size: int,           
        hidden_size: int = 256,    
        dropout: float = 0.2,
        num_conv_layers: int = 3,  
        out_channel: int = 32,     
        channel_mul: int = 2,      
        kernel_size: int = 3,      
        stride: int = 1            
    ) -> None:
        super().__init__()
        self.input_size = input_size
        encoder_layers = []
        in_channels = 1
        current_channels = out_channel
        conv_shapes = [input_size]
        
        for i in range(num_conv_layers):
            conv_shape = ((conv_shapes[-1] - kernel_size) // stride) + 1
            conv_shapes.append(conv_shape)
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, current_channels, kernel_size, stride),
                    nn.PReLU()
                )
            )
            
            in_channels = current_channels
            current_channels *= channel_mul
        
        self.encoder = nn.Sequential(*encoder_layers)
        encoder_output_size = out_channel * (channel_mul ** (num_conv_layers - 1)) * conv_shapes[-1]
        fc_hidden1 = encoder_output_size // 4
        if fc_hidden1 > hidden_size * 2:
            fc_hidden1 = hidden_size * 2
        fc_hidden2 = hidden_size
        
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_output_size, fc_hidden1),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden1, fc_hidden2),
            nn.PReLU(),
            nn.Dropout(dropout * 0.8),
            nn.Linear(fc_hidden2, 3) 
        )
        
        decoder_layers = []
        dec_in_channels = out_channel * (channel_mul ** (num_conv_layers - 1))
        dec_out_channels = out_channel * (channel_mul ** (num_conv_layers - 2))
        
        for i in range(num_conv_layers):
            tconv_shape = (conv_shapes[num_conv_layers - i - 1] - 1) * stride + kernel_size
            if i == num_conv_layers - 1:
                dec_out_channels = 1
            target_shape = conv_shapes[num_conv_layers - i - 1]
            padding = 0
            output_padding = 0
            
            if tconv_shape > target_shape:
                padding = tconv_shape - target_shape
            elif tconv_shape < target_shape:
                output_padding = target_shape - tconv_shape
            
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        dec_in_channels,
                        dec_out_channels,
                        kernel_size,
                        stride,
                        output_padding=output_padding
                    ),
                    nn.PReLU()
                )
            )
            
            if i < num_conv_layers - 1:
                dec_in_channels = dec_out_channels
                dec_out_channels //= channel_mul
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print("Auto-CNN模型结构:")
        print(f"  输入层: {input_size}")
        print(f"  编码器卷积层: {out_channel} → {fc_hidden1} → {fc_hidden2}")
        print(f"  输出层: {fc_hidden2} → 3")
        print(f"  解码器反卷积层: {num_conv_layers}层")
    
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)  
        
        seq_len = x.shape[2]
        if seq_len != self.input_size:
            x = self._adjust_sequence_length(x, seq_len)

        encoded = self.encoder(x)
        pred = self.predictor(encoded)
   
        return pred 
        

    def predict_only(self, x: torch.Tensor) -> torch.Tensor:
       
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)
        
        seq_len = x.shape[2]
        if seq_len != self.input_size:
            x = self._adjust_sequence_length(x, seq_len)
        
        encoded = self.encoder(x)
        pred = self.predictor(encoded)
        return pred
    
    def reconstruct_only(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 2:  
            x = x.unsqueeze(1)
        
        seq_len = x.shape[2]
        if seq_len != self.input_size:
            x = self._adjust_sequence_length(x, seq_len)
        
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        recon = recon.squeeze(1)
        return recon
    
    def _adjust_sequence_length(self, x: torch.Tensor, current_len: int) -> torch.Tensor:
        target_len = self.input_size
        
        if current_len == target_len:
            return x
        
        if abs(current_len - target_len) / target_len < 0.3:
            x = nn.functional.interpolate(
                x, 
                size=target_len, 
                mode='linear', 
                align_corners=False
            )
        else:
            if current_len < target_len:
                padding = target_len - current_len
                x = nn.functional.pad(x, (0, padding), 'constant', 0)
            else:
                x = x[:, :, :target_len]
        
        return x


class XASAEMLP(nn.Module):
    def __init__(
        self,
        input_size: int,                   
        hidden_size: int = 256,            
        dropout: float = 0.2,              
        num_encoder_layers: int = 3,       
        shrink_rate: float = 1.0,          
        activation: str = "relu",          
        latent_factor: float = 0.5,        
    ):
        super().__init__()
        if input_size < 50:
            hidden_size = min(hidden_size, 256)
        self.input_size = input_size
        self.output_size = 3  
        act_fn = activation_map.get(activation.lower(), nn.ReLU())
        enc_layers = []
        current_size = input_size
        
        for i in range(num_encoder_layers):
            next_size = int(hidden_size * (shrink_rate ** i))
            if next_size < 8:  
                next_size = 8
            
            enc_layers.append(nn.Linear(current_size, next_size))
            enc_layers.append(act_fn)
            enc_layers.append(nn.Dropout(dropout))
            current_size = next_size
        
        latent_size = max(int(input_size * latent_factor), 16)
        enc_layers.append(nn.Linear(current_size, latent_size))
        enc_layers.append(act_fn)
        
        self.encoder = nn.Sequential(*enc_layers)
        pred_layers = []
        pred_current_size = latent_size
        pred_hidden_size = hidden_size // 2 if input_size < 50 else hidden_size
        pred_layers.append(nn.Linear(pred_current_size, pred_hidden_size))
        pred_layers.append(act_fn)
        pred_layers.append(nn.Dropout(dropout))
        
        pred_layers.append(nn.Linear(pred_hidden_size, pred_hidden_size // 2))
        pred_layers.append(act_fn)
        pred_layers.append(nn.Dropout(dropout))
        pred_layers.append(nn.Linear(pred_hidden_size // 2, 3))
        self.predictor = nn.Sequential(*pred_layers)
        
        dec_layers = []
        dec_current_size = latent_size
        for i in range(num_encoder_layers):
            size = int(hidden_size * (shrink_rate ** (num_encoder_layers - i - 1)))
            if size < 8:
                size = 8
            
            dec_layers.append(nn.Linear(dec_current_size, size))
            dec_layers.append(act_fn)
            dec_layers.append(nn.Dropout(dropout))
            dec_current_size = size
        dec_layers.append(nn.Linear(dec_current_size, input_size))
        self.decoder = nn.Sequential(*dec_layers)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)
    
        if x.shape[1] != self.input_size:
            x = self._adjust_input_dimension(x)
        
        encoded = self.encoder(x)
        predictions = self.predictor(encoded)
        reconstructed = self.decoder(encoded)
        
        return predictions
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.view(x.shape[0], -1)

        if x.shape[1] != self.input_size:
            x = self._adjust_input_dimension(x)
        
        encoded = self.encoder(x)
        return self.predictor(encoded)
    
    def _adjust_input_dimension(self, x: torch.Tensor) -> torch.Tensor:
        current_size = x.shape[1]
        
        if current_size == self.input_size:
            return x
        
        print(f"调整输入维度: {current_size} → {self.input_size}")
        
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        x = F.adaptive_avg_pool1d(x, self.input_size)
        return x.squeeze(1)
    

class MultiHeadCNN(nn.Module):

    def __init__(
        self,
        input_size: int,                   
        num_heads: int = 2,                
        hidden_size: int = 256,           
        dropout: float = 0.2,              
        num_conv_layers: int = 3,          
        activation: str = "silu",          
        out_channels: int = 32,            
        channel_multiplier: int = 2,        
        kernel_size: int = 3,               
        stride: int = 1,                  
        head_num_hidden_layers: int = 2,   
        head_hidden_size: int = 512,       
        head_shrink_rate: float = 1.0,     
        output_operation: str = "mean",    
    ):
        super().__init__()
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_properties = 3 
        self.output_operation = output_operation
        
        if input_size < 50:
            out_channels = min(out_channels, 16)
            head_hidden_size = min(head_hidden_size, 256)
        
        conv_layers = []
        in_channels = 1  
        current_channels = out_channels
        
        for i in range(num_conv_layers):
            conv_layer = nn.Conv1d(
                in_channels, 
                current_channels, 
                kernel_size, 
                stride,
                padding=kernel_size//2  
            )
            
            layer_components = [
                conv_layer,
                nn.BatchNorm1d(current_channels),
                activation_map.get(activation.lower(), nn.SiLU()),
                nn.Dropout(p=dropout)
            ]
            
            if i < num_conv_layers - 1:
                pool_size = 2 if input_size >= 200 else 1
                if pool_size > 1:
                    layer_components.append(nn.MaxPool1d(pool_size))
            
            conv_layers.append(nn.Sequential(*layer_components))
            
            in_channels = current_channels
            current_channels *= channel_multiplier
        
        self.conv_layers = nn.Sequential(*conv_layers)
        conv_output_size = self._get_conv_output_size(input_size)
        self.flatten_size = in_channels * conv_output_size

        print("MultiHeadCNN模型结构:")
        print(f"  输入层: 1×{input_size} (单通道)")
        print(f"  卷积层: {in_channels} → {current_channels}, kernel={kernel_size}")
        print(f"  输出层: {in_channels}×{conv_output_size}")
        print(f"  展平层: {in_channels}×{conv_output_size} → {self.flatten_size}")

        self.heads = nn.ModuleList([
            CNNHead(
                in_size=self.flatten_size,
                out_size=self.num_properties,  
                hidden_size=head_hidden_size,
                dropout=dropout,
                num_hidden_layers=head_num_hidden_layers,
                shrink_rate=head_shrink_rate,
                activation=activation
            )
            for _ in range(self.num_heads)
        ])
    
    def _get_conv_output_size(self, in_size: int) -> int:
        dummy_input = torch.randn(1, 1, in_size)
        
        with torch.no_grad():
            output = self.conv_layers(dummy_input)
        
        return output.shape[2] 
    
    
    def forward(self, x: torch.Tensor, active_head_idx: int = None) -> torch.Tensor:
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] != 1:
            x = x.view(x.shape[0], -1).unsqueeze(1)
        
        if x.shape[2] != self.input_size:
            x = self._adjust_input_dimension(x)
    
        x = self.conv_layers(x)
        batch_size = x.shape[0]
        x = torch.flatten(x, 1)
        
        if x.shape[1] != self.flatten_size:
            x = F.adaptive_avg_pool1d(
                x.unsqueeze(1), 
                self.flatten_size
            ).squeeze(1)
        if active_head_idx is not None:
            if active_head_idx >= 0 and active_head_idx < self.num_heads:
                return self.heads[active_head_idx](x)

        head_outputs = []
        for head in self.heads:
            output = head(x)  # [batch_size, 3]
            head_outputs.append(output)
        
        if self.output_operation == "stack": # [num_heads, batch_size, 3]
            return torch.stack(head_outputs, dim=0)
        elif self.output_operation == "concat": # [batch_size, num_heads * 3]
            return torch.cat(head_outputs, dim=1)
        elif self.output_operation == "mean": # [batch_size, 3]
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            return torch.mean(stacked, dim=0)
        elif self.output_operation == "max":  # [batch_size, 3]
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            return torch.max(stacked, dim=0)[0]
        elif self.output_operation == "min": # [batch_size, 3]
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            return torch.min(stacked, dim=0)[0]
        else:
            raise ValueError(f"Unknown output_operation: {self.output_operation}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, active_head_idx=None)
    
    def get_head_predictions(self, x: torch.Tensor) -> torch.Tensor:
        original_op = self.output_operation
        self.output_operation = "stack"
        result = self.forward(x, active_head_idx=None)
        self.output_operation = original_op
        
        return result
    
    def _adjust_input_dimension(self, x: torch.Tensor) -> torch.Tensor:
        current_size = x.shape[2] 
        
        if current_size == self.input_size:
            return x
        
        print(f"调整输入维度: {current_size} → {self.input_size}")
        x = F.adaptive_avg_pool1d(x, self.input_size)
        return x

class CNNHead(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int = 3,  
        hidden_size: int = 512,
        dropout: float = 0.1,
        num_hidden_layers: int = 2,
        shrink_rate: float = 1.0,
        activation: str = "silu",
    ):
        super().__init__()
        
        act_fn = activation_map.get(activation.lower(), nn.SiLU())
        layers = []
        current_size = in_size
        
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate ** i))
            if next_size < 8: 
                next_size = 8
            
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.BatchNorm1d(next_size))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            current_size = next_size
        
        layers.append(nn.Linear(current_size, out_size))
        layers.append(nn.Softplus())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class MultiHeadMLP(nn.Module):
    def __init__(
        self,
        input_size: int,                 
        num_heads: int = 2,               
        hidden_size: int = 512,           
        dropout: float = 0.2,              
        num_hidden_layers: int = 3,        
        shrink_rate: float = 1.0,          
        activation: str = "silu",          
        head_num_hidden_layers: int = 2,    
        head_hidden_size: int = 512,        
        head_shrink_rate: float = 1.0,      
        output_operation: str = "mean",    
        use_batchnorm: bool = True,        
    ):

        super().__init__()
        
        self.input_size = input_size
        self.num_heads = num_heads
        self.num_properties = 3 
        self.output_operation = output_operation
        
        if input_size < 50:
            hidden_size = min(hidden_size, 256)
            head_hidden_size = min(head_hidden_size, 256)
            shrink_rate = max(shrink_rate, 0.8)  
        
        act_fn = activation_map.get(activation.lower(), nn.SiLU())
        dense_layers = []
        current_size = input_size
        shared_layers_info = [f"{input_size}"]
        
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate ** i))
            if next_size < 8: 
                next_size = 8
            
            dense_layers.append(nn.Linear(current_size, next_size))
            
            if use_batchnorm:
                dense_layers.append(nn.BatchNorm1d(next_size))
            
            dense_layers.append(act_fn)
            dense_layers.append(nn.Dropout(dropout))
            
            shared_layers_info.append(f"{next_size}")
            current_size = next_size
        
        self.dense_layers = nn.Sequential(*dense_layers)
        self.shared_output_size = current_size
    
        self.heads = nn.ModuleList([
            MLPHead(
                in_size=self.shared_output_size,
                out_size=self.num_properties,  
                hidden_size=head_hidden_size,
                dropout=dropout,
                num_hidden_layers=head_num_hidden_layers,
                shrink_rate=head_shrink_rate,
                activation=activation,
                use_batchnorm=use_batchnorm
            )
            for _ in range(self.num_heads)
        ])
    
    def forward(self, x: torch.Tensor, active_head_idx: int = None) -> torch.Tensor:
        if len(x.shape) == 3:
            if x.shape[1] == 1:
                x = x.squeeze(1) 
            else:
                x = x.view(x.shape[0], -1)
        
        if x.shape[1] != self.input_size:
            x = self._adjust_input_dimension(x)
        
        shared = self.dense_layers(x)
        if active_head_idx is not None:
            if active_head_idx >= 0 or active_head_idx < self.num_heads:
                return self.heads[active_head_idx](shared)
        
        head_outputs = []
        for head in self.heads:
            output = head(shared)  # [batch_size, 3]
            head_outputs.append(output)
        
        return self._combine_head_outputs(head_outputs)
    
    def _combine_head_outputs(self, head_outputs: List[torch.Tensor]) -> torch.Tensor:
        if self.output_operation == "stack":
            return torch.stack(head_outputs, dim=0)
        elif self.output_operation == "concat":
            return torch.cat(head_outputs, dim=1)
        elif self.output_operation == "mean":
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            return torch.mean(stacked, dim=0)
        elif self.output_operation == "max":
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            return torch.max(stacked, dim=0)[0]
        elif self.output_operation == "min":
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            return torch.min(stacked, dim=0)[0]
        elif self.output_operation == "weighted_mean":
            if not hasattr(self, 'head_weights'):
                self.head_weights = nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
            
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, 3]
            weights = F.softmax(self.head_weights, dim=0).view(-1, 1, 1)
            return torch.sum(stacked * weights, dim=0)
        
        else:
            raise ValueError(f"Unknown output_operation: {self.output_operation}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, active_head_idx=None)
    
    def get_head_predictions(self, x: torch.Tensor) -> torch.Tensor:
        original_op = self.output_operation
        self.output_operation = "stack"
        result = self.forward(x, active_head_idx=None)
        self.output_operation = original_op
        
        return result
    
    def get_head_weights(self) -> torch.Tensor:
        if hasattr(self, 'head_weights'):
            return F.softmax(self.head_weights, dim=0)
        else:
            return torch.ones(self.num_heads) / self.num_heads
    
    def _adjust_input_dimension(self, x: torch.Tensor) -> torch.Tensor:
        current_size = x.shape[1]
        if current_size == self.input_size:
            return x
        
        print(f"调整输入维度: {current_size} → {self.input_size}")
        
        if abs(current_size - self.input_size) / self.input_size < 0.5:
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
            x = F.interpolate(
                x, 
                size=self.input_size, 
                mode='linear', 
                align_corners=False
            ).squeeze(1)  # [batch, seq_len]
        else:
            x = x.unsqueeze(1)  # [batch, 1, seq_len]
            x = F.adaptive_avg_pool1d(x, self.input_size)
            x = x.squeeze(1)
        
        return x

class MLPHead(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int = 3,  
        hidden_size: int = 512,
        dropout: float = 0.1,
        num_hidden_layers: int = 2,
        shrink_rate: float = 1.0,
        activation: str = "silu",
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        act_fn = activation_map.get(activation.lower(), nn.SiLU())
        
        layers = []
        current_size = in_size
        
        for i in range(num_hidden_layers):
            next_size = int(hidden_size * (shrink_rate ** i))
            if next_size < 8:  
                next_size = 8
            
            layers.append(nn.Linear(current_size, next_size))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(next_size))
            
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            current_size = next_size

        layers.append(nn.Linear(current_size, out_size))
        layers.append(nn.Softplus())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class XASMHTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,                  
        num_heads: int = 2,                
        hidden_size: int = 128,           
        dropout: float = 0.1,             
        n_attention_heads: int = 8,       
        n_self_attn_layers: int = 2,       
        n_cross_attn_layers: int = 3,      
        num_properties: int = 3,           
        output_operation: str = "mean",    
    ):

        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_properties = num_properties
        self.output_operation = output_operation
        self.attn_weights = [] 
        self.ln_input = nn.LayerNorm(input_size)
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.self_blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_size, n_attention_heads, dropout)
            for _ in range(n_self_attn_layers)
        ])
        self.property_embedding = nn.Parameter(
            torch.randn(num_properties, hidden_size)
        )
        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(hidden_size, n_attention_heads, dropout)
            for _ in range(n_cross_attn_layers)
        ])
        self.property_self_attn = SelfAttentionBlock(hidden_size, n_attention_heads, dropout)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_properties),
                nn.Softplus()  
            )
            for _ in range(num_heads)
        ])
        print("TransformerMultiHead模型结构:")
        print(f"  输入层: {input_size} → {hidden_size}")
        print(f"  隐藏层维度: {hidden_size}")
        print(f"  输出层: {hidden_size} → {num_properties}")
    
    def forward(self, x: torch.Tensor, active_head_idx: int = None) -> torch.Tensor:
        batch_size = x.shape[0]
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        seq_len = x.shape[1]
        feature_dim = x.shape[2]
        if feature_dim != self.input_size:
            x = self._adjust_input_dimension(x)

        x = self.ln_input(x)
        x = self.input_proj(x) 
        for block in self.self_blocks:
            x = block(x, key_padding_mask=None)
        
        context = x
        property_emb = self.property_embedding.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_properties, hidden_size]
        
        x = property_emb
        attn_weights_list = []
        
        for block in self.cross_blocks:
            x, attn = block(query=x, context=context, key_padding_mask=None)
            attn_weights_list.append(attn)
        
        self.attn_weights = attn_weights_list
        x = self.property_self_attn(x)
        x = self.final_norm(x)
        x = self.final_mlp(x) + x
        if active_head_idx is not None:
            if active_head_idx >= 0 and active_head_idx < self.num_heads:
                head = self.prediction_heads[active_head_idx]
                output = head(x)  
                return self._extract_predictions(output)
        
        head_outputs = []
        for head in self.prediction_heads:
            output = head(x)  # [batch, num_properties, num_properties]
            predictions = self._extract_predictions(output)  # [batch, num_properties]
            head_outputs.append(predictions)
        
        return self._combine_head_outputs(head_outputs)
    
    def _extract_predictions(self, head_output: torch.Tensor) -> torch.Tensor:
        batch_size = head_output.shape[0]
        predictions = torch.stack([
            head_output[:, i, i] for i in range(self.num_properties)
        ], dim=1)
        return predictions
    
    def _combine_head_outputs(self, head_outputs: list) -> torch.Tensor:
        if self.output_operation == "stack":
            return torch.stack(head_outputs, dim=0)
        elif self.output_operation == "concat":
            return torch.cat(head_outputs, dim=1)
        elif self.output_operation == "mean":
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, num_properties]
            return torch.mean(stacked, dim=0)
        elif self.output_operation == "max":
            stacked = torch.stack(head_outputs, dim=0)  # [num_heads, batch, num_properties]
            return torch.max(stacked, dim=0)[0]
        else:
            raise ValueError(f"Unknown output_operation: {self.output_operation}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, active_head_idx=None)
    
    def get_attn_weights(self):
        return self.attn_weights
    
    def _adjust_input_dimension(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, current_size = x.shape
        
        if current_size == self.input_size:
            return x
        
        print(f"调整输入维度: {current_size} → {self.input_size}")
        x_reshaped = x.view(batch_size * seq_len, current_size).unsqueeze(1)
        x_adjusted = F.adaptive_avg_pool1d(x_reshaped, self.input_size).squeeze(1)

        return x_adjusted.view(batch_size, seq_len, self.input_size)

class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, query, context, key_padding_mask=None):
        attn_out, attn_weights = self.cross_attn(
            query=query, key=context, value=context, key_padding_mask=key_padding_mask
        )
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights

       
        

