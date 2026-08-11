

import warnings
warnings.filterwarnings("ignore")


import json
import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from configurations import xafs_group, xanes_group, exafs_group, simple_feature_names, group_feat_map

class PropertyPredictorTrainer:
    """属性预测训练器"""
    
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        self.save_dir = f"res/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"结果保存路径: {self.save_dir}")
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=15, 
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader, graph_builder=None):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            if graph_builder:
                graph_batch = graph_builder.batch_spectra(data)
                output = self.model(graph_batch)
            else:
                output = self.model(data)

            self.optimizer.zero_grad()
            
            loss = self.criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / max(batch_count, 1)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                all_predictions.append(output.cpu().numpy())
                all_targets.append(target.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        return avg_loss, all_predictions, all_targets
    
    def train(self, train_loader, val_loader, file, epochs=200, patience=50):
        print("开始训练模型...")
        print("=" * 60)

        graph_builder = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, graph_builder)
            self.train_losses.append(train_loss)
            val_loss, val_preds, val_targets = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.scheduler.step(val_loss)
            
            val_r2_scores = []
            for i in range(3):
                r2 = r2_score(val_targets[:, i], val_preds[:, i])
                val_r2_scores.append(r2)
        
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}:")
                print(f"  验证R² - 形成能: {val_r2_scores[0]:.4f}, "
                      f"费米能: {val_r2_scores[1]:.4f}, 能带: {val_r2_scores[2]:.4f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_r2_scores': val_r2_scores}, file)
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n早停触发！在epoch {epoch+1}")
                    break
        
        print(f"\n训练完成！最佳验证损失: {self.best_val_loss:.6f}")

        checkpoint = torch.load(file)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return self.train_losses, self.val_losses
    
    def evaluate(self, test_loader, dataset):
        print("\n" + "="*60)
        print("在测试集上评估模型...")
        print("="*60)
        
        test_loss, test_preds, test_targets = self.validate(test_loader)
        
        test_preds_original = dataset.denormalize_properties(test_preds)
        test_targets_original = dataset.denormalize_properties(test_targets)
        
        property_names = ['形成能', '费米能', '能带']
        units = ['eV', 'eV', 'eV']
        
        results = {}
        
        for i in range(3):
            pred_norm_i = test_preds[:, i]
            target_norm_i = test_targets[:, i]
            mae_norm = mean_absolute_error(target_norm_i, pred_norm_i)
            rmse_norm = np.sqrt(mean_squared_error(target_norm_i, pred_norm_i))

            pred_i = test_preds_original[:, i]
            target_i = test_targets_original[:, i]
            r2 = r2_score(target_i, pred_i)
            mae_original = mean_absolute_error(target_i, pred_i)
            rmse_original = np.sqrt(mean_squared_error(target_i, pred_i))

            scaling_factor = 2.0  
            mae_normalized = mae_norm / scaling_factor
            rmse_normalized = rmse_norm / scaling_factor
            
            mae_normalized = np.clip(mae_normalized, 0, 1)
            rmse_normalized = np.clip(rmse_normalized, 0, 1)
            
            results[property_names[i]] = {
                'MAE_original': mae_original,
                'RMSE_original': rmse_original,
                'MAE_normalized': mae_normalized,
                'RMSE_normalized': rmse_normalized,
                'R2': r2
            }
            print(f"\n{property_names[i]} ({units[i]}):")
            print(f"  MAE\tRMSE\t MAE_norm\tRMSE_norm\tR²")
            print(f"  {mae_original:.4f}\t{rmse_original:.4f}\t{mae_normalized:.4f}\t{rmse_normalized:.4f}\t{r2:.4f}")
        
        return test_preds_original, test_targets_original, results
 
    
    def plot_predictions_vs_targets(self, predictions, targets, dataset):
        property_names = ['formation energy', 'fermi energy', 'band gap']
        units = ['eV', 'eV', 'eV']
        
        for i in range(3):
            pred_i = predictions[:, i]
            target_i = targets[:, i]
            
            r2 = r2_score(target_i, pred_i)
            mae = mean_absolute_error(target_i, pred_i)
            rmse = np.sqrt(mean_squared_error(target_i, pred_i))
            
            fig, ax = plt.subplots(figsize=(6, 5))                    
            ax.scatter(target_i, pred_i, alpha=0.6, s=50, edgecolors='w')
            
            min_val = min(target_i.min(), pred_i.min())
            max_val = max(target_i.max(), pred_i.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            prop_name_clean = property_names[i].replace(' ', '_')    
            ax.set_xlabel(f'真实 {property_names[i]} ({units[i]})')
            ax.set_ylabel(f'预测 {property_names[i]} ({units[i]})')
            ax.set_title(f'{property_names[i]} 预测 (R² = {r2:.3f})')
            ax.grid(True, alpha=0.3)
            
            ax.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}', 
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            png_path = f'{self.save_dir}/_{prop_name_clean}_predictions_vs_targets.png'
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)       
            
            txt_path = f'{self.save_dir}/_{prop_name_clean}_targets_predictions.txt'
            np.savetxt(txt_path, np.column_stack((target_i, pred_i)), 
                    header='True\tPredicted', delimiter='\t', fmt='%.6f')
            
            print(f"已保存: {png_path}")   
            print(f"已保存: {txt_path}")
        
        print("所有属性的散点图和数值文件已生成")     

class MultiTaskAttentionAnalyzer:
    def __init__(self, model, descriptor_names=None):
        self.model = model
        self.descriptor_names = descriptor_names if descriptor_names else []
    
    def analyze_task_attention(self, xrd_input, xafs_input, task_name):
        self.model.eval()
        if xrd_input is None and xafs_input is None:
            return {
                'task_name': task_name,
                'xrd_features': {},
                'xafs_features': {},
            }
        
        with torch.set_grad_enabled(True):
            if xrd_input is not None and xrd_input.requires_grad is False:
                xrd_input.requires_grad_(True)
            if xafs_input is not None:
                for k in xafs_input:
                    if xafs_input[k].requires_grad is False:
                        xafs_input[k].requires_grad_(True)

            outputs = self.model(xrd_input, xafs_input)
            task_output = outputs[task_name]
            self.model.zero_grad()
            task_output.sum().backward(retain_graph=False)  
            analysis_results = {
                'task_name': task_name,
                'xrd_features': {},
                'xafs_features': {},
            }
            
            if xrd_input is not None and xrd_input.grad is not None:
                xrd_grad = xrd_input.grad.abs()
                xrd_grad_mean = xrd_grad.mean(dim=0)
                
                n_features = min(len(xrd_grad_mean), 256)
                for i in range(n_features):
                    grad_value = xrd_grad_mean[i].item()
                    if grad_value > 1e-8:  
                        analysis_results['xrd_features'][f'xrd_feature_{i}'] = grad_value
            
            if xafs_input is not None and self.descriptor_names:   
                for desc_name in self.descriptor_names:
                    if desc_name in xafs_input:
                        desc_tensor = xafs_input[desc_name]
                        if desc_tensor.grad is not None:
                            desc_grad = desc_tensor.grad.abs()
                            desc_importance = desc_grad.mean().item()
                            if desc_importance > 1e-8:
                                analysis_results['xafs_features'][desc_name] = desc_importance
                        else:
                            analysis_results['xafs_features'][desc_name] = 1e-8
            
            return analysis_results

class MultiTaskLoss(nn.Module):
    def __init__(self, tasks, task_names=None):
        super().__init__()
        self.tasks = tasks
        self.task_names = task_names if task_names else [f'task_{i}' for i in range(tasks)]
        self.log_vars = nn.Parameter(torch.zeros(tasks))
        
    def forward(self, preds, targets):
        num_tasks = len(preds)
        losses = []
        for i in range(num_tasks):
            loss = F.mse_loss(preds[i], targets[i])
            losses.append(loss)
        return self._forward_losses(losses)
    
    def _forward_losses(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss
    
    def get_task_weights(self):
        with torch.no_grad():
            weights = torch.exp(-self.log_vars).cpu().numpy()
            return weights.tolist()
    
    def get_task_weights_dict(self):
        with torch.no_grad():
            weights = torch.exp(-self.log_vars).cpu().numpy()
            return {name: float(weight) for name, weight in zip(self.task_names, weights)}
    
    def get_log_vars(self):
        with torch.no_grad():
            return self.log_vars.cpu().numpy().tolist()

class EarlyStopping:
    def __init__(self, 
                 patience=30,         
                 min_delta=1e-4,       
                 lr_reset_factor=0.5,   
                 restore_best_weights=True,    
                 verbose=True):        
        
        self.patience = patience
        self.min_delta = min_delta
        self.lr_reset_factor = lr_reset_factor
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = np.inf
        self.best_state_dict = None
        self.early_stop = False
        self.lr_reduced_count = 0
        
        self.val_loss_history = []
        self.best_epoch = 0
        
        if self.verbose:
            print(f"早停机制初始化: patience={patience}, min_delta={min_delta}")
    
    def __call__(self, val_loss, model, optimizer=None):
        self.val_loss_history.append(val_loss)
        
        lr_reduced = False
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]['lr']
            if hasattr(self, 'last_lr') and current_lr < self.last_lr * 0.9:
                lr_reduced = True
                self.lr_reduced_count += 1
                reset_value = int(self.patience * self.lr_reset_factor)
                self.counter = max(0, self.counter - reset_value)
                if self.verbose:
                    print(f"学习率下降 ({self.lr_reduced_count}次): {self.last_lr:.2e} -> {current_lr:.2e}, "
                          f"重置早停计数: {self.counter + reset_value} -> {self.counter}")
            self.last_lr = current_lr
        
        if val_loss < self.best_loss - self.min_delta:
            improvement = self.best_loss - val_loss
            self.best_loss = val_loss
            self.best_epoch = len(self.val_loss_history) - 1
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_state_dict = model.state_dict().copy()
            
            if self.verbose:
                print(f"验证损失改进 {improvement:.6f}，新最佳: {val_loss:.6f}，重置早停计数")
        else:
            self.counter += 1
            if self.verbose and self.counter % 10 == 0 and not lr_reduced:
                print(f"无改进，早停计数: {self.counter}/{self.patience}")
        
        if self.counter >= self.patience:
            self.early_stop = True
            if self.verbose:
                print(f"早停触发，耐心值耗尽 ({self.counter}/{self.patience})")
                print(f"最佳epoch: {self.best_epoch + 1}，最佳验证损失: {self.best_loss:.6f}")
                print(f"学习率下降次数: {self.lr_reduced_count}")
        
    def restore_best_model(self, model):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
            if self.verbose:
                print(f"已恢复最佳模型 (epoch {self.best_epoch + 1})，验证损失: {self.best_loss:.6f}")
            return True
        return False


class FusionTrainer:
    def __init__(self, 
                 model: nn.Module,
                 device: str):
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.task_names = model.task_names
        self.num_tasks = len(self.task_names)

        self.criterion = None        
        self.criterion_type = None    
        self.bandgap_cls_weight = 0.5
        self.huber_delta = 1.0
        
        self.save_dir = f"res/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"结果保存路径: {self.save_dir}")
    
    def prepare_batch_data(self, batch):
        if batch['xrd'] is not None:
            xrd_data = batch['xrd'].to(self.device)
        else:
            xrd_data = None
        
        if batch['xafs_descriptors'] is not None:
            xafs_dict = batch['xafs_descriptors']
            xafs_data = {}
            for name, tensor in xafs_dict.items():
                if tensor.dim() == 1:
                    tensor = tensor.unsqueeze(-1)
                elif tensor.dim() != 2:
                    raise ValueError(f"Unexpected dimension for {name}: {tensor.dim()}")
                xafs_data[name] = tensor.to(self.device)
        else:
            xafs_data = None
        
        properties = batch['properties'].to(self.device)
        
        task_to_idx = {'formation_energy': 0, 'fermi_energy': 1, 'band_gap': 2}
        targets = []
        for task in self.task_names:
            idx = task_to_idx.get(task, 0)
            targets.append(properties[:, idx])
        
        return xrd_data, xafs_data, targets
    
    def _init_criterion(self, criterion_type, bandgap_cls_weight=0.5, huber_delta=1.0):
        self.criterion_type = criterion_type
        self.bandgap_cls_weight = bandgap_cls_weight
        self.huber_delta = huber_delta
        
        if criterion_type == 'multitask':
            self.criterion = MultiTaskLoss(tasks=3, task_names=['formation', 'fermi', 'bandgap'])
            print("使用 MultiTaskLoss")
        else:
            self.criterion = None
            print(f"使用 {criterion_type} 损失直接计算")
    
    def _compute_loss(self, outputs, targets):
        """
        根据 self.criterion_type 计算损失
        Returns: total_loss (标量), per_task_mse_losses (列表，每个任务的原始 MSE)
        """
        preds = [outputs[task] for task in self.task_names]
        targets_list = targets 

        total_loss = self.criterion(preds, targets_list)
        per_task_mse = [F.mse_loss(p, t).item() for p, t in zip(preds, targets_list)]

        return total_loss, per_task_mse
    
    def _inverse_band_gap(self, y_transformed):
        return y_transformed
        
    def test(self, test_loader, dataset=None, visualize_mid=None):
        print("\n" + "="*60)
        print("在测试集上评估模型...")
        print("="*60)

        self.model.eval()

        all_outputs = []
        all_targets = []
        all_attention_weights = []
        all_cross_attn_xrd2xafs = []   
        all_cross_attn_xafs2xrd = []  
        
        task_attention_analysis = {task: [] for task in self.task_names}
        test_losses = []
        test_task_losses = {task: [] for task in self.task_names}
        descriptor_names = None
        analyzer = None

        detailed_importance = None  
        single_material_sample = None   
        all_fused_features = []  
        all_sample_importance = {task: [] for task in self.task_names}

        for batch_idx, batch in enumerate(tqdm(test_loader)):
            xrd_data, xafs_data, targets = self.prepare_batch_data(batch)
            
            if descriptor_names is None and batch_idx == 0:
                if xafs_data is not None:
                    descriptor_names = list(xafs_data.keys())
                else:
                    descriptor_names = []
                analyzer = MultiTaskAttentionAnalyzer(self.model, descriptor_names)
            if visualize_mid is not None and single_material_sample is None:
                if 'mid' in batch:
                    mids = batch['mid']
                    for i, mid in enumerate(mids):
                        if mid == visualize_mid:
                            xrd_single = xrd_data[i:i+1] if xrd_data is not None else None
                            xafs_single = {k: v[i:i+1] for k, v in xafs_data.items()} if xafs_data is not None else None
                            single_material_sample = (xrd_single, xafs_single)
                            break

            if batch_idx == 0 and detailed_importance is None:
                xrd_one = xrd_data[:1] if xrd_data is not None else None
                xafs_one = {k: v[:1] for k, v in xafs_data.items()} if xafs_data is not None else None
                detailed_importance = {}
                for task_name in self.task_names:
                    xrd_imp, xafs_imp = self.model.get_detailed_feature_importance(
                        xrd_one, xafs_one, task_name=task_name
                    )
                    detailed_importance[task_name] = {
                        'xrd': xrd_imp,
                        'xafs': xafs_imp
                    }

            if batch_idx == 0 and detailed_importance is not None:
                for task_name in self.task_names:
                    xrd_imp_dict = detailed_importance[task_name]['xrd']
                    xafs_imp_dict = detailed_importance[task_name]['xafs']
                    all_imp = {**xrd_imp_dict, **xafs_imp_dict}
                    if not all_imp:
                        print(f"任务 {task_name} 无有效特征重要性，跳过")
                        continue
                    
                    total = sum(all_imp.values())
                    if total > 1e-8:
                        for k in all_imp:
                            all_imp[k] /= total
                    
                    xrd_total = sum(v for k, v in all_imp.items() if k.startswith('xrd_'))
                    xafs_total = sum(v for k, v in all_imp.items() if not k.startswith('xrd_'))
                    
                    detailed_importance[task_name]['global_norm'] = all_imp
                    detailed_importance[task_name]['xrd_total_ratio'] = xrd_total
                    detailed_importance[task_name]['xafs_total_ratio'] = xafs_total

            if 'mid' in batch and batch['mid'] is not None:
                mids = batch['mid']
                if len(mids) > 0:
                    sample_imp = self.compute_shap_hierarchical_importance_samples(
                        xrd_data, xafs_data, self.task_names, mids
                    )
                    for task in self.task_names:
                        all_sample_importance[task].extend(sample_imp.get(task, []))

            if batch_idx == 0 and xrd_data is not None and xafs_data is not None:
                shap_results = self.compute_shap_hierarchical_importance(xrd_data, xafs_data, self.task_names)
                self._print_shap_results(shap_results)
                output = os.path.join(self.save_dir, "shap_hierarchical_importance.json")
                with open(output, 'w') as f:
                    json.dump(shap_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
                    print(f"Feature Attribution: {output}")

            with torch.no_grad():
                outputs = self.model(xrd_data, xafs_data)

            if 'fused_features' in outputs and outputs['fused_features'] is not None:
                fused_np = outputs['fused_features'].detach().cpu().numpy()
                all_fused_features.append(fused_np)
            
            if 'modal_attention' in outputs:
                modal_att = outputs['modal_attention']
                if modal_att is not None:   
                    modal_att = modal_att.detach().cpu().numpy()
                    all_attention_weights.append(modal_att)

            if 'cross_attention_xrd2xafs' in outputs and outputs['cross_attention_xrd2xafs'] is not None:
                attn_xrd2xafs = outputs['cross_attention_xrd2xafs'].detach().cpu()
                all_cross_attn_xrd2xafs.append(attn_xrd2xafs)
            if 'cross_attention_xafs2xrd' in outputs and outputs['cross_attention_xafs2xrd'] is not None:
                attn_xafs2xrd = outputs['cross_attention_xafs2xrd'].detach().cpu()
                all_cross_attn_xafs2xrd.append(attn_xafs2xrd)
            
            task_loss_sum = 0.0
            for i, task in enumerate(self.task_names):
                pred = outputs[task]
                target = targets[i]   
                loss = F.mse_loss(pred, target).item()
                test_task_losses[task].append(loss)
                task_loss_sum += loss
            total_loss = task_loss_sum / self.num_tasks
            test_losses.append(total_loss)

            all_outputs.append({task: outputs[task].detach().cpu().numpy() for task in self.task_names})
            all_targets.append({task: targets[i].detach().cpu().numpy() for i, task in enumerate(self.task_names)})

            if batch_idx == 0 and analyzer is not None:
                with torch.no_grad():
                    xrd_one = xrd_data[:1] if xrd_data is not None else None
                    xafs_one = {k: v[:1] for k, v in xafs_data.items()} if xafs_data is not None else None
                        
                for task_name in self.task_names:
                    xrd_task = xrd_data[:1].clone().requires_grad_(True) if xrd_data is not None else None
                    xafs_task = {k: v[:1].clone().requires_grad_(True) for k, v in xafs_data.items()} if xafs_data is not None else None
                    if xrd_task is None and xafs_task is None:
                        continue

                    task_analysis = analyzer.analyze_task_attention(xrd_task, xafs_task, task_name)
                    task_attention_analysis[task_name].append(task_analysis)

        if any(len(lst) > 0 for lst in all_sample_importance.values()):
            samples_output = os.path.join(self.save_dir, "shap_hierarchical_importance_samples.json")
            with open(samples_output, 'w') as f:
                def convert(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, torch.Tensor):
                        return obj.detach().cpu().tolist()
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert(i) for i in obj]
                    return obj
                cleaned = convert(all_sample_importance)
                json.dump(cleaned, f, indent=2)
            print(f"Feature Attribution (per-sample): {samples_output}")
        else:
            print("警告：未收集到任何样本的层级重要性数据，可能测试集中没有 'mid' 字段")

        predictions = {task: np.concatenate([out[task] for out in all_outputs]) for task in self.task_names}
        targets = {task: np.concatenate([t[task] for t in all_targets]) for task in self.task_names}
        
        test_metrics = {}
        scaling_factor = 2.0
        for task in self.task_names:
            pred_trans = predictions[task]
            true_trans = targets[task]
            pred = pred_trans
            true = true_trans

            mae_norm = mean_absolute_error(true, pred)
            rmse_norm = np.sqrt(mean_squared_error(true, pred))
            mae_normalized = np.clip(mae_norm / scaling_factor, 0, 1)
            rmse_normalized = np.clip(rmse_norm / scaling_factor, 0, 1)

            test_metrics[f'{task}_mae_normalized'] = mae_normalized
            test_metrics[f'{task}_rmse_normalized'] = rmse_normalized

        test_metrics['test_total_loss'] = np.mean(test_losses)
        for task in self.task_names:
            test_metrics[f'test_{task}_loss'] = np.mean(test_task_losses[task])

        if all_attention_weights:
            attention_weights = np.concatenate(all_attention_weights, axis=0)
            attention_weights = torch.softmax(torch.tensor(attention_weights), dim=1).numpy()
            xrd_attention = attention_weights[:, 0]
            xafs_attention = attention_weights[:, 1]
            
            test_metrics['xrd_attention_mean'] = float(xrd_attention.mean())
            test_metrics['xrd_attention_std'] = float(xrd_attention.std())
            test_metrics['xafs_attention_mean'] = float(xafs_attention.mean())
            test_metrics['xafs_attention_std'] = float(xafs_attention.std())
            
            test_metrics['xrd_attention_sum'] = float(xrd_attention.sum() / len(xrd_attention))
            test_metrics['xafs_attention_sum'] = float(xafs_attention.sum() / len(xafs_attention))
            test_metrics['attention_total_sum'] = test_metrics['xrd_attention_sum'] + test_metrics['xafs_attention_sum']

        if all_cross_attn_xrd2xafs:
            attn_xrd2xafs_tensor = torch.cat(all_cross_attn_xrd2xafs, dim=0)
            attn_xrd2xafs_mean = attn_xrd2xafs_tensor.mean().item()
            attn_xrd2xafs_std = attn_xrd2xafs_tensor.std().item()
            test_metrics['cross_attn_xrd2xafs_mean'] = attn_xrd2xafs_mean
            test_metrics['cross_attn_xrd2xafs_std'] = attn_xrd2xafs_std
        else:
            test_metrics['cross_attn_xrd2xafs_mean'] = None
            test_metrics['cross_attn_xrd2xafs_std'] = None
        
        if all_cross_attn_xafs2xrd:
            attn_xafs2xrd_tensor = torch.cat(all_cross_attn_xafs2xrd, dim=0)
            attn_xafs2xrd_mean = attn_xafs2xrd_tensor.mean().item()
            attn_xafs2xrd_std = attn_xafs2xrd_tensor.std().item()
            test_metrics['cross_attn_xafs2xrd_mean'] = attn_xafs2xrd_mean
            test_metrics['cross_attn_xafs2xrd_std'] = attn_xafs2xrd_std
        else:
            test_metrics['cross_attn_xafs2xrd_mean'] = None
            test_metrics['cross_attn_xafs2xrd_std'] = None

        if dataset and dataset.property_mean is not None:
            task_to_idx = {'formation_energy': 0, 'fermi_energy': 1, 'band_gap': 2}
            for task in self.task_names:
                idx = task_to_idx.get(task, 0)
                pred_denorm = self.denormalize_single_property(predictions[task], idx, dataset)
                true_denorm = self.denormalize_single_property(targets[task], idx, dataset)
                pred_orig = pred_denorm
                true_orig = true_denorm
                
                predictions[f'{task}_denorm'] = pred_orig
                targets[f'{task}_denorm'] = true_orig
                test_metrics[f'{task}_mae_original'] = mean_absolute_error(true_orig, pred_orig)
                test_metrics[f'{task}_rmse_original'] = np.sqrt(mean_squared_error(true_orig, pred_orig))
                test_metrics[f'{task}_r2'] = r2_score(true_orig, pred_orig)

        print("\n" + "="*15 + " 整体测试集结果 " + "="*15)
        for task in self.task_names:
            print(f"\n{task}:")
            print(f"  MAE\tRMSE\tMAE_norm\tRMSE_norm\tR²")
            mae_original = test_metrics.get(f'{task}_mae_original', 0)
            rmse_original = test_metrics.get(f'{task}_rmse_original', 0)
            mae_normalized = test_metrics.get(f'{task}_mae_normalized', 0)
            rmse_normalized = test_metrics.get(f'{task}_rmse_normalized', 0)
            r2 = test_metrics.get(f'{task}_r2', 0)
            print(f"  {mae_original:.4f}\t{rmse_original:.4f}\t{mae_normalized:.4f}\t{rmse_normalized:.4f}\t{r2:.4f}")

        for task in self.task_names:
            if task_attention_analysis[task] and descriptor_names is not None:
                print(f"\n" + "="*10 + f" {task}任务的注意力分析 " + "="*10)
                self._print_task_attention_analysis(task_attention_analysis[task][0], descriptor_names)

        if all_fused_features:
            fused_all = np.concatenate(all_fused_features, axis=0)
            fused_save_path = os.path.join(self.save_dir, "test_fused_features.npy")
            np.save(fused_save_path, fused_all)
            print(f"测试集特征已保存至: {fused_save_path}")

        results = {
            'predictions': predictions,
            'targets': targets,
            'metrics': test_metrics,
            'task_attention_analysis': task_attention_analysis,
            'attention_weights': np.concatenate(all_attention_weights, axis=0) if all_attention_weights else None,
            'cross_attention_xrd2xafs': torch.cat(all_cross_attn_xrd2xafs, dim=0).numpy() if all_cross_attn_xrd2xafs else None,
            'cross_attention_xafs2xrd': torch.cat(all_cross_attn_xafs2xrd, dim=0).numpy() if all_cross_attn_xafs2xrd else None,
            'detailed_feature_importance': detailed_importance
        }

        orig_predictions = {}
        orig_targets = {}
        for task in self.task_names:
            orig_predictions[task] = predictions[f'{task}_denorm']   
            orig_targets[task] = targets[f'{task}_denorm']

        self.save_test_results(results)
        self.save_predictions_and_plot_scatter(orig_predictions, orig_targets)
        
        if visualize_mid is not None and single_material_sample is not None:
            xrd_single, xafs_single = single_material_sample
            self.save_single_material_feature_info(
                mid=visualize_mid,
                xrd_input=xrd_single,
                xafs_input=xafs_single,
                ref_tasks=['formation_energy', 'fermi_energy', 'band_gap'],
                task_names=['formation_energy', 'fermi_energy', 'band_gap']
            )
            
        elif visualize_mid is not None:
            print(f"警告: 未能在测试集中找到材料 {visualize_mid}，跳过特征变换可视化")

        return results

    def compute_shap_hierarchical_importance(self, xrd_input, xafs_input, task_names):
        self.model.eval()
        device = next(self.model.parameters()).device

        if xrd_input is not None:
            xrd_input = xrd_input.clone().detach().to(device)
        if xafs_input is not None:
            xafs_input = {k: v.clone().detach().to(device) for k, v in xafs_input.items()}
        desc_names = sorted(xafs_input.keys()) if xafs_input else []
        desc_dims = {n: xafs_input[n].shape[1] for n in desc_names}
        parts = []
        if xrd_input is not None:
            parts.append(xrd_input)
        for n in desc_names:
            parts.append(xafs_input[n])
        flat_input = torch.cat(parts, dim=1)
        flat_input.requires_grad_(True)

        def forward_from_flat(flat):
            pos = 0
            xrd_part = None
            xafs_part = {}
            if xrd_input is not None:
                xrd_len = xrd_input.shape[1]
                xrd_part = flat[:, pos:pos+xrd_len]
                pos += xrd_len
            for n in desc_names:
                dim = desc_dims[n]
                xafs_part[n] = flat[:, pos:pos+dim]
                pos += dim
            return self.model(xrd_part, xafs_part)

        feature_blocks = []
        cur = 0
        if xrd_input is not None:
            xrd_len = xrd_input.shape[1]
            feature_blocks.append(('xrd', cur, cur+xrd_len))
            cur += xrd_len
        for n in desc_names:
            dim = desc_dims[n]
            feature_blocks.append((n, cur, cur+dim))
            cur += dim

        xafs_group_config = {
            'xafs': xafs_group,
            'xanes': xanes_group,
            'exafs': exafs_group,
            'global': simple_feature_names
        }
        desc_to_group = {}
        for gname, dlist in xafs_group_config.items():
            for d in dlist:
                desc_to_group[d] = gname

        results = {}
        for task_name in task_names:
            outputs = forward_from_flat(flat_input)
            pred = outputs[task_name].squeeze()
            fused = outputs.get('fused_features')
            if fused is None:
                raise ValueError("模型 forward 必须返回 'fused_features'")

            fused.retain_grad()
            self.model.zero_grad()
            flat_input.retain_grad()
            loss = pred.sum()
            loss.backward(retain_graph=True)

            grad_flat = flat_input.grad
            fused_grad = fused.grad
            if grad_flat is None:
                raise RuntimeError(f"flat.grad is None for {task_name}")

            importance_flat = (grad_flat * flat_input).abs().detach().cpu().numpy()
            shap_mean = importance_flat.mean(axis=0)

            group_raw = {g: 0.0 for g in ['xrd', 'xafs', 'xanes', 'exafs', 'global']}
            desc_raw = {}
            xrd_block = next((b for b in feature_blocks if b[0]=='xrd'), None)
            if xrd_block:
                s, e = xrd_block[1], xrd_block[2]
                group_raw['xrd'] = float(shap_mean[s:e].sum())
            for name, s, e in feature_blocks:
                if name == 'xrd':
                    continue
                val = float(shap_mean[s:e].sum())
                desc_raw[name] = val
                g = desc_to_group.get(name, 'xafs')
                group_raw[g] += val

            total_raw = sum(group_raw.values())
            if total_raw > 0:
                group_percent = {g: (v / total_raw) * 100.0 for g, v in group_raw.items()}
            else:
                group_percent = {g: 0.0 for g in group_raw}

            desc_percent = {}
            for name, raw_val in desc_raw.items():
                g = desc_to_group.get(name, 'xafs')
                group_total_raw = group_raw[g]
                if group_total_raw > 0:
                    desc_percent[name] = (raw_val / group_total_raw) * group_percent[g]
                else:
                    desc_percent[name] = 0.0

            if fused_grad is not None:
                fused_imp_raw = (fused_grad * fused).abs().mean(dim=0).detach().cpu().numpy()
                total_fused = fused_imp_raw.sum()
                if total_fused > 0:
                    fused_importance = (fused_imp_raw / total_fused).tolist()
                else:
                    fused_importance = fused_imp_raw.tolist()
                fused_np = fused.detach().cpu().numpy()
                if fused_np.shape[0] > 1:
                    fused_corr = np.corrcoef(fused_np.T).tolist()
                else:
                    fused_corr = None
            else:
                fused_importance, fused_corr = None, None

            attn = self._get_attention_matrices(xrd_input, xafs_input)

            results[task_name] = {
                'group_importance': group_percent,
                'descriptor_importance': desc_percent,
                'fused_shap_values': fused_importance,
                'fused_corr_matrix': fused_corr,
                'group_attention': attn.get('group_attention'),
                'group_names': attn.get('group_names', []),
                'xrd_desc_interaction': attn.get('xrd_desc_interaction'),
                'desc_interaction_names': attn.get('desc_interaction_names', []),
                'xrd_encoded_interaction': attn.get('xrd_encoded_interaction'),
            }

            flat_input.grad = None
            if fused is not None:
                fused.grad = None

        return results
    
    def save_single_material_feature_info(
        self,
        mid: str,
        xrd_input,
        xafs_input,
        ref_tasks: list = None,                      
        task_names: list = None,                     
    ):
        os.makedirs(self.save_dir, exist_ok=True)
        if isinstance(xrd_input, np.ndarray):
            xrd_input = torch.from_numpy(xrd_input).float()
        if isinstance(xafs_input, dict):
            xafs_input = {k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v
                        for k, v in xafs_input.items()}

        if xrd_input.dim() == 1:
            xrd_input = xrd_input.unsqueeze(0)          # [1, D_xrd]
        xrd_input = xrd_input.to(self.device)
        xrd_input.requires_grad_(True)

        for k in xafs_input:
            if xafs_input[k].dim() == 1:
                xafs_input[k] = xafs_input[k].unsqueeze(0)
            xafs_input[k] = xafs_input[k].to(self.device)
            xafs_input[k].requires_grad_(True)

        self.model.eval()

        intermediate = {}        
        def safe_forward_hook(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out_tensor = out[0]
                else:
                    out_tensor = out
                if isinstance(out_tensor, torch.Tensor):
                    intermediate[name] = out_tensor
            return hook

        if hasattr(self.model.xrd_branch, 'fc3'):
            self.model.xrd_branch.fc3.register_forward_hook(safe_forward_hook('xrd_features'))

        processor = self.model.xafs_processor.processor
        group_encoders = {
            'xafs_encoder': getattr(processor, 'xafs_encoder', None),
            'xanes_encoder': getattr(processor, 'xanes_encoder', None),
            'exafs_encoder': getattr(processor, 'exafs_encoder', None)
        }
        for name, encoder in group_encoders.items():
            if encoder is not None:
                encoder.register_forward_hook(safe_forward_hook(f'{name}_output'))

        if hasattr(processor, 'projection'):
            last_linear = processor.projection[-1]
            if isinstance(last_linear, nn.Linear):
                last_linear.register_forward_hook(safe_forward_hook('xafs_fused'))

        self.model.fusion_layer.register_forward_hook(safe_forward_hook('fused_features'))
        if self.model.fusion_type == 'cross_attention':
            if hasattr(self.model, 'cross_attn_xrd2xafs'):
                self.model.cross_attn_xrd2xafs.register_forward_hook(safe_forward_hook('cross_attn_xrd2xafs_out'))
            if hasattr(self.model, 'cross_attn_xafs2xrd'):
                self.model.cross_attn_xafs2xrd.register_forward_hook(safe_forward_hook('cross_attn_xafs2xrd_out'))

        outputs = self.model(xrd_input, xafs_input)
        feature_names = ['xrd_features', 'xafs_fused', 'fused_features']
        importance_by_feature = {feat_name: {} for feat_name in feature_names}

        for task in ref_tasks:
            task_output = outputs[task]  # shape [1]
            for feat_name in feature_names:
                if feat_name not in intermediate:
                    continue
                feat_tensor = intermediate[feat_name]  # [1, D]

                grad_list = torch.autograd.grad(task_output, feat_tensor,
                                                retain_graph=True,
                                                allow_unused=True)
                grad = grad_list[0]
                if grad is None:
                    imp = feat_tensor.abs().detach().cpu().numpy().squeeze(0)
                else:
                    imp = (grad * feat_tensor).abs().detach().cpu().numpy().squeeze(0)
                total = imp.sum()
                if total > 1e-8:
                    imp = imp / total
                importance_by_feature[feat_name][task] = imp.tolist()

        multi_task_fused_shap = {}
        fused_tensor = intermediate.get('fused_features')
        if fused_tensor is None:
            raise RuntimeError("无法获取 fused_features，请确保 hooks 正确捕获")

        for task_name in task_names:
            if not fused_tensor.requires_grad:
                fused_tensor.requires_grad_(True)
            pred_task = outputs[task_name].squeeze()
            fused_grad = torch.autograd.grad(pred_task, fused_tensor,
                                            retain_graph=True,
                                            allow_unused=True)[0]
            if fused_grad is None:
                shap_raw = fused_tensor.abs().detach().cpu().numpy().squeeze(0)
            else:
                shap_raw = (fused_grad * fused_tensor).abs().detach().cpu().numpy().squeeze(0)
            total = shap_raw.sum()
            if total > 1e-8:
                shap_norm = (shap_raw / total).tolist()
            else:
                shap_norm = shap_raw.tolist()
            multi_task_fused_shap[task_name] = shap_norm

        def to_list(tensor):
            return tensor.detach().cpu().numpy().squeeze(0).tolist()

        xrd_vals = to_list(intermediate['xrd_features']) if 'xrd_features' in intermediate else []
        xafs_fused_vals = to_list(intermediate['xafs_fused']) if 'xafs_fused' in intermediate else []
        fused_vals = to_list(intermediate['fused_features']) if 'fused_features' in intermediate else []

        group_outputs = {}
        for gname in ['xafs_encoder_output', 'xanes_encoder_output', 'exafs_encoder_output']:
            if gname in intermediate:
                arr = intermediate[gname].detach().cpu().numpy().squeeze(0)
                if arr.ndim > 1:
                    arr = arr.flatten()
                group_outputs[gname] = arr.tolist()

        cross_outputs = {}
        if 'cross_attn_xrd2xafs_out' in intermediate:
            arr = intermediate['cross_attn_xrd2xafs_out'].detach().cpu().numpy().squeeze(0)
            if arr.ndim > 1:
                arr = arr.flatten()
            cross_outputs['xrd2xafs'] = arr.tolist()
        if 'cross_attn_xafs2xrd_out' in intermediate:
            arr = intermediate['cross_attn_xafs2xrd_out'].detach().cpu().numpy().squeeze(0)
            if arr.ndim > 1:
                arr = arr.flatten()
            cross_outputs['xafs2xrd'] = arr.tolist()

        json_data = {
            'mid': mid,
            'ref_tasks': ref_tasks,
            'tasks_for_fused_shap': task_names,
            'xrd_features': {
                'values': xrd_vals,
                'importance': importance_by_feature.get('xrd_features', {})
            },
            'xafs_fused_features': {
                'values': xafs_fused_vals,
                'importance': importance_by_feature.get('xafs_fused', {})
            },
            'fused_features': {
                'values': fused_vals,
                'importance': importance_by_feature.get('fused_features', {})
            },
            'multi_task_fused_shap': multi_task_fused_shap,
            'xafs_group_outputs': group_outputs,
            'cross_attn_outputs': cross_outputs
        }

        json_path = os.path.join(self.save_dir, f'{mid}_feature_data.json')
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"材料 {mid} 特征数据已保存至: {json_path}")

    def compute_shap_hierarchical_importance_samples(self, xrd_input, xafs_input, task_names, mids):
        """
        Args:
            xrd_input: XRD数据，形状 [B, xrd_dim]
            xafs_input: XAFS字典，每个值形状 [B, dim]
            task_names: 任务名称列表
            mids: 当前批次的 mid 列表，长度 B
        Returns:
            Dict: 键为任务名，值为列表，每个元素包含 mid, group_importance, descriptor_importance
        """
        if mids is None or len(mids) == 0:
            return {}

        self.model.eval()
        device = next(self.model.parameters()).device

        if xrd_input is not None:
            xrd_input = xrd_input.clone().detach().to(device)
        if xafs_input is not None:
            xafs_input = {k: v.clone().detach().to(device) for k, v in xafs_input.items()}

        desc_names = sorted(xafs_input.keys()) if xafs_input else []
        desc_dims = {n: xafs_input[n].shape[1] for n in desc_names}
        parts = []
        if xrd_input is not None:
            parts.append(xrd_input)
        for n in desc_names:
            parts.append(xafs_input[n])
        flat_input = torch.cat(parts, dim=1)
        flat_input.requires_grad_(True)

        feature_blocks = []
        cur = 0
        if xrd_input is not None:
            xrd_len = xrd_input.shape[1]
            feature_blocks.append(('xrd', cur, cur + xrd_len))
            cur += xrd_len
        for n in desc_names:
            dim = desc_dims[n]
            feature_blocks.append((n, cur, cur + dim))
            cur += dim

        xafs_group_config = {
            'xafs': xafs_group,      
            'xanes': xanes_group,
            'exafs': exafs_group,
            'global': simple_feature_names
        }
        desc_to_group = {}
        for gname, dlist in xafs_group_config.items():
            for d in dlist:
                desc_to_group[d] = gname

        batch_size = flat_input.shape[0]
        results_by_task = {task: [] for task in task_names}

        def forward_from_flat(flat):
            pos = 0
            xrd_part = None
            xafs_part = {}
            if xrd_input is not None:
                xrd_len = xrd_input.shape[1]
                xrd_part = flat[:, pos:pos + xrd_len]
                pos += xrd_len
            for n in desc_names:
                dim = desc_dims[n]
                xafs_part[n] = flat[:, pos:pos + dim]
                pos += dim
            return self.model(xrd_part, xafs_part)

        for task_name in task_names:
            outputs = forward_from_flat(flat_input)
            pred = outputs[task_name].squeeze()
            fused = outputs.get('fused_features')
            if fused is None:
                raise ValueError("模型 forward 必须返回 'fused_features'")

            fused.retain_grad()
            self.model.zero_grad()
            flat_input.retain_grad()
            loss = pred.sum()
            loss.backward(retain_graph=True)

            grad_flat = flat_input.grad
            if grad_flat is None:
                raise RuntimeError(f"flat.grad is None for {task_name}")

            importance_flat = (grad_flat * flat_input).abs().detach().cpu().numpy()
            for i in range(batch_size):
                mid = mids[i] if i < len(mids) else f"sample_{i}"
                imp_i = importance_flat[i]  
                group_raw = {g: 0.0 for g in ['xrd', 'xafs', 'xanes', 'exafs', 'global']}
                desc_raw = {}
                xrd_block = next((b for b in feature_blocks if b[0] == 'xrd'), None)
                if xrd_block:
                    s, e = xrd_block[1], xrd_block[2]
                    group_raw['xrd'] = float(imp_i[s:e].sum())

                for name, s, e in feature_blocks:
                    if name == 'xrd':
                        continue
                    val = float(imp_i[s:e].sum())
                    desc_raw[name] = val
                    g = desc_to_group.get(name, 'xafs')
                    group_raw[g] += val

                total_raw = sum(group_raw.values())
                if total_raw > 0:
                    group_percent = {g: (v / total_raw) * 100.0 for g, v in group_raw.items()}
                else:
                    group_percent = {g: 0.0 for g in group_raw}

                desc_percent = {}
                for name, raw_val in desc_raw.items():
                    g = desc_to_group.get(name, 'xafs')
                    group_total_raw = group_raw[g]
                    if group_total_raw > 0:
                        desc_percent[name] = (raw_val / group_total_raw) * group_percent[g]
                    else:
                        desc_percent[name] = 0.0

                results_by_task[task_name].append({
                    'mid': mid,
                    'group_importance': group_percent,
                    'descriptor_importance': desc_percent
                })

            flat_input.grad = None
            if fused is not None:
                fused.grad = None

        return results_by_task

    def _get_attention_matrices(self, xrd_input, xafs_input):
        matrices = {}
        processor = self.model.xafs_processor.processor
        all_config_desc = xafs_group + xanes_group + exafs_group + simple_feature_names
        actual_desc = [d for d in all_config_desc if d in xafs_input]  
        try:
            if hasattr(processor, 'cross_attention') and processor.cross_attention is not None:
                with torch.no_grad():
                    group_features = []
                    group_names = []
                    if processor.xafs_encoder is not None and any(d in actual_desc for d in xafs_group):
                        group_features.append(processor.xafs_encoder(xafs_input))
                        group_names.append('xafs')
                    if processor.xanes_encoder is not None and any(d in actual_desc for d in xanes_group):
                        group_features.append(processor.xanes_encoder(xafs_input))
                        group_names.append('xanes')
                    if processor.exafs_encoder is not None and any(d in actual_desc for d in exafs_group):
                        group_features.append(processor.exafs_encoder(xafs_input))
                        group_names.append('exafs')
                    if processor.global_encoder is not None and any(d in actual_desc for d in simple_feature_names):
                        global_tensors = [xafs_input[name] for name in simple_feature_names if name in xafs_input]
                        if global_tensors:
                            global_concat = torch.cat(global_tensors, dim=1)
                            global_feat = processor.global_encoder(global_concat)
                            group_features.append(global_feat)
                            group_names.append('global')

                    if len(group_features) > 1:
                        adjusted = []
                        for idx, feat in enumerate(group_features):
                            if hasattr(processor, 'dim_adjust_layers') and idx < len(processor.dim_adjust_layers):
                                adj = processor.dim_adjust_layers[idx]
                                if feat.shape[1] != processor.attention_dim:
                                    feat = adj(feat)
                            adjusted.append(feat)
                        group_tensor = torch.stack(adjusted, dim=1)
                        _, attn_weights = processor.cross_attention(group_tensor, group_tensor, group_tensor,
                                                                    need_weights=True)
                        group_attn_matrix = attn_weights.mean(dim=0).detach().cpu().numpy()  # [G, G]
                        matrices['group_attention'] = group_attn_matrix.tolist()
                        matrices['group_names'] = group_names

                        desc_to_gidx = {}
                        for gidx, gname in enumerate(group_names):
                            if gname == 'xafs':
                                for d in xafs_group:
                                    if d in actual_desc:
                                        desc_to_gidx[d] = gidx
                            elif gname == 'xanes':
                                for d in xanes_group:
                                    if d in actual_desc:
                                        desc_to_gidx[d] = gidx
                            elif gname == 'exafs':
                                for d in exafs_group:
                                    if d in actual_desc:
                                        desc_to_gidx[d] = gidx
                            elif gname == 'global':
                                for d in simple_feature_names:
                                    if d in actual_desc:
                                        desc_to_gidx[d] = gidx
                        nd = len(actual_desc)
                        desc_attn = np.zeros((nd, nd))
                        for i, d1 in enumerate(actual_desc):
                            gi = desc_to_gidx.get(d1)
                            if gi is None:
                                continue
                            for j, d2 in enumerate(actual_desc):
                                gj = desc_to_gidx.get(d2)
                                if gj is None:
                                    continue
                                desc_attn[i, j] = group_attn_matrix[gi, gj]
                        matrices['desc_attention'] = desc_attn.tolist()
                        matrices['desc_order'] = actual_desc
        except Exception as e:
            print(f"[警告] 组间注意力提取失败: {e}")

        try:
            if xrd_input is not None and xafs_input is not None and hasattr(self.model, 'xrd_branch'):
                with torch.no_grad():
                    xrd_feat = self.model.xrd_branch(xrd_input)  # [B, D]
                    desc_outputs = []
                    valid_desc = []

                    for encoder in [processor.xafs_encoder, processor.xanes_encoder, processor.exafs_encoder]:
                        if encoder is not None and hasattr(encoder, 'encoders'):
                            for name, enc in encoder.encoders.items():
                                if name in actual_desc:
                                    out = enc(xafs_input[name])
                                    desc_outputs.append(out)
                                    valid_desc.append(name)
                    
                    for name in simple_feature_names:
                        if name in actual_desc:
                            feat = xafs_input[name]  # [B, 1]
                            desc_outputs.append(feat)
                            valid_desc.append(name)

                    if desc_outputs:
                        desc_scalars = [out.mean(dim=1, keepdim=True) for out in desc_outputs]
                        desc_mat = torch.cat(desc_scalars, dim=1)
                        xrd_np = xrd_feat.abs().mean(dim=0).cpu().numpy()   # [D]
                        desc_np = desc_mat.abs().mean(dim=0).cpu().numpy()   # [num_desc]
                        
                        raw_interaction = np.outer(xrd_np, desc_np)          # [D, num_desc]
                        total = raw_interaction.sum()
                        if total > 0:
                            interaction = raw_interaction / total
                        else:
                            interaction = raw_interaction
                        matrices['xrd_desc_interaction'] = interaction.tolist()
                        matrices['desc_interaction_names'] = valid_desc
        except Exception as e:
            print(f"[警告] XRD-描述符交互失败: {e}")

        try:
            if self.model.fusion_type == 'cross_attention' and hasattr(self.model, 'xrd_proj') and hasattr(self.model, 'xafs_proj'):
                W_q = self.model.xrd_proj.weight
                W_k = self.model.xafs_proj.weight
                interaction_dim = torch.mm(W_q, W_k.T)
                matrices['xrd_encoded_interaction'] = interaction_dim.detach().cpu().numpy().tolist()
        except Exception as e:
            print(f"[警告] 维度交互失败: {e}")

        return matrices


    def _print_shap_results(self, shap_results):
        for task_name, res in shap_results.items():
            print("=" * 60)
            print(f"\n[Gradiant-based] {task_name} 任务的层次重要性")
            print(f"=== {task_name} 任务的组级别重要性 ===")
            g = res['group_importance']
            print(f"XRD 整体: {g.get('xrd',0):.6f}%")
            print(f"signal 组: {g.get('xafs',0):.6f}%")
            print(f"xanes 组: {g.get('xanes',0):.6f}%")
            print(f"exafs 组: {g.get('exafs',0):.6f}%")
            print(f"addition 组: {g.get('global',0):.6f}%")

            print(f"\n=== {task_name} 中xafs特征级别排序 ===")
            desc_imp = res['descriptor_importance']
            sorted_desc = sorted(desc_imp.items(), key=lambda x: x[1], reverse=True)
            for name, imp in sorted_desc:
                group = self._get_desc_group(name)
                print(f"  [{group}] {name}: {imp:.6f}%")


    def _get_desc_group(self, desc_name: str) -> str:
        if desc_name in xafs_group:
            return 'XAFS'
        elif desc_name in xanes_group:
            return 'XANES'
        elif desc_name in exafs_group:
            return 'EXAFS'
        elif desc_name in simple_feature_names:
            return 'global'
        else:
            return 'unknown'
    
    def train_epoch(self, dataloader, optimizer, scheduler=None):
        self.model.train()
        total_loss = 0
        task_losses = [0] * len(self.task_names)
        
        for batch in dataloader:
            xrd_data, xafs_data, targets = self.prepare_batch_data(batch)
            optimizer.zero_grad()
            outputs = self.model(xrd_data, xafs_data)

            loss, per_task_mse = self._compute_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  
            optimizer.step()
            for i, mse in enumerate(per_task_mse):
                task_losses[i] += mse

            metrics = {'total_loss': total_loss / len(dataloader)}
            for i, task in enumerate(self.task_names):
                metrics[f'{task}_loss'] = task_losses[i] / len(dataloader)

            if scheduler:
                scheduler.step()

        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        task_losses = [0] * self.num_tasks
        all_predictions = {task: [] for task in self.task_names}
        all_targets = {task: [] for task in self.task_names}
        
        for batch in dataloader:
            xrd_data, xafs_data, targets = self.prepare_batch_data(batch)
            outputs = self.model(xrd_data, xafs_data)
            
            loss, per_task_mse = self._compute_loss(outputs, targets)
            total_loss += loss.item()
            for i, mse in enumerate(per_task_mse):
                task_losses[i] += mse
            for i, task in enumerate(self.task_names):
                all_predictions[task].extend(outputs[task].cpu().numpy())
                all_targets[task].extend(targets[i].cpu().numpy())

        metrics = {'val_total_loss': total_loss / len(dataloader)}
        for i, task in enumerate(self.task_names):
            pred_trans = np.array(all_predictions[task])
            true_trans = np.array(all_targets[task])
            pred = pred_trans
            true = true_trans

            metrics[f'val_{task}_loss'] = task_losses[i] / len(dataloader)
            metrics[f'{task}_mse'] = mean_squared_error(true, pred)
            metrics[f'{task}_mae'] = mean_absolute_error(true, pred)
            metrics[f'{task}_r2'] = r2_score(true, pred)
            metrics[f'{task}_rmse'] = np.sqrt(metrics[f'{task}_mse'])
        
        predictions = {task: (np.array(all_predictions[task]), np.array(all_targets[task])) 
                    for task in self.task_names}
        return metrics, predictions
    
    def train(self, train_loader, val_loader, criterion_type='multitask', 
              epochs=100, learning_rate=1e-3, weight_decay=5e-4, patience=40, min_delta=1e-4, 
              bandgap_cls_weight=0.5, huber_delta=1.0):  
        
        self._init_criterion(criterion_type, bandgap_cls_weight, huber_delta)
    
        params_group = [
            {'params': self.model.fusion_layer.parameters(), 'lr': 5e-4, 'weight_decay': 5e-4},
        ]
        # XRD 分支
        if hasattr(self.model, 'xrd_branch') and self.model.xrd_branch is not None:
            params_group.append(
                {'params': self.model.xrd_branch.parameters(), 'lr': 1e-4, 'weight_decay': 5e-4})
        
        # XAFS 处理器
        if hasattr(self.model, 'xafs_processor') and self.model.xafs_processor is not None:
            params_group.append(
                {'params': self.model.xafs_processor.parameters(), 'lr': 5e-4, 'weight_decay': 5e-4})

        for task_name in self.task_names:
            head = self.model.task_heads[task_name]
            lr = 2e-3 if task_name == 'band_gap' else 1e-3
            params_group.append({'params': head.parameters(), 'lr': lr, 'weight_decay': 5e-4})

        optimizer = optim.AdamW(params_group, lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                    optimizer, T_0=20, T_mult=2, eta_min=1e-6)

        early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta,
            lr_reset_factor=0.7,
            restore_best_weights=True,
            verbose=True
        )
        
        history = {
            'train_loss': [], 'val_loss': [],
            'task_weights': [], 'learning_rates': [],
            'best_epoch': 0, 'best_val_loss': float('inf')
        }
        for task in self.task_names:
            history[f'train_{task}_loss'] = []
            history[f'val_{task}_loss'] = []
        
        print("开始训练模型...")
        print("=" * 60)

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, optimizer)
            train_loss = train_metrics['total_loss']
            
            val_metrics, _ = self.validate(val_loader)
            val_loss = val_metrics['val_total_loss']
            scheduler.step(val_loss)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            for task in self.task_names:
                history[f'train_{task}_loss'].append(train_metrics[f'{task}_loss'])
                history[f'val_{task}_loss'].append(val_metrics[f'val_{task}_loss'])
            
            if self.criterion is not None and hasattr(self.criterion, 'get_task_weights'):
                history['task_weights'].append(self.criterion.get_task_weights())
            else:
                history['task_weights'].append([1.0] * len(self.task_names))
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            early_stopping(val_loss, self.model, optimizer)
            if val_loss < history['best_val_loss']:
                history['best_val_loss'] = val_loss
                history['best_epoch'] = epoch
            
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}/{epochs}")
                for task in self.task_names:
                    rmse_key = f'{task}_rmse'
                    r2_key = f'{task}_r2'
                    if rmse_key in val_metrics and r2_key in val_metrics:
                        print(f"  验证 {task} - RMSE: {val_metrics[rmse_key]:.4f}, R²: {val_metrics[r2_key]:.4f}")
            
            if early_stopping.early_stop:
                print(f"\n早停在 epoch {epoch+1}")
                print(f"最佳验证损失: {early_stopping.best_loss:.6f}")
                break
        
        early_stopping.restore_best_model(self.model)
        self.save_final_model(history)
        
        print("训练完成!")
        return history
    
    def _compute_global_attention_from_model(self, task_attention_analysis, global_modal_attention, descriptor_names):
        tasks = ['formation_energy', 'fermi_energy', 'band_gap']
        
        if len(global_modal_attention) < 2:
            return self._compute_global_attention_average(task_attention_analysis, descriptor_names)
        
        xrd_global_ratio = global_modal_attention[0]
        xafs_global_ratio = global_modal_attention[1]
        
        global_xrd_features = {}
        global_xafs_features = {}
        
        for task in tasks:
            if task in task_attention_analysis and task_attention_analysis[task]:
                analysis_data = task_attention_analysis[task][0]
                
                xrd_features = analysis_data.get('xrd_features', {})
                for feat_name, weight in xrd_features.items():
                    if feat_name not in global_xrd_features or weight > global_xrd_features[feat_name]:
                        global_xrd_features[feat_name] = weight
                
                xafs_features = analysis_data.get('xafs_features', {})
                for desc_name, weight in xafs_features.items():
                    if desc_name not in global_xafs_features or weight > global_xafs_features[desc_name]:
                        global_xafs_features[desc_name] = weight
        
        return {
            'xrd_features': global_xrd_features,
            'xafs_features': global_xafs_features,
            'xrd_ratio': xrd_global_ratio,
            'xafs_ratio': xafs_global_ratio,
            'source': 'model',
            'task_count': len(tasks)
        }

    def _compute_global_attention_average(self, task_attention_analysis, descriptor_names):
        tasks = ['formation_energy', 'fermi_energy', 'band_gap']
        
        global_xrd_features = {}
        global_xafs_features = {}
        
        xrd_feature_counts = {}
        xafs_feature_counts = {}
        
        for task in tasks:
            if task in task_attention_analysis and task_attention_analysis[task]:
                analysis_data = task_attention_analysis[task][0]
                
                xrd_features = analysis_data.get('xrd_features', {})
                for feat_name, weight in xrd_features.items():
                    if feat_name not in global_xrd_features:
                        global_xrd_features[feat_name] = 0
                        xrd_feature_counts[feat_name] = 0
                    global_xrd_features[feat_name] += weight
                    xrd_feature_counts[feat_name] += 1
                
                xafs_features = analysis_data.get('xafs_features', {})
                for desc_name, weight in xafs_features.items():
                    if desc_name not in global_xafs_features:
                        global_xafs_features[desc_name] = 0
                        xafs_feature_counts[desc_name] = 0
                    global_xafs_features[desc_name] += weight
                    xafs_feature_counts[desc_name] += 1
        
        for feat_name in global_xrd_features:
            if xrd_feature_counts[feat_name] > 0:
                global_xrd_features[feat_name] /= xrd_feature_counts[feat_name]
        
        for desc_name in global_xafs_features:
            if xafs_feature_counts[desc_name] > 0:
                global_xafs_features[desc_name] /= xafs_feature_counts[desc_name]
        
        xrd_total = sum(global_xrd_features.values())
        xafs_total = sum(global_xafs_features.values())
        global_total = xrd_total + xafs_total
        
        if global_total > 0:
            global_xrd_ratio = xrd_total / global_total
            global_xafs_ratio = xafs_total / global_total
        else:
            global_xrd_ratio = 0.5
            global_xafs_ratio = 0.5
        
        print(f"  基于{len(tasks)}个任务平均计算: XRD={global_xrd_ratio:.4f}, XAFS={global_xafs_ratio:.4f}")
        
        return {
            'xrd_features': global_xrd_features,
            'xafs_features': global_xafs_features,
            'xrd_ratio': global_xrd_ratio,
            'xafs_ratio': global_xafs_ratio,
            'source': 'average',
            'task_count': len(tasks)
        }

    def _print_task_attention_analysis(self, analysis_data, descriptor_names):
        if not analysis_data:
            print("没有可用的注意力分析数据")
            return
        
        print("特征重要性权重分析:")
        xrd_features = analysis_data.get('xrd_features', {})
        xafs_features = analysis_data.get('xafs_features', {})
        
        xrd_raw_sum = sum(xrd_features.values())
        xafs_raw_sum = sum(xafs_features.values())
        total_raw_sum = xrd_raw_sum + xafs_raw_sum
        
        if total_raw_sum == 0:
            print("  错误：所有特征权重和为0")
            return
        
        xrd_ratio = xrd_raw_sum / total_raw_sum
        xafs_ratio = xafs_raw_sum / total_raw_sum
        
        all_features_dict = {}
        if xrd_raw_sum > 0:
            for feat_name, weight in xrd_features.items():
                normalized_weight = (weight / xrd_raw_sum) * xrd_ratio
                all_features_dict[feat_name] = normalized_weight
        
        if xafs_raw_sum > 0:
            for desc_name, weight in xafs_features.items():
                normalized_weight = (weight / xafs_raw_sum) * xafs_ratio
                all_features_dict[desc_name] = normalized_weight
        
        total_weight = sum(all_features_dict.values())
        if abs(total_weight - 1.0) > 0.001 and total_weight > 0:
            scale = 1.0 / total_weight
            for key in all_features_dict:
                all_features_dict[key] *= scale
            total_weight = sum(all_features_dict.values())

        all_features = []  
        xrd_total_weight = 0.0
        for feat_name, weight in all_features_dict.items():
            if isinstance(feat_name, str) and 'xrd' in feat_name.lower():
                xrd_total_weight += weight
            else:
                matched = False
                for desc_name in descriptor_names:
                    if desc_name == feat_name:
                        all_features.append((f'[{group_feat_map.get(desc_name)}] {desc_name}', weight))
                        matched = True
                        break
                if not matched:
                    all_features.append((f'[{group_feat_map.get(desc_name)}] {feat_name}', weight))
        
        if xrd_total_weight > 0:
            all_features.append(('[XRD] XRD总体', xrd_total_weight))
        
        print("\n特征重要性排序:")
        print("-" * 80)
        
        if not all_features:
            print("  没有可用的特征重要性数据")
            return
        
        all_features.sort(key=lambda x: x[1], reverse=True)
        top_features = all_features[:10]
        
        for i, (feat_name, weight) in enumerate(top_features, 1):
            percentage = (weight / total_weight) * 100
            print(f"  {i:2d}. {feat_name:<30}\t{weight:.6f} ({percentage:.2f}%)")
    

    def save_checkpoint(self, epoch, history, train_metrics, val_metrics):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'history': {}
        }
        
        for key, value in history.items():
            if isinstance(value, list):
                if len(value) > epoch + 1:
                    checkpoint['history'][key] = value[:epoch+1]
                else:
                    checkpoint['history'][key] = value
            elif isinstance(value, (int, float, str, bool)) or value is None:
                checkpoint['history'][key] = value
            else:
                try:
                    checkpoint['history'][key] = str(value)
                except:
                    checkpoint['history'][key] = None
        
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint, checkpoint_path)
        
        json_path = os.path.join(self.save_dir, f"metrics_epoch_{epoch+1}.json")
        
        json_metrics = {
            'epoch': epoch,
            'train_loss': float(train_metrics['total_loss']),
            'val_loss': float(val_metrics['val_total_loss']),
            'formation_r2': float(val_metrics.get('formation_r2', 0)),
            'fermi_r2': float(val_metrics.get('fermi_r2', 0)),
            'bandgap_r2': float(val_metrics.get('bandgap_r2', 0))
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        if (epoch + 1) % 10 == 0:
            print(f"  检查点已保存到: {checkpoint_path}")
            print(f"  指标已保存到: {json_path}")
        
        return checkpoint_path
    
    def save_final_model(self, history):
        model_path = os.path.join(self.save_dir, "final_model.pth")
    
        model_config = {
            'xrd_seq_len': getattr(self.model, 'xrd_seq_len', 1024),
            'xafs_total_dim': getattr(self.model, 'xafs_total_dim', 500),
            'hidden_dim': getattr(self.model, 'hidden_dim', 256),
            'descriptor_dims': getattr(self.model.xafs_processor, 'descriptor_dims', {}),
            'output_activation': getattr(self.model, 'output_activation', 'none'),
            'fusion_type': getattr(self.model, 'fusion_type', 'concat'),
            'cross_attn_heads': getattr(self.model.cross_attn, 'num_heads', 4) if hasattr(self.model, 'cross_attn') else 4,
            'cross_attn_dropout': getattr(self.model, 'cross_attn_dropout', 0.1),
            'use_modal_attention': getattr(self.model, 'use_modal_attention', False),
        }
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': model_config,
            'history': self._prepare_history_for_saving(history)
        }, model_path)
        
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w') as f:
            serializable_history = self._prepare_history_for_saving(history, for_json=True)
            json.dump(serializable_history, f, indent=2, ensure_ascii=False)
        
        print(f"模型已保存到: {model_path}")
        print(f"训练历史已保存到: {history_path}")
        
        summary_path = os.path.join(self.save_dir, "training_summary.txt")
        self._save_training_summary(history, summary_path)
        
        return model_path

    def _prepare_history_for_saving(self, history, for_json=False):
        prepared_history = {}
        
        for key, value in history.items():
            if isinstance(value, list) and len(value) > 0:
                if for_json:
                    prepared_list = []
                    for item in value:
                        if isinstance(item, (np.float32, np.float64)):
                            prepared_list.append(float(item))
                        elif isinstance(item, np.ndarray):
                            prepared_list.append(item.tolist())
                        elif isinstance(item, list):
                           
                            nested_prepared = []
                            for nested_item in item:
                                if isinstance(nested_item, (np.float32, np.float64)):
                                    nested_prepared.append(float(nested_item))
                                elif isinstance(nested_item, np.ndarray):
                                    nested_prepared.append(nested_item.tolist())
                                else:
                                    nested_prepared.append(nested_item)
                            prepared_list.append(nested_prepared)
                        else:
                            prepared_list.append(item)
                    prepared_history[key] = prepared_list
                else:
                    prepared_history[key] = value
            elif isinstance(value, (np.float32, np.float64)):
                prepared_history[key] = float(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                prepared_history[key] = value
            else:
                try:
                    prepared_history[key] = float(value)
                except:
                    prepared_history[key] = str(value)
        
        return prepared_history

    def _save_training_summary(self, history, summary_path):
        with open(summary_path, 'w') as f:
            f.write("多模态融合模型训练摘要\n")
            f.write("=" * 60 + "\n\n")
            
            if 'best_epoch' in history and 'best_val_loss' in history:
                f.write(f"最佳epoch: {history['best_epoch'] + 1}\n")
                f.write(f"最佳验证损失: {history['best_val_loss']:.6f}\n")
            
            if 'val_loss' in history and len(history['val_loss']) > 0:
                final_val_loss = history['val_loss'][-1]
                f.write(f"最终验证损失: {final_val_loss:.6f}\n\n")
            
            if 'val_formation' in history and len(history['val_formation']) > 0:
                f.write("最终任务损失:\n")
                f.write(f"  形成能损失: {history['val_formation'][-1]:.6f}\n")
                f.write(f"  费米能损失: {history['val_fermi'][-1]:.6f}\n")
                f.write(f"  能带隙损失: {history['val_bandgap'][-1]:.6f}\n")
            
            if 'learning_rates' in history and len(history['learning_rates']) > 0:
                f.write(f"\n最终学习率: {history['learning_rates'][-1]:.2e}\n")
        
        print(f"训练摘要已保存到: {summary_path}")
    
    def save_test_results(self, results):
        results_path = os.path.join(self.save_dir, "test_results.json")
        
        serializable_results = {
            'metrics': {},
            'attention_stats': {}
        }
        
        for key, value in results['metrics'].items():
            if isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                serializable_results['metrics'][key] = float(value) if np.issubdtype(type(value), np.floating) else int(value)
            elif isinstance(value, (float, int, str, bool)) or value is None:
                serializable_results['metrics'][key] = value
            else:
                try:
                    serializable_results['metrics'][key] = float(value)
                except:
                    serializable_results['metrics'][key] = str(value)
        
        serializable_results['attention_stats'] = {
            'xrd_mean': float(results['metrics']['xrd_attention_mean']),
            'xrd_std': float(results['metrics']['xrd_attention_std']),
            'xafs_mean': float(results['metrics']['xafs_attention_mean']),
            'xafs_std': float(results['metrics']['xafs_attention_std'])
        }
        
        predictions_path = os.path.join(self.save_dir, "predictions.npz")
        save_dict = {
            'formation_pred': np.array(results['predictions'].get('formation_energy', [])),
            'fermi_pred': np.array(results['predictions'].get('fermi_energy', [])),
            'bandgap_pred': np.array(results['predictions'].get('band_gap', [])),
            'formation_true': np.array(results['targets'].get('formation_energy', [])),
            'fermi_true': np.array(results['targets'].get('fermi_energy', [])),
            'bandgap_true': np.array(results['targets'].get('band_gap', [])),
            'attention_weights': results.get('attention_weights', np.array([]))
        }
        if results.get('cross_attention_xrd2xafs') is not None:
            save_dict['cross_attn_xrd2xafs'] = results['cross_attention_xrd2xafs']
        if results.get('cross_attention_xafs2xrd') is not None:
            save_dict['cross_attn_xafs2xrd'] = results['cross_attention_xafs2xrd']
        np.savez_compressed(predictions_path, **save_dict)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"测试结果已保存到: {results_path}")
        print(f"预测数据已保存到: {predictions_path}")
        
        summary_path = os.path.join(self.save_dir, "test_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("测试结果摘要\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("主要指标:\n")
            f.write("-" * 30 + "\n")
            for key in ['formation_energy', 'fermi_energy', 'band_gap']:
                f.write(f"{key}:\n")
                f.write(f"  R²: {serializable_results['metrics'].get(f'{key}_r2', 0):.4f}\n")
                f.write(f"  RMSE: {serializable_results['metrics'].get(f'{key}_rmse', 0):.4f}\n")
                f.write(f"  MAE: {serializable_results['metrics'].get(f'{key}_mae', 0):.4f}\n")
                
                if f'{key}_denorm_r2' in serializable_results['metrics']:
                    f.write(f"  反标准化 R²: {serializable_results['metrics'][f'{key}_denorm_r2']:.4f}\n")
                    f.write(f"  反标准化 RMSE: {serializable_results['metrics'][f'{key}_denorm_rmse']:.4f}\n")
                f.write("\n")
            
            f.write("注意力统计:\n")
            f.write("-" * 30 + "\n")
            f.write(f"XRD注意力均值: {serializable_results['attention_stats']['xrd_mean']:.4f}\n")
            f.write(f"XRD注意力标准差: {serializable_results['attention_stats']['xrd_std']:.4f}\n")
            f.write(f"XAFS注意力均值: {serializable_results['attention_stats']['xafs_mean']:.4f}\n")
            f.write(f"XAFS注意力标准差: {serializable_results['attention_stats']['xafs_std']:.4f}\n")
        
        print(f"测试摘要已保存到: {summary_path}")

    def save_predictions_and_plot_scatter(self, predictions, targets, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        
        scatter_dir = os.path.join(save_dir, 'scatter_plots')
        os.makedirs(scatter_dir, exist_ok=True)
        
        for task in self.task_names:
            pred = predictions[task]
            true = targets[task]
            
            txt_path = os.path.join(scatter_dir, f'{task}_predictions.txt')
            np.savetxt(txt_path, np.column_stack((true, pred)), 
                    header='true_value predicted_value', comments='')
            
            plt.figure(figsize=(6, 6))
            plt.scatter(true, pred, alpha=0.6, edgecolors='k', s=30)
            min_val = min(true.min(), pred.min())
            max_val = max(true.max(), pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            plt.xlabel('True Value')
            plt.ylabel('Predicted Value')
            plt.title(f'{task} - Scatter Plot')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(scatter_dir, f'{task}_scatter.png'), dpi=300)
            plt.close()
        
        print(f"预测值和散点图已保存至: {scatter_dir}")

    def visualize_single_material_feature_transformation(self, mid, xrd_input, xafs_input, task_name='fermi_energy'):

        if isinstance(xrd_input, np.ndarray):
            xrd_input = torch.from_numpy(xrd_input).float()
        if isinstance(xafs_input, dict):
            xafs_input = {k: torch.from_numpy(v).float() if isinstance(v, np.ndarray) else v for k, v in xafs_input.items()}
        
        if xrd_input.dim() == 1:
            xrd_input = xrd_input.unsqueeze(0)
        xrd_input = xrd_input.to(self.device)
        xrd_input.requires_grad_(True)
        
        for k in xafs_input:
            if xafs_input[k].dim() == 1:
                xafs_input[k] = xafs_input[k].unsqueeze(0)
            xafs_input[k] = xafs_input[k].to(self.device)
            xafs_input[k].requires_grad_(True)
        
        self.model.eval()
        
        intermediate = {}
        def safe_forward_hook(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out_tensor = out[0]
                else:
                    out_tensor = out
                if isinstance(out_tensor, torch.Tensor):
                    intermediate[name] = out_tensor.clone()
                else:
                    print(f"警告: {name} 的输出不是张量，类型为 {type(out_tensor)}，已跳过")
            return hook
        
        if hasattr(self.model.xrd_branch, 'fc3'):
            self.model.xrd_branch.fc3.register_forward_hook(safe_forward_hook('xrd_features'))
        
        processor = self.model.xafs_processor.processor
        group_encoders = {
            'xafs_encoder': getattr(processor, 'xafs_encoder', None),
            'xanes_encoder': getattr(processor, 'xanes_encoder', None),
            'exafs_encoder': getattr(processor, 'exafs_encoder', None)
        }
        for name, encoder in group_encoders.items():
            if encoder is not None:
                encoder.register_forward_hook(safe_forward_hook(f'{name}_output'))
        
        if hasattr(processor, 'projection'):
            last_linear = processor.projection[-1]
            if isinstance(last_linear, nn.Linear):
                last_linear.register_forward_hook(safe_forward_hook('xafs_fused'))
        
        self.model.fusion_layer.register_forward_hook(safe_forward_hook('fused_features'))
        
        if self.model.fusion_type == 'cross_attention':
            if hasattr(self.model, 'cross_attn_xrd2xafs'):
                self.model.cross_attn_xrd2xafs.register_forward_hook(safe_forward_hook('cross_attn_xrd2xafs_out'))
            if hasattr(self.model, 'cross_attn_xafs2xrd'):
                self.model.cross_attn_xafs2xrd.register_forward_hook(safe_forward_hook('cross_attn_xafs2xrd_out'))
        
        outputs = self.model(xrd_input, xafs_input)
        pred = outputs[task_name].squeeze()
        
        importance = {}
        for name, feat in intermediate.items():
            if not isinstance(feat, torch.Tensor):
                continue
            if feat.requires_grad:
                grad = autograd.grad(pred, feat, retain_graph=True, allow_unused=True)[0]
                if grad is None:
                    imp = feat.abs().squeeze(0).detach().cpu().numpy()
                else:
                    imp = grad.abs().squeeze(0).detach().cpu().numpy()
            else:
                imp = feat.abs().squeeze(0).detach().cpu().numpy()
            imp_sum = imp.sum()
            if imp_sum > 1e-8:
                imp = imp / imp_sum
            importance[name] = imp
        
        xrd_original = xrd_input.squeeze(0).detach().cpu().numpy()
        xafs_original_list = []
        for name, tensor in xafs_input.items():
            xafs_original_list.append(tensor.squeeze(0).detach().cpu().numpy())
        xafs_original = np.concatenate(xafs_original_list) if xafs_original_list else np.array([])
        
        fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
        
        if 'xrd_features' in intermediate and 'xrd_features' in importance:
            xrd_feat = intermediate['xrd_features'].squeeze(0).detach().cpu().numpy()
            xrd_imp = importance['xrd_features']
            ax1 = axes1[0]
            ax1.plot(xrd_feat, color='blue', label='XRD feature value')
            ax1.set_xlabel('Feature Index')
            ax1.set_ylabel('Feature Value', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.set_title('(a) XRD Features Importance')
            ax1.grid(alpha=0.3)
            ax1_imp = ax1.twinx()
            ax1_imp.plot(xrd_imp, color='red', linestyle='--', label='Importance')
            ax1_imp.set_ylabel('Normalized Importance', color='red')
            ax1_imp.tick_params(axis='y', labelcolor='red')
            ax1_imp.set_ylim(0, 0.06)
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax1_imp.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            axes1[0].text(0.5, 0.5, 'XRD features not captured', ha='center')
        
        if 'xafs_fused' in intermediate and 'xafs_fused' in importance:
            xafs_feat = intermediate['xafs_fused'].squeeze(0).detach().cpu().numpy()
            xafs_imp = importance['xafs_fused']
            ax2 = axes1[1]
            ax2.plot(xafs_feat, color='blue', label='XAFS feature value')
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Feature Value', color='blue')
            ax2.tick_params(axis='y', labelcolor='blue')
            ax2.set_title('(b) XAFS Features Importance')
            ax2.grid(alpha=0.3)
            ax2_imp = ax2.twinx()
            ax2_imp.plot(xafs_imp, color='red', linestyle='--', label='Importance')
            ax2_imp.set_ylabel('Normalized Importance', color='red')
            ax2_imp.tick_params(axis='y', labelcolor='red')
            ax2_imp.set_ylim(0, 0.06)
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_imp.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            axes1[1].text(0.5, 0.5, 'XAFS features not captured', ha='center')
        
        if 'fused_features' in intermediate and 'fused_features' in importance:
            fused_feat = intermediate['fused_features'].squeeze(0).detach().cpu().numpy()
            fused_imp = importance['fused_features']
            ax3 = axes1[2]
            ax3.plot(fused_feat, color='blue', label='Fused feature value')
            ax3.set_xlabel('Feature Index')
            ax3.set_ylabel('Feature Value', color='blue')
            ax3.tick_params(axis='y', labelcolor='blue')
            ax3.set_title('(c) Fused Features Importance')
            ax3.grid(alpha=0.3)
            ax3_imp = ax3.twinx()
            ax3_imp.plot(fused_imp, color='red', linestyle='--', label='Importance')
            ax3_imp.set_ylabel('Normalized Importance', color='red')
            ax3_imp.tick_params(axis='y', labelcolor='red')
            ax3_imp.set_ylim(0, 0.06)
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3_imp.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            axes1[2].text(0.5, 0.5, 'Fused features not captured', ha='center')
        
        plt.tight_layout()
        save_path1 = os.path.join(self.save_dir, f'{mid}_feature_importance.png')
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        plt.close()
        
        fig2 = plt.figure(figsize=(18, 15))
        gs = fig2.add_gridspec(3, 2, height_ratios=[1, 1.5, 1])
        
        ax_a1 = fig2.add_subplot(gs[0, 0])
        ax_a1.plot(xrd_original, color='black')
        ax_a1.set_title('(a) Raw XRD Spectrum')
        ax_a1.set_xlabel('Point Index')
        ax_a1.set_ylabel('Intensity')
        ax_a1.grid(alpha=0.3)
        
        ax_a2 = fig2.add_subplot(gs[0, 1])
        ax_a2.plot(xafs_original, color='orange', alpha=0.6)
        ax_a2.set_title('(a) Raw XAFS Descriptors (concatenated)')
        ax_a2.set_xlabel('Descriptor Index (concatenated)')
        ax_a2.set_ylabel('Value')
        ax_a2.grid(alpha=0.3)
        
        ax_b1 = fig2.add_subplot(gs[1, 0])
        if 'xrd_features' in intermediate:
            xrd_seq = intermediate['xrd_features'].squeeze(0).detach().cpu().numpy()
            ax_b1.plot(xrd_seq, label='XRDCNN output', color='blue')
            ax_b1.set_title('(b) Feature Extraction - XRDCNN Output')
            ax_b1.set_xlabel('Feature Index')
            ax_b1.legend()
            ax_b1.grid(alpha=0.3)
        else:
            ax_b1.text(0.5, 0.5, 'XRD features not captured', ha='center')
        
        ax_b2 = fig2.add_subplot(gs[1, 1])
        curves = []
        curve_names = []
        for enc_name in ['xafs_encoder_output', 'xanes_encoder_output', 'exafs_encoder_output']:
            if enc_name in intermediate:
                feat = intermediate[enc_name].squeeze(0).detach().cpu().numpy()
                curves.append(feat)
                curve_names.append(enc_name.replace('_output', ''))
        if 'xafs_fused' in intermediate:
            feat = intermediate['xafs_fused'].squeeze(0).detach().cpu().numpy()
            curves.append(feat)
            curve_names.append('group_fusion')
        if curves:
            max_len = max(len(c) for c in curves)
            padded_curves = [np.pad(c, (0, max_len - len(c)), constant_values=0) for c in curves]
            for c, name in zip(padded_curves, curve_names):
                ax_b2.plot(c, label=name)
            ax_b2.set_title('(b) Feature Extraction - XAFS Group Features')
            ax_b2.set_xlabel('Feature Index (padded to same length)')
            ax_b2.legend(loc='upper right')
            ax_b2.grid(alpha=0.3)
        else:
            ax_b2.text(0.5, 0.5, 'XAFS features not captured', ha='center')
        
        ax_c = fig2.add_subplot(gs[2, :])
        curves_c = []
        names_c = []
        if 'cross_attn_xrd2xafs_out' in intermediate:
            feat = intermediate['cross_attn_xrd2xafs_out'].squeeze(0).detach().cpu().numpy()
            if feat.ndim == 2:
                feat = feat.flatten()
            elif feat.ndim == 3:
                feat = feat.mean(axis=0).flatten()
            curves_c.append(feat)
            names_c.append('cross_attn_xrd2xafs')
        if 'cross_attn_xafs2xrd_out' in intermediate:
            feat = intermediate['cross_attn_xafs2xrd_out'].squeeze(0).detach().cpu().numpy()
            if feat.ndim == 2:
                feat = feat.flatten()
            elif feat.ndim == 3:
                feat = feat.mean(axis=0).flatten()
            curves_c.append(feat)
            names_c.append('cross_attn_xafs2xrd')
        if 'fused_features' in intermediate:
            feat = intermediate['fused_features'].squeeze(0).detach().cpu().numpy()
            curves_c.append(feat)
            names_c.append('fusion_layer')
        if curves_c:
            max_len_c = max(len(c) for c in curves_c)
            padded_c = [np.pad(c, (0, max_len_c - len(c)), constant_values=0) for c in curves_c]
            for c, name in zip(padded_c, names_c):
                ax_c.plot(c, label=name)
            ax_c.set_title('(c) Cross-attention & Fusion Outputs')
            ax_c.set_xlabel('Feature Index (padded)')
            ax_c.legend()
            ax_c.grid(alpha=0.3)
        else:
            ax_c.text(0.5, 0.5, 'No cross-attention or fusion outputs captured', ha='center')
        
        plt.tight_layout()
        save_path2 = os.path.join(self.save_dir, f'{mid}_feature_transformation.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"材料 {mid} 特征重要性图已保存至: {save_path1}")
        print(f"材料 {mid} 特征变换过程图已保存至: {save_path2}")


    def _get_sample_by_mid(self, mid, dataloader, dataset):
        if dataset is not None:
            if hasattr(dataset, 'get_sample_by_mid'):
                return dataset.get_sample_by_mid(mid)
            elif hasattr(dataset, 'samples'):
                for sample in dataset.samples:
                    if sample[0] == mid:
                        return sample[1], sample[2], sample[3]
        if dataloader is not None:
            for batch in dataloader:
                if 'mid' in batch:
                    indices = (batch['mid'] == mid).nonzero(as_tuple=True)[0]
                    if len(indices) > 0:
                        idx = indices[0].item()
                        xrd = batch['xrd'][idx].cpu().numpy()
                        xafs = {k: v[idx].cpu().numpy() for k, v in batch['xafs_descriptors'].items()}
                        prop = batch['properties'][idx].cpu().numpy()
                        return xrd, xafs, prop
        return None


    def plot_task_weights(self, history):
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(history['task_weights']) + 1)
        task_weights = np.array(history['task_weights'])
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, task_weights[:, 0], label='形成能权重', linewidth=2)
        plt.plot(epochs, task_weights[:, 1], label='费米能权重', linewidth=2)
        plt.plot(epochs, task_weights[:, 2], label='能带隙权重', linewidth=2)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('任务权重', fontsize=12)
        plt.title('多任务权重变化', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        weight_plot_path = os.path.join(self.save_dir, "task_weights_evolution.png")
        plt.savefig(weight_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"任务权重变化图已保存到: {weight_plot_path}")

    def denormalize_single_property(self, normalized_values, property_idx, dataset):
        if dataset.property_mean is None or dataset.property_std is None:
            return normalized_values
        
        if isinstance(dataset.property_mean, torch.Tensor):
            mean = dataset.property_mean[property_idx].item()
            std = dataset.property_std[property_idx].item()
        else:
            mean = dataset.property_mean[property_idx]
            std = dataset.property_std[property_idx]
        
        return normalized_values * std + mean


from typing import Dict, List, Tuple, Any
def get_top_features(feature_dict: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
    if not feature_dict:
        return []
    
    sorted_features = sorted(feature_dict.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    return sorted_features[:min(top_k, len(sorted_features))]

def print_task_specific_analysis(per_task_results):
    tasks = list(per_task_results.keys())
    
    print("\n模态注意力对比:")
    print("-"*60)
    print(f"{'任务':<20} {'XRD注意力':<12} {'XAFS注意力':<12} {'XRD/XAFS比例'}")
    print("-"*60)
    
    for task in tasks:
        modal_att = per_task_results[task]['modal_attention']
        xrd_att = modal_att[0]
        xafs_att = modal_att[1]
        ratio = xrd_att / xafs_att if xafs_att > 0 else float('inf')
        print(f"{task:<20} {xrd_att:.4f}        {xafs_att:.4f}        {ratio:.2f}")
    
    for task in tasks:
        print(f"\n{task.upper()} 重要特征:")
        print("-"*60)
        
        xrd_features = per_task_results[task]['xrd_features']
        top_xrd = get_top_features(xrd_features, top_k=3)
        
        if top_xrd:
            print("XRD特征:")
            for feature_name, importance in top_xrd:
                if feature_name.startswith('xrd_feature_'):
                    idx = int(feature_name.split('_')[-1])
                    print(f"  • 特征{idx}: {importance:.4f}")
        
        xafs_features = per_task_results[task]['xafs_features']
        top_xafs = get_top_features(xafs_features, top_k=3)
        
        if top_xafs:
            print("\nXAFS特征:")
            for desc_name, importance in top_xafs:
                print(f"  • {desc_name}: {importance:.4f}")
                