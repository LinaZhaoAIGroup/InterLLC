import os
import time
import json
import torch
import numpy as np
from datetime import datetime

from xrd_models import *
from xas_models import *
from configurations import fxas_len_map


def get_model(args, fxas):
    if args.spectrum == "xrd":
        if args.xrd_model == "cnn1": model = XRDCNN1()
        elif args.xrd_model == "cnn2": model = XRDCNN2()
        elif args.xrd_model == "cnn3": model = XRDCNN3()
        elif args.xrd_model == "cnn4": model = XRDCNN4()
        elif args.xrd_model == "cnn5": model = XRDCNN5()
        elif args.xrd_model == "cnn6": model = XRDCNN6()
        elif args.xrd_model == "cnn7": model = XRDCNN7()
        elif args.xrd_model == "cnn8": model = XRDCNN8()
        elif args.xrd_model == "cnn9": model = XRDCNN9()
        elif args.xrd_model == "cnn10": model = XRDCNN10()
        elif args.xrd_model == "cnn11": model = XRDCNN11()
        elif args.xrd_model == "mlp": model = XRDMLP()
        elif args.xrd_model == "rnn": model = XRDRNN()
        elif args.xrd_model == "lstm": model = XRDLSTM()
        elif args.xrd_model == "gru": model = XRDGRU()
        elif args.xrd_model == "birnn": model = XRDBiRNN()
        elif args.xrd_model == "bilstm": model = XRDBiLSTM()
        elif args.xrd_model == "bigru": model = XRDBiGRU()
        elif args.xrd_model == "transf": model = XRDTransformer()
        elif args.xrd_model == "itransf": model = XRDiTransformer()
        elif args.xrd_model == "patchtst": model = XRDPatchTST()

        model_pth = f'res/{args.xrd_model}.pth'

    elif args.spectrum == "xafs":
        fxas_len = fxas_len_map.get(fxas, 100)

        if args.xas_model == "mlp": model = MLP()
        elif args.xas_model == "cnn": model = XASCNN(input_size=fxas_len)
        elif args.xas_model == "lstm":
            model = XASLSTM(input_size=fxas_len)
        elif args.xas_model == "aecnn":
            model = XASAECNN(input_size=fxas_len)
        elif args.xas_model == "aemlp":
            model = XASAEMLP(input_size=fxas_len)
        elif args.xas_model == "mhcnn":
            model = MultiHeadCNN(input_size=fxas_len, num_heads=3, output_operation="mean")
        elif args.xas_model == "mhmlp":
            model = MultiHeadMLP(input_size=fxas_len, num_heads=3, output_operation="mean")
        elif args.xas_model == "transf":
            model = XASMHTransformer(input_size=fxas_len, num_heads=12)
        

        model_pth = f'res/{args.xas_model}_{fxas}.pth'
        
    
    elif "+" in args.spectrum:
        fuse = args.spectrum.split("+")
        if len(fuse) == 2:
            model = None
        else:
            print("three-dimensional fusion is considered")
            model = None
        model_pth = f'res/{args.spectrum}_{args.fxas}.pth'

    return model, model_pth

def save_model_checkpoint(model, optimizer, train_losses, val_losses, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model_architecture': str(model.__class__.__name__),
        'save_time': time.time()
    }
    
    torch.save(checkpoint, filename)
    print(f"模型检查点已保存到: {filename}")
    
    return checkpoint

def save_training_config(config, filename):
    config_dict = {
        'lr': config.lr,
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'input_size': config.input_size,
        'debug': config.debug
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    print(f"训练配置已保存到: {filename}")
    
    return config_dict


def save_dataset_split(train_dataset, val_dataset, test_dataset, processor):
    os.makedirs("results", exist_ok=True)
    
    header = "material_id\tformula\tformation_energy\tfermi_energy\tband_gap\n"
    original_props = {}
    for mid, props in processor.properties.items():
        original_props[mid] = [
            props.get('formation_energy', 0.0),
            props.get('fermi_energy', 0.0),
            props.get('band_gap', 0.0)
        ]
    
    def extract_properties(mid):
        return original_props.get(mid, (0,0,0))
    
    # 训练集
    train_entries = []
    for mid in train_dataset.material_ids:
        formation, fermi, bandgap = extract_properties(mid)
        train_entries.append(f"{mid}\t{mid}\t{formation}\t{fermi}\t{bandgap}\n")
    
    # 验证集
    val_entries = []
    for mid in val_dataset.material_ids:
        formation, fermi, bandgap = extract_properties(mid)
        val_entries.append(f"{mid}\t{mid}\t{formation}\t{fermi}\t{bandgap}\n")
    
    # 测试集
    test_entries = []
    for mid in test_dataset.material_ids:
        formation, fermi, bandgap = extract_properties(mid)
        test_entries.append(f"{mid}\t{mid}\t{formation}\t{fermi}\t{bandgap}\n")
    
    with open("results/train.txt", "w") as f:
        f.write(header)
        f.writelines(train_entries)
    with open("results/validate.txt", "w") as f:
        f.write(header)
        f.writelines(val_entries)
    with open("results/test.txt", "w") as f:
        f.write(header)
        f.writelines(test_entries)
    
    print(f"数据集已保存至 results/ 目录（所有文件均存储原始属性值）")

def save_training_results(test_results, spectrum, cite, xas, fxas):
    save_dir = f"results/multimodal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    config = {
        'spectrum': spectrum,
        'cite': cite,
        'xas': xas,
        'fxas': fxas,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    metrics = test_results['metrics']
    serializable_metrics = {}
    
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64, np.float16)):
            serializable_metrics[key] = float(value)
        elif isinstance(value, (np.int32, np.int64, np.int16, np.int8, np.uint8)):
            serializable_metrics[key] = int(value)
        elif isinstance(value, (float, int, str, bool)) or value is None:
            serializable_metrics[key] = value
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        else:
            try:
                serializable_metrics[key] = float(value)
            except (ValueError, TypeError):
                serializable_metrics[key] = str(value)
    
    metrics_path = os.path.join(save_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
    
    predictions_path = os.path.join(save_dir, 'predictions.npz')
    
    formation_pred = np.array(test_results['predictions'].get('formation_energy', []))
    fermi_pred = np.array(test_results['predictions'].get('fermi_energy', []))
    bandgap_pred = np.array(test_results['predictions'].get('band_gap', []))
    formation_true = np.array(test_results['targets'].get('formation_energy', []))
    fermi_true = np.array(test_results['targets'].get('fermi_energy', []))
    bandgap_true = np.array(test_results['targets'].get('band_gap', []))
    
    np.savez_compressed(
        predictions_path,
        formation_pred=formation_pred,
        fermi_pred=fermi_pred,
        bandgap_pred=bandgap_pred,
        formation_true=formation_true,
        fermi_true=fermi_true,
        bandgap_true=bandgap_true
    )
    
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("多模态融合模型测试结果摘要\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("配置信息:\n")
        f.write(f"  光谱类型: {spectrum}\n")
        f.write(f"  位点类型: {cite}\n")
        f.write(f"  XAS类型: {xas}\n")
        f.write(f"  特征提取方法: {fxas}\n\n")
        
        f.write("主要性能指标:\n")
        f.write("-" * 40 + "\n")
        
        property_names = ['formation_energy', 'fermi_energy', 'band_gap']
        metric_types = ['r2', 'rmse', 'mae']
        
        f.write(f"{'属性':<15} {'R²':<10} {'RMSE':<10} {'MAE':<10}\n")
        f.write("-" * 45 + "\n")
        
        for prop in property_names:
            r2_key = f'{prop}_r2'
            rmse_key = f'{prop}_rmse'
            mae_key = f'{prop}_mae'
            
            r2 = serializable_metrics.get(r2_key, 0)
            rmse = serializable_metrics.get(rmse_key, 0)
            mae = serializable_metrics.get(mae_key, 0)
            
            f.write(f"{prop:<15} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f}\n")
            
            denorm_r2_key = f'{prop}_denorm_r2'
            if denorm_r2_key in serializable_metrics:
                denorm_r2 = serializable_metrics[denorm_r2_key]
                denorm_rmse = serializable_metrics.get(f'{prop}_denorm_rmse', 0)
                f.write(f"  反标准化     {denorm_r2:<10.4f} {denorm_rmse:<10.4f}\n")
        
        f.write("\n注意力统计:\n")
        f.write("-" * 40 + "\n")
        if 'xrd_attention_mean' in serializable_metrics:
            f.write(f"XRD注意力均值: {serializable_metrics['xrd_attention_mean']:.4f}\n")
            f.write(f"XRD注意力标准差: {serializable_metrics['xrd_attention_std']:.4f}\n")
            f.write(f"XAFS注意力均值: {serializable_metrics['xafs_attention_mean']:.4f}\n")
            f.write(f"XAFS注意力标准差: {serializable_metrics['xafs_attention_std']:.4f}\n")
    
    print(f"\n训练结果已保存到: {save_dir}")
    print(f"配置文件: {config_path}")
    print(f"测试指标: {metrics_path}")
    print(f"预测数据: {predictions_path}")
    print(f"结果摘要: {summary_path}")
    
    return save_dir

