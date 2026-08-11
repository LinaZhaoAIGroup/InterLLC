
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import ml_models
from processor import Dataset, DataProcessor, MultiModalDataset
from trainer import PropertyPredictorTrainer, FusionTrainer
from utils import save_model_checkpoint, get_model, save_training_results
from configurations import simple_feature_names
from fusion import MultiModalFusionModel


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_args():
    parser = argparse.ArgumentParser()

    ### Input and Data Configurations ###
    parser.add_argument('--data', type=str, default="abx3", help='a2bbx6, abx3, all')
    parser.add_argument('--cite', type=str, default="B", help='A, B, X, all')
    parser.add_argument('--spectrum', type=str, default="xrd+xafs", help='xrd, xafs, xrd+xafs')

    ### Training Configurations ###
    parser.add_argument('--fun', type=str, default="train", help="train, case")
    parser.add_argument('--mode', type=str, default="two", help="single, two")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--task', nargs='+', default=["formation_energy", "fermi_energy", "band_gap"])
    parser.add_argument('--xrd_model', type=str, default="cnn9",
                        help="cnn1, cnn2, cnn3, cnn4, cnn5, cnn6, cnn7, cnn8, cnn9, cnn10, cnn11, \
                              mlp, rnn, lstm, gru, birnn, bilstm, bigru, transf, itransf, patchtst")
    parser.add_argument('--xas', type=str, default="xafs")
    parser.add_argument('--xas_feat', type=bool, default=True)
    parser.add_argument('--fxas', nargs='+', default=["cwt", "cdf", "wacsf", "soap2", "pdos", "msr1"])
    parser.add_argument('--xas_model', type=str, default="aemlp",
                        help="cnn, mlp, rnn, lstm, aecnn, aemlp, transf")
    
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--xrd_dropout_fc', type=float, default=0.2)
    parser.add_argument('--xafs_dropout_rate', type=float, default=0.1)
    parser.add_argument('--xafs_fusion_dropout', type=float, default=0.2)
    parser.add_argument('--modal_attention_dropout', type=float, default=0.2)
    parser.add_argument('--fusion_layer_dropout', type=float, default=0.4)
    parser.add_argument('--task_head_dropout', type=float, default=0.2)
    parser.add_argument('--xafs_fusion_type', type=str, default='hierarchical', help="hierarchical, concat")
    parser.add_argument('--fusion_type', type=str, default="cross_attention", help="concat, cross_attention")
    parser.add_argument('--cross_attn_heads', type=int, default=8)
    parser.add_argument('--cross_attn_dropout', type=float, default=0.1)

    ### for AB fusion test ###
    parser.add_argument('--use_xrd', type=bool, default=True)
    parser.add_argument('--use_xafs', type=bool, default=True)
    parser.add_argument('--ratio', type=float, default=0)

    ### Multiple Task Learning ###
    parser.add_argument('--loss', type=str, default="multitask")
    parser.add_argument('--bandgap_cls_weight', type=float, default=0.5)
    parser.add_argument('--huber_delta', type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    return args

def comparer(args):
    # 数据预处理和筛选
    processor = DataProcessor(data_dir="data")
    valid_materials = processor.filter_materials_with_all_data(args.data, args.xas)
    
    if len(valid_materials) == 0:
        print("错误：没有找到符合条件的材料！")
        return

    sequences, properties, material_ids, formulas = processor.prepare_spectral_sequences(args.spectrum, args.cite, args.xas)

    # 划分数据集
    X_train, X_temp, y_train, y_temp, ids_train, ids_temp, formulas_train, formulas_temp = train_test_split(
        sequences, properties, material_ids, formulas,
        test_size=0.3, random_state=42, stratify=None
    )
    X_val, X_test, y_val, y_test, ids_val, ids_test, formulas_val, formulas_test = train_test_split(
        X_temp, y_temp, ids_temp, formulas_temp,
        test_size=0.5, random_state=42, stratify=None
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {X_train.shape[0]} 个样本")
    print(f"  验证集: {X_val.shape[0]} 个样本")
    print(f"  测试集: {X_test.shape[0]} 个样本")
    
    # 创建数据集和数据加载器
    train_dataset = Dataset(X_train, y_train)
    val_dataset = Dataset(X_val, y_val)
    test_dataset = Dataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    with open("ml_test_dataset_info.txt", "w") as f:
        f.write("material_id\tformula\tformation_energy\tfermi_energy\tband_gap\n")
        for mid, formula, props in zip(ids_test, formulas_test, y_test):
            f.write(f"{mid}\t{formula}\t{props[0]:.6f}\t{props[1]:.6f}\t{props[2]:.6f}\n")
    print("测试集信息已保存至 ml_test_dataset_info.txt")

    # 创建模型
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model, model_pth = get_model(args, fxas='cwt')
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(model)

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    # 训练模型
    trainer = PropertyPredictorTrainer(model, device)
    _, _ = trainer.train(train_loader, val_loader, model_pth,
                                             epochs=args.epochs, patience=args.patience)
    
    test_predictions, test_targets, results = trainer.evaluate(test_loader, test_dataset)
    trainer.plot_predictions_vs_targets(test_predictions, test_targets, test_dataset)
    save_model_checkpoint(
                model=model,
                optimizer=trainer.optimizer,
                train_losses=trainer.train_losses,
                val_losses=trainer.val_losses,
                epoch=len(trainer.train_losses),
                filename=model_pth
            )
    
    # # 保存配置
    # save_training_config(config, 'res/training_config.json')
    
    print(f"\n 训练完成！")
    
    return model, processor, results

def prepare_datasets_for_fusion(xrd_sequences, xafs_dict, property_values, descriptor_names,
                                material_ids, test_size=0.2, val_size=0.1, test_material_ids=None, 
                                true_props_dict=None, random_seed=42):
    """
    准备多谱训练数据集
    Args:
        xafs_dict: 字典格式 {descriptor_name: [n_samples, descriptor_dim]}
    """
    if xrd_sequences is None and xafs_dict is None:
        raise ValueError("至少需要一个模态数据 (XRD 或 XAFS)")
    
    if test_material_ids is not None and len(test_material_ids) > 0:
        # 将 test_material_ids 转换为在 material_ids 中的索引
        np.random.seed(random_seed) 
        n_samples = len(property_values)
        indices = np.arange(n_samples)
        
        test_idx = []
        missing_ids = []
        for mid in test_material_ids:
            try:
                idx = material_ids.index(mid)
                test_idx.append(idx)
            except ValueError:
                missing_ids.append(mid)
        
        if missing_ids:
            print(f"警告：以下 test_material_ids 未在 material_ids 中找到，已忽略: {missing_ids}")
        
        test_idx = np.array(test_idx)
        temp_idx = np.array([i for i in indices if i not in test_idx]) #  
        val_ratio = val_size / (1 - test_size if test_size else 1)  
        n_val = int(val_size * n_samples)
        n_val = min(n_val, len(temp_idx) - 1) 
        if n_val > 0:
            val_idx = np.random.choice(temp_idx, size=n_val, replace=False)
            train_idx = np.array([i for i in temp_idx if i not in val_idx])
        else:
            val_idx = np.array([])
            train_idx = temp_idx
        print(f"基于指定的 test_material_ids数据集划分")

    else:
        # 随机划分逻辑
        indices = np.arange(len(property_values))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=42)
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_idx)} 样本")
    print(f"验证集: {len(val_idx)} 样本")
    print(f"测试集: {len(test_idx)} 样本")

    # 切片 material_ids
    train_mids = [material_ids[i] for i in train_idx]
    val_mids = [material_ids[i] for i in val_idx]
    test_mids  = [material_ids[i] for i in test_idx]

    if test_material_ids is not None and true_props_dict is not None:
        test_props_list = [true_props_dict[mid] for mid in test_mids]
        test_property_values = np.array(test_props_list, dtype=np.float32)
    else:
        test_property_values = property_values[test_idx]
    
    # 根据索引分割XAFS字典
    def split_xafs_dict(xafs_dict, indices):
        if xafs_dict is None:
            return None
        result = {}
        for name, features in xafs_dict.items():
            result[name] = features[indices]
        return result
    
    # 创建MultiModalDataset
    train_dataset = MultiModalDataset(
        xrd_sequences=xrd_sequences[train_idx] if xrd_sequences is not None else None,
        xafs_dict=split_xafs_dict(xafs_dict, train_idx) if xafs_dict is not None else None,
        properties=property_values[train_idx],
        descriptor_names=descriptor_names,
        material_ids=train_mids,
    )
    
    val_dataset = MultiModalDataset(
        xrd_sequences=xrd_sequences[val_idx] if xrd_sequences is not None else None,
        xafs_dict=split_xafs_dict(xafs_dict, val_idx) if xafs_dict is not None else None,
        properties=property_values[val_idx],
        descriptor_names=descriptor_names,
        material_ids=val_mids,
    )
    
    test_dataset = MultiModalDataset(
        xrd_sequences=xrd_sequences[test_idx] if xrd_sequences is not None else None,
        xafs_dict=split_xafs_dict(xafs_dict, test_idx) if xafs_dict is not None else None,
        properties=test_property_values,
        descriptor_names=descriptor_names,
        material_ids=test_mids,
    )

    train_dataset.normalize()
    if train_dataset.property_mean is not None:
        val_dataset.properties = (val_dataset.properties - train_dataset.property_mean) / train_dataset.property_std
        test_dataset.properties = (test_dataset.properties - train_dataset.property_mean) / train_dataset.property_std
        
        val_dataset.property_mean = train_dataset.property_mean
        val_dataset.property_std = train_dataset.property_std
        test_dataset.property_mean = train_dataset.property_mean
        test_dataset.property_std = train_dataset.property_std
    
    return train_dataset, val_dataset, test_dataset, test_idx

def read_ids_and_props(txt_path="ml_test_dataset_info.txt"):
    if txt_path == None:
        return None, None
    
    test_material_ids = []
    test_props = {}  # mid -> [formation_energy, fermi_energy, band_gap]
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if i == 0:  
                    continue
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 5:
                        mid = parts[0]
                        formation = float(parts[2])
                        fermi = float(parts[3])
                        band_gap = float(parts[4])
                        test_material_ids.append(mid)
                        test_props[mid] = [formation, fermi, band_gap]
        print(f"从 {txt_path} 读取到 {len(test_material_ids)} 个测试材料ID及其属性")
    else:
        print(f"警告: 未找到测试集文件 {txt_path}，将使用默认随机划分")
        test_material_ids = None
        test_props = None
    return test_material_ids, test_props


def dict_collate_fn(batch):
    """
    自定义 collate 函数，支持缺失模态
    """
    has_xrd = 'xrd' in batch[0] and batch[0]['xrd'] is not None
    has_xafs = 'xafs_descriptors' in batch[0] and batch[0]['xafs_descriptors'] is not None

    xrd_batch = []
    xafs_batch_dict = {}
    properties_batch = []
    mid_batch = []

    if has_xafs:
        sample_xafs = batch[0]['xafs_descriptors']
        desc_names = list(sample_xafs.keys())
        for name in desc_names:
            xafs_batch_dict[name] = []

    for sample in batch:
        if has_xrd: xrd_batch.append(sample['xrd'])
        if has_xafs:
            for name in desc_names:
                xafs_batch_dict[name].append(sample['xafs_descriptors'][name])

        properties_batch.append(sample['properties'])

        if 'mid' in sample:
            mid_batch.append(sample['mid'])

    output = {}

    if has_xrd:
        xrd_tensor = torch.stack(xrd_batch, dim=0)
        output['xrd'] = xrd_tensor
    else:
        output['xrd'] = None   

    if has_xafs:
        xafs_dict_out = {}
        for name in desc_names:
            xafs_dict_out[name] = torch.stack(xafs_batch_dict[name], dim=0)
        output['xafs_descriptors'] = xafs_dict_out
    else:
        output['xafs_descriptors'] = None

    properties_tensor = torch.stack(properties_batch, dim=0)
    output['properties'] = properties_tensor

    if mid_batch:
        output['mid'] = mid_batch

    return output

def main(device):
    processor = DataProcessor(data_dir="data/")
    valid_materials = processor.filter_materials_with_all_data(args.data, args.xas)
    
    if len(valid_materials) == 0:
        print("错误：没有找到符合条件的材料！")
        return
    
    xrd_sequences, xafs_dict, property_values = processor.prepare_multi_sequences(args.cite, args.fxas, args.xas_feat)
    if not args.use_xrd:
        xrd_sequences = None
        print("只利用 XAFS 数据进行多属性预测")
    if not args.use_xafs:
        xafs_dict = None
        print("只利用 XRD 数据进行多属性预测")
    if xrd_sequences is None and xafs_dict is None: 
        print("错误：xrd与xafs参数错误")
        return
    
    descriptor_dims = {}
    if xafs_dict is not None:
        for name, features in xafs_dict.items():
            if name in simple_feature_names:
                descriptor_dims[name] = 1
            else:
                descriptor_dims[name] = features.shape[1]
    else:
        print("未使用 XAFS 数据，维度为空")
    
    if xafs_dict is not None:
        descriptor_names = list(descriptor_dims.keys()) if args.xas_feat else args.fxas
    else:
        descriptor_names = []

    test_material_ids, test_true_props = read_ids_and_props(txt_path=None)
    train_dataset, val_dataset, test_dataset, test_idx \
            = prepare_datasets_for_fusion(xrd_sequences, xafs_dict, property_values, descriptor_names, 
                                          material_ids=valid_materials, test_material_ids=test_material_ids,
                                          true_props_dict=test_true_props,    
                                          random_seed=42)
    save_dataset_split(train_dataset, val_dataset, test_dataset, processor)
    
    if args.use_xrd and train_dataset.xrd_sequences is not None:
        xrd_seq_len = train_dataset.xrd_sequences.shape[1]
    else:
        xrd_seq_len = 1024  

    if args.use_xafs and train_dataset.xafs_dict is not None:
        descriptor_dims = {}
        for name in train_dataset.descriptor_names:
            feat = train_dataset.xafs_dict[name]
            descriptor_dims[name] = 1 if feat.dim() == 1 else feat.shape[1]
    else:
        descriptor_dims = {}

    if args.ratio > 0:
        subset_size = int(len(train_dataset) * args.ratio)
        from torch.utils.data import Subset
        indices = list(range(subset_size))
        train_dataset = Subset(train_dataset, indices)
        print(f"\n训练子集抽取完成: {len(train_dataset)} 样本 (占原始训练集的 {args.ratio*100:.0f}%)")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                             num_workers=2, collate_fn=dict_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=2, collate_fn=dict_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=2, collate_fn=dict_collate_fn)

    model = MultiModalFusionModel(
                xrd_seq_len=xrd_seq_len,
                xafs_total_dim=sum(descriptor_dims.values()) if args.use_xafs else 0,  
                descriptor_split_dims=descriptor_dims,  
                hidden_dim=args.hidden_dim,
                xrd_dropout_fc=args.xrd_dropout_fc,
                xafs_dropout_rate=args.xafs_dropout_rate,
                xafs_fusion_dropout=args.xafs_fusion_dropout,
                modal_attention_dropout=args.modal_attention_dropout,
                fusion_layer_dropout=args.fusion_layer_dropout,
                task_head_dropout=args.task_head_dropout,
                fusion_type=args.fusion_type,      
                cross_attn_heads=args.cross_attn_heads,                 
                cross_attn_dropout=args.cross_attn_dropout, 
                xafs_fusion_type=args.xafs_fusion_type,
                device=device,
                task_names=args.task,
                use_xrd=args.use_xrd,      
                use_xafs=args.use_xafs,
        )

    print(f"\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    
    trainer = FusionTrainer(model=model, device=device if torch.cuda.is_available() else 'cpu')
    history = trainer.train(train_loader=train_loader, val_loader=val_loader, 
                            epochs=args.epochs, criterion_type=args.loss, 
                            bandgap_cls_weight=args.bandgap_cls_weight,
                            huber_delta=args.huber_delta,
                        )
    save_trained_model(trainer, train_dataset, history)
    test_results = trainer.test(test_loader, dataset=test_dataset) 
    save_training_results(test_results, args.spectrum, args.cite, args.xas, args.fxas)
    
    return trainer, processor, test_results

def save_trained_model(trainer, train_dataset, history):
    model_path = "res/final_model.pth"
    os.makedirs("res", exist_ok=True)

    actual_xrd_seq_len = train_dataset.xrd_sequences.shape[1] if train_dataset.xrd_sequences is not None else 1024
    actual_descriptor_dims = {}
    if train_dataset.xafs_dict is not None:
        for name in train_dataset.descriptor_names:
            feat = train_dataset.xafs_dict[name]
            actual_descriptor_dims[name] = 1 if feat.dim() == 1 else feat.shape[1]
    
    torch.save({
        'model_state_dict': trainer.model.state_dict(),
        'model_config': {
            'xrd_seq_len': actual_xrd_seq_len,
            'xafs_total_dim': sum(actual_descriptor_dims.values()) if args.use_xafs else 0,
            'descriptor_split_dims': actual_descriptor_dims,
            'hidden_dim': args.hidden_dim,
            'xrd_dropout_fc': args.xrd_dropout_fc,
            'xafs_dropout_rate': args.xafs_dropout_rate,
            'xafs_fusion_dropout': args.xafs_fusion_dropout,
            'modal_attention_dropout': args.modal_attention_dropout,
            'fusion_layer_dropout': args.fusion_layer_dropout,
            'task_head_dropout': args.task_head_dropout,
            'fusion_type': args.fusion_type,
            'cross_attn_heads': args.cross_attn_heads,
            'cross_attn_dropout': args.cross_attn_dropout,
            'xafs_fusion_type': args.xafs_fusion_type,
            'task_names': args.task,
            'use_xrd': args.use_xrd,
            'use_xafs': args.use_xafs,
        },
        'property_mean': train_dataset.property_mean,
        'property_std': train_dataset.property_std,
        'history': history
    }, model_path)
    print(f"最优模型已保存至: {model_path}")

def print_testdataset_info(processor, valid_materials, test_idx, save_dir='res/'):
    import os
    
    all_mids = []
    all_formulas = []

    for mid in valid_materials:
        if mid in processor.properties:
            formula = processor.properties[mid].get('formula', '')
        else:
            formula = ''
        all_mids.append(mid)
        all_formulas.append(formula)
    
    test_mids = [all_mids[i] for i in test_idx]
    test_formulas = [all_formulas[i] for i in test_idx]
    os.makedirs(save_dir, exist_ok=True)
    mid_formula_path = os.path.join(save_dir, 'test_mid_formula.txt')
    with open(mid_formula_path, 'w') as f:
        f.write("mid\tformula\n")
        for mid, formula in zip(test_mids, test_formulas):
            f.write(f"{mid}\t{formula}\n")
    print(f"测试集的 mid 和 formula 已保存至: {mid_formula_path}")


def save_dataset_split(train_dataset, val_dataset, test_dataset, processor, output_dir='res/'):
    os.makedirs(output_dir, exist_ok=True)
    def get_original_rows(material_ids):
        rows = []
        for mid in material_ids:
            props = processor.properties[mid]          
            formula = props.get('formula', '')
            formation = props['formation_energy']
            fermi = props['fermi_energy']
            band_gap = props['band_gap']
            rows.append((mid, formula, formation, fermi, band_gap))
        return rows

    train_rows = get_original_rows(train_dataset.material_ids)
    with open(os.path.join(output_dir, "train.txt"), 'w') as f:
        f.write("material_id\tformula\tformation_energy\tfermi_energy\tband_gap\n")
        for mid, formula, formation, fermi, band_gap in train_rows:
            f.write(f"{mid}\t{formula}\t{formation:.6f}\t{fermi:.6f}\t{band_gap:.6f}\n")

    val_rows = get_original_rows(val_dataset.material_ids)
    with open(os.path.join(output_dir, "validate.txt"), 'w') as f:
        f.write("material_id\tformula\tformation_energy\tfermi_energy\tband_gap\n")
        for mid, formula, formation, fermi, band_gap in val_rows:
            f.write(f"{mid}\t{formula}\t{formation:.6f}\t{fermi:.6f}\t{band_gap:.6f}\n")

    test_rows = get_original_rows(test_dataset.material_ids)
    with open(os.path.join(output_dir, "test.txt"), 'w') as f:
        f.write("material_id\tformula\tformation_energy\tfermi_energy\tband_gap\n")
        for mid, formula, formation, fermi, band_gap in test_rows:
            f.write(f"{mid}\t{formula}\t{formation:.6f}\t{fermi:.6f}\t{band_gap:.6f}\n")
    print(f"数据集分割信息已保存至 {output_dir}")

    
def run_case_fun(args, device, case_ids, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    print(f"\n===== Case 模式：测试 {case_ids} =====")
    processor = DataProcessor(data_dir="data")
    valid_materials = processor.filter_materials_with_all_data(args.data, args.xas)
    if len(valid_materials) == 0:
        print("错误：没有找到符合条件的材料！")
        return None, None, None

    xrd_sequences, xafs_dict, property_values = processor.prepare_multi_sequences(
        args.cite, args.fxas, args.xas_feat)

    if not args.use_xrd:
        xrd_sequences = None
    if not args.use_xafs:
        xafs_dict = None
    if xrd_sequences is None and xafs_dict is None:
        print("错误：xrd与xafs参数错误")
        return None, None, None

    descriptor_dims = {}
    if xafs_dict is not None:
        for name, features in xafs_dict.items():
            if name in simple_feature_names:
                descriptor_dims[name] = 1
            else:
                descriptor_dims[name] = features.shape[1]
    descriptor_names = list(descriptor_dims.keys()) if args.xas_feat else args.fxas

    np.random.seed(seed)
    n_samples = len(property_values)
    indices = np.arange(n_samples)
    id_to_idx = {mid: idx for idx, mid in enumerate(valid_materials)}
    train_val_idx, base_test_idx = train_test_split(indices, test_size=test_ratio, random_state=seed)
    val_size_from_train_val = val_ratio / (train_ratio + val_ratio)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size_from_train_val, random_state=seed)

    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()
    base_test_idx = base_test_idx.tolist()

    case_idx = []
    missing_cases = []
    for mid in case_ids:
        if mid in id_to_idx:
            idx = id_to_idx[mid]
            case_idx.append(idx)
        else:
            missing_cases.append(mid)
    if missing_cases:
        print(f"警告：以下 case 材料不在有效材料中，已忽略: {missing_cases}")

    for idx in case_idx:
        if idx not in base_test_idx:
            base_test_idx.append(idx)
    test_idx = list(set(base_test_idx))
    print(f"\n数据集划分（强制测试集包含 {len(case_idx)} 个 case）:")
    print(f"训练集: {len(train_idx)} 样本")
    print(f"验证集: {len(val_idx)} 样本")
    print(f"测试集: {len(test_idx)} 样本")

    train_mids = [valid_materials[i] for i in train_idx]
    val_mids = [valid_materials[i] for i in val_idx]
    test_mids = [valid_materials[i] for i in test_idx]

    test_property_values = property_values[test_idx]
    def split_xafs_dict(xafs_dict, indices):
        if xafs_dict is None:
            return None
        return {name: features[indices] for name, features in xafs_dict.items()}

    train_dataset = MultiModalDataset(
        xrd_sequences=xrd_sequences[train_idx] if xrd_sequences is not None else None,
        xafs_dict=split_xafs_dict(xafs_dict, train_idx) if xafs_dict is not None else None,
        properties=property_values[train_idx],
        descriptor_names=descriptor_names,
        material_ids=train_mids,
    )
    val_dataset = MultiModalDataset(
        xrd_sequences=xrd_sequences[val_idx] if xrd_sequences is not None else None,
        xafs_dict=split_xafs_dict(xafs_dict, val_idx) if xafs_dict is not None else None,
        properties=property_values[val_idx],
        descriptor_names=descriptor_names,
        material_ids=val_mids,
    )
    test_dataset = MultiModalDataset(
        xrd_sequences=xrd_sequences[test_idx] if xrd_sequences is not None else None,
        xafs_dict=split_xafs_dict(xafs_dict, test_idx) if xafs_dict is not None else None,
        properties=test_property_values,
        descriptor_names=descriptor_names,
        material_ids=test_mids,
    )

    train_dataset.normalize()
    if train_dataset.property_mean is not None:
        val_dataset.properties = (val_dataset.properties - train_dataset.property_mean) / train_dataset.property_std
        test_dataset.properties = (test_dataset.properties - train_dataset.property_mean) / train_dataset.property_std
        val_dataset.property_mean = train_dataset.property_mean
        val_dataset.property_std = train_dataset.property_std
        test_dataset.property_mean = train_dataset.property_mean
        test_dataset.property_std = train_dataset.property_std

    save_dataset_split(train_dataset, val_dataset, test_dataset, processor)
    if args.use_xrd and train_dataset.xrd_sequences is not None:
        xrd_seq_len = train_dataset.xrd_sequences.shape[1]
    else:
        xrd_seq_len = 1024

    if args.use_xafs and train_dataset.xafs_dict is not None:
        descriptor_dims = {}
        for name in train_dataset.descriptor_names:
            feat = train_dataset.xafs_dict[name]
            descriptor_dims[name] = 1 if feat.dim() == 1 else feat.shape[1]
    else:
        descriptor_dims = {}

    if args.ratio > 0:
        subset_size = int(len(train_dataset) * args.ratio)
        from torch.utils.data import Subset
        indices = list(range(subset_size))
        train_dataset = Subset(train_dataset, indices)
        print(f"训练子集抽取完成: {len(train_dataset)} 样本")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, collate_fn=dict_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, collate_fn=dict_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, collate_fn=dict_collate_fn)
    model = MultiModalFusionModel(
        xrd_seq_len=xrd_seq_len,
        xafs_total_dim=sum(descriptor_dims.values()) if args.use_xafs else 0,
        descriptor_split_dims=descriptor_dims,
        hidden_dim=args.hidden_dim,
        xrd_dropout_fc=args.xrd_dropout_fc,
        xafs_dropout_rate=args.xafs_dropout_rate,
        xafs_fusion_dropout=args.xafs_fusion_dropout,
        modal_attention_dropout=args.modal_attention_dropout,
        fusion_layer_dropout=args.fusion_layer_dropout,
        task_head_dropout=args.task_head_dropout,
        fusion_type=args.fusion_type,
        cross_attn_heads=args.cross_attn_heads,
        cross_attn_dropout=args.cross_attn_dropout,
        xafs_fusion_type=args.xafs_fusion_type,
        device=device,
        task_names=args.task,
        use_xrd=args.use_xrd,
        use_xafs=args.use_xafs,
    )

    print(f"\n模型结构:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")

    trainer = FusionTrainer(model=model,
                            device=device if torch.cuda.is_available() else 'cpu')

    history = trainer.train(train_loader=train_loader, val_loader=val_loader,
                            epochs=args.epochs, criterion_type=args.loss,
                            bandgap_cls_weight=args.bandgap_cls_weight,
                            huber_delta=args.huber_delta)

    save_trained_model(trainer, train_dataset, history)
    test_results = trainer.test(test_loader, dataset=test_dataset, visualize_mid='mp-995191')
    save_training_results(test_results, args.spectrum, args.cite, args.xas, args.fxas)

    print(f"\nCase 模式完成！结果已保存")
    return trainer, processor, test_results


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    if args.mode == 'single':
        model, processor, results = comparer(args)
    elif args.mode == 'two':   
        if args.fun == 'train':
            model, processor, results = main(device)
        elif args.fun == 'case': 
            case_ids = ["mp-23037", "mp-675022", "mp-675524", "mp-998333", "mp-998428", "mp-567629", "mp-567681",
                        "mp-570223", "mp-998323", "mp-998322", "mp-5811", "mp-27214", "mp-867844", "mp-554601", "mp-21043",
                        "mp-5020", "mp-5986", "mp-19990", "mp-504715", "mp-995191", "mp-5827", "mp-5229"]
            run_case_fun(args, device, case_ids)
        else:
            print(f"fun parameter error: {args.fun}")
    else:
        ml_models.ml_comparer(args)
    