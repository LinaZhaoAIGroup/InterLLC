import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from pymatgen.core.composition import Composition
from typing import Optional, List

import figures 
from xas_features import create_des_features, simple_descriptors_xas
from configurations import fxas_spectrum_type, fxas_len_map, simple_feature_names


class Dataset(Dataset):
    def __init__(self, sequences, properties, transform=None):
        """
        Args:
            sequences: 谱学序列数据，形状 [n_samples, sequence_length]
            properties: 属性数据，形状 [n_samples, 3]
        """
        self.sequences = torch.FloatTensor(sequences)
        self.properties = torch.FloatTensor(properties)
        self.transform = transform
        
        self.sequence_mean = torch.mean(self.sequences, dim=0)
        self.sequence_std = torch.std(self.sequences, dim=0) + 1e-8
        
        self.property_mean = torch.mean(self.properties, dim=0)
        self.property_std = torch.std(self.properties, dim=0) + 1e-8
        
        self.sequences = (self.sequences - self.sequence_mean) / self.sequence_std
        self.properties = (self.properties - self.property_mean) / self.property_std
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        properties = self.properties[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, properties
    
    def denormalize_properties(self, normalized_properties):
        if isinstance(normalized_properties, torch.Tensor):
            normalized_properties = normalized_properties.cpu().numpy()
        
        return normalized_properties * self.property_std.numpy() + self.property_mean.numpy()
    
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


class MultiModalDataset(Dataset):

    def __init__(self, 
                 xrd_sequences: Optional[np.ndarray] = None,     
                 xafs_dict: Optional[dict] = None,               
                 properties: Optional[np.ndarray] = None,
                 descriptor_names: List[str] = None,
                 material_ids: List[str] = None,
                 transform=None):
        if xrd_sequences is None and xafs_dict is None:
            raise ValueError("至少需要提供 XRD 或 XAFS 中的一个模态数据")
        
        if xrd_sequences is not None:
            self.n_samples = len(xrd_sequences)
        elif xafs_dict is not None:
            first_key = next(iter(xafs_dict.keys()))
            self.n_samples = len(xafs_dict[first_key])
        else:
            self.n_samples = len(properties)  
        
        self.properties = torch.FloatTensor(properties)
        self.transform = transform
        self.material_ids = material_ids
        
        if xrd_sequences is not None:
            self.xrd_sequences = torch.FloatTensor(xrd_sequences)
        else:
            self.xrd_sequences = None
        
        if xafs_dict is not None:
            self.xafs_dict = {}
            for name, features in xafs_dict.items():
                self.xafs_dict[name] = torch.FloatTensor(features)
            if descriptor_names is None:
                descriptor_names = list(xafs_dict.keys())
        else:
            self.xafs_dict = None
            descriptor_names = []
        
        self.descriptor_names = descriptor_names if descriptor_names else []
        print(f"属性数据形状: {self.properties.shape}")
        
        self.xrd_mean = None
        self.xrd_std = None
        self.property_mean = None
        self.property_std = None
    
    def normalize(self):
        if self.xrd_sequences is not None:
            self.xrd_mean = torch.mean(self.xrd_sequences, dim=0)
            self.xrd_std = torch.std(self.xrd_sequences, dim=0) + 1e-8
            self.xrd_sequences = (self.xrd_sequences - self.xrd_mean) / self.xrd_std
        
        self.property_mean = torch.mean(self.properties, dim=0)
        self.property_std = torch.std(self.properties, dim=0) + 1e-8
        self.properties = (self.properties - self.property_mean) / self.property_std
        
        print("数据标准化完成")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        result = {}
        
        if self.xrd_sequences is not None:
            xrd_seq = self.xrd_sequences[idx]
            if self.transform:
                xrd_seq = self.transform(xrd_seq)
            result['xrd'] = xrd_seq
        else:
            result['xrd'] = None   
        
        if self.xafs_dict is not None:
            xafs_sample_dict = {}
            for name in self.descriptor_names:
                feat = self.xafs_dict[name][idx]
                if self.transform:
                    feat = self.transform(feat)
                xafs_sample_dict[name] = feat
            result['xafs_descriptors'] = xafs_sample_dict
        else:
            result['xafs_descriptors'] = None   
        
        properties = self.properties[idx]
        if self.transform:
            properties = self.transform(properties)
        result['properties'] = properties
        
        if self.material_ids is not None:
            result['mid'] = self.material_ids[idx]
        
        return result

class DataProcessor:
    
    def __init__(self, data_dir="mp_abo3"):
        self.data_dir = data_dir
        self.materials = []
        self.spectral_data = {}  
        self.properties = {}     
        self.abx3 = self._get_abx3()
        self.a2bbx6 = self._get_a2bbx6()
        self.fxas_len_map = fxas_len_map
        self.fxas_spectrum_type = fxas_spectrum_type
        
    def filter_materials_with_all_data(self, data, xas='xafs', material_ids=None):
        print("开始筛选材料数据...")
        print("=" * 60)
        
        if material_ids == None:
            if data == 'abx3':
                allowed = self.abx3.keys()
            elif data == 'a2bbx6':
                allowed = self.a2bbx6.keys()
            elif data == 'all':
                allowed = self.abx3.keys() | self.a2bbx6.keys()
        elif material_ids and data == 'abx3':
            allowed = material_ids
        material_dirs = [d for d in os.listdir(self.data_dir) 
                        if os.path.isdir(os.path.join(self.data_dir, d)) and d in allowed]
        
        valid_materials = []
        invalid_reasons = {
            'missing_files': 0,
            'missing_properties': 0,
            'empty_spectra': 0,
            'invalid_data': 0,
            'not_abx3': 0
        }
        
        for material_id in tqdm(material_dirs, desc="筛选材料"):
            try:
                required_files = {
                    'base': f"{material_id}.json",
                    'xrd': f"{material_id}_xrd.json",
                    'xas': f"{material_id}_{xas}.json"
                }
                missing_files = []
                for key, filename in required_files.items():
                    filepath = os.path.join(self.data_dir, material_id, filename)
                    if not os.path.exists(filepath):
                        missing_files.append(key)
                
                if missing_files:
                    invalid_reasons['missing_files'] += 1
                    continue
                
                base_path = os.path.join(self.data_dir, material_id, required_files['base'])
                with open(base_path, 'r') as f:
                    base_data = json.load(f)
                formula = base_data.get('formula', '')
                if data == 'abx3':
                    if material_id not in self.abx3.keys():
                        invalid_reasons['not_abx3'] += 1
                        continue
            
                required_props = ['formation_energy', 'fermi_energy', 'band_gap']
                missing_props = []
                for prop in required_props:
                    if prop not in base_data or base_data[prop] is None:
                        missing_props.append(prop)
                
                if missing_props:
                    invalid_reasons['missing_properties'] += 1
                    continue
                
                formation_energy = float(base_data['formation_energy'])
                fermi_energy = float(base_data.get('fermi_energy', 0) or 0)
                band_gap = float(base_data['band_gap'])
                
                if not (-10 < formation_energy < 10):
                    invalid_reasons['invalid_data'] += 1
                    continue
                
                spectral_data = {}
                for key in ['xrd', 'xas']:
                    filepath = os.path.join(self.data_dir, material_id, required_files[key])
                    with open(filepath, 'r') as f:
                        spectral_data[key] = json.load(f)
                
                if (not spectral_data['xrd'] or 
                    not spectral_data['xas']):
                    invalid_reasons['empty_spectra'] += 1
                    continue
                
                valid_materials.append(material_id)
                self.spectral_data[material_id] = spectral_data
                self.properties[material_id] = {
                    'material_id': material_id,
                    'formula': base_data.get('formula', ''),
                    'formation_energy': formation_energy,
                    'fermi_energy': fermi_energy,
                    'band_gap': band_gap
                }
                
            except Exception as e:
                invalid_reasons['invalid_data'] += 1
                continue
        
        self.materials = valid_materials
        
        print(f"\n筛选完成!")
        print(f"总材料数: {len(material_dirs)}")
        print(f"有效材料数: {len(self.materials)} ({len(self.materials)/len(material_dirs)*100:.1f}%)")

        if self.materials:
            self._analyze_dataset(data)
        
        return self.materials
    
    def _get_abx3(self):
        ABX3 = {}
        with open('abx3.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    mid = parts[0]
                    formula = parts[1]
                    ABX3[mid] = formula

        print(f"包含ABX3材料数：{len(ABX3)}")
        return ABX3
    
    def _get_a2bbx6(self):
        A2BBX6 = {}
        with open('a2bbx6_mid_formula.txt', 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    mid = parts[0]
                    formula = parts[1]
                    A2BBX6[mid] = formula

        print(f"A2BBX6材料数：{len(A2BBX6)}")
        return A2BBX6
    
    def _is_abx3_perovskite(self, formula):
        """
        判断化学式是否符合ABX3钙钛矿结构
        支持的格式示例: CsPbI3, CH3NH3PbI3, Cs2AgBiBr6 (双钙钛矿)
        """
        if not formula:
            return False
        
        formula = formula.strip()

        exclusion_patterns = [
            r'\(H\d+O\d+\)',      # 水合氢离子团簇，如 (H6O5)
            r'·\d*H[₂2]?O',       # 结晶水
            r'hydrate',           # 水合物
            r'\[.*\]',            # 配合物
        ]
        for pattern in exclusion_patterns:
            if re.search(pattern, formula, re.IGNORECASE):
                return False
        
        non_perovskite_anions = ['PO4', 'SO4', 'CO3', 'SiO4', 'NO3', 'BO3']
        formula_upper = formula.upper()
        for anion in non_perovskite_anions:
            if anion in formula_upper:
                return False
        
        # 常见的ABX3钙钛矿
        pattern1 = re.compile(r'^([A-Z][a-z\d]*)([A-Z][a-z\d]*)([A-Z][a-z\d]*)_?3$', re.IGNORECASE)
        
        # A2BB'X6 (双钙钛矿)
        pattern2 = re.compile(r'^([A-Z][a-z\d]*)_?2([A-Z][a-z\d]*)([A-Z][a-z\d]*)([A-Z][a-z\d]*)_?6$', re.IGNORECASE)
        
        # 带有下划线的表示法: ABX_3 或 A_2BB'X_6
        pattern3 = re.compile(r'^([A-Z][a-z\d]*)_?([A-Z][a-z\d]*)_?([A-Z][a-z\d]*)_?3$', re.IGNORECASE)
        pattern4 = re.compile(r'^([A-Z][a-z\d]*)_?2_?([A-Z][a-z\d]*)_?([A-Z][a-z\d]*)_?([A-Z][a-z\d]*)_?6$', re.IGNORECASE)
        
        if (pattern1.match(formula) or 
            pattern2.match(formula) or 
            pattern3.match(formula) or 
            pattern4.match(formula)):
            return True
        
        elements = re.findall(r'([A-Z][a-z]?)(\d*\.?\d*)', formula)
        
        if not elements:
            return False
        
        element_count = len(elements)
        common_a_sites = ['Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca']
        common_b_sites = ['Pb', 'Sn', 'Ge', 'Ti', 'Zr', 'Hf', 'Nb', 'Ta']
        common_x_sites = ['I', 'Br', 'Cl', 'F', 'O']
        
        formula_lower = formula.lower()
        has_b_element = any(b_elem.lower() in formula_lower for b_elem in common_b_sites)
        has_x_element = any(x_elem.lower() in formula_lower for x_elem in common_x_sites)
        if not (has_b_element and has_x_element):
            return False
        
        has_three = '3' in formula
        has_six = '6' in formula
        
        if not (has_three or has_six):
            x_total = 0
            for elem, count in elements:
                if any(x_elem.lower() == elem.lower() for x_elem in common_x_sites):
                    x_total += float(count) if count else 1
            
            total_atoms = sum(float(count) if count else 1 for _, count in elements)
            if x_total > 0:
                ratio = x_total / (total_atoms - x_total)
                if not (1.5 <= ratio <= 3.5 or 4.5 <= ratio <= 7.5):
                    return False

        perovskite_indicators = ['pb', 'sn', 'ge', 'ti', 'i', 'br', 'cl', 'f', 'cs', 'rb']
        
        for indicator in perovskite_indicators:
            if indicator in formula_lower:
                if '3' in formula or '6' in formula:
                    return True
        
        return False
    
    def _get_perovskite_sites(self, formula):
        if not formula:
            return None, None, None

        formula = formula.strip()
        common_a_sites = [
            'Cs', 'Rb', 'K', 'Na', 'Li', 'Ba', 'Sr', 'Ca', 'Mg', 'Be',
            'MA', 'FA', 'CH3NH3', 'NH2CHNH2', 'CH(NH2)2', 'NH4',
            'CH3', 'C2H5', 'C3H7', 'C4H9', 'C5H11', 'C6H13',
            'DMA', 'TMA', 'EA', 'PMA', 'PEA', 'GA', 'AZ', 'IM',
            'La', 'Y', 'Sc', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Bi', 'Tl', 'Ag', 'Au', 'Pb', 'Sn', 'P'
        ]
        common_x_sites = [
            'I', 'Br', 'Cl', 'F',
            'O', 'S', 'Se', 'Te',
            'N', 'P', 'As', 'Sb', 'Bi',
            'H', 'C', 'B', 'Sn', 'In', 'Ge', 'Au', 'Si'               
        ]
        common_b_sites = [
            'Pb', 'Sn', 'Ge', 'Ti', 'Zr', 'Hf', 'Nb', 'Ta',
            'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Al', 'Ga', 'In', 'Sb', 'Bi', 'Ag', 'Mg', 'Cd',
            'Rh', 'Ir', 'Ru', 'Os', 'Pd', 'Pt', 'Au', 'Hg'
        ]
        electronegativity = {
            'H': 2.20, 'Li': 0.98, 'Be': 1.57, 'Na': 0.93, 'Mg': 1.31,
            'K': 0.82, 'Ca': 1.00, 'Rb': 0.82, 'Sr': 0.95, 'Cs': 0.79,
            'Ba': 0.89, 'Sc': 1.36, 'Y': 1.22, 'La': 1.10, 'Ce': 1.12,
            'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20,
            'Gd': 1.20, 'Tb': 1.20, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
            'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Ti': 1.54, 'Zr': 1.33,
            'Hf': 1.30, 'V': 1.63, 'Nb': 1.60, 'Ta': 1.50, 'Cr': 1.66,
            'Mo': 2.16, 'W': 2.36, 'Mn': 1.55, 'Re': 1.90, 'Fe': 1.83,
            'Ru': 2.20, 'Os': 2.20, 'Co': 1.88, 'Rh': 2.28, 'Ir': 2.20,
            'Ni': 1.91, 'Pd': 2.20, 'Pt': 2.28, 'Cu': 1.90, 'Ag': 1.93,
            'Au': 2.54, 'Zn': 1.65, 'Cd': 1.69, 'Hg': 2.00, 'Al': 1.61,
            'Ga': 1.81, 'In': 1.78, 'Tl': 1.62, 'C': 2.55, 'Si': 1.90,
            'Ge': 2.01, 'Sn': 1.96, 'Pb': 2.33, 'N': 3.04, 'P': 2.19,
            'As': 2.18, 'Sb': 2.05, 'Bi': 2.02, 'O': 3.44, 'S': 2.58,
            'Se': 2.55, 'Te': 2.10, 'F': 3.98, 'Cl': 3.16, 'Br': 2.96,
            'I': 2.66, 'B': 2.04
        }

        try:
            comp = Composition(formula)
        except:
            return None, None, None

        elements = {el: comp[el] for el in comp.elements}
        total_atoms = sum(elements.values())
        if total_atoms == 0:
            return None, None, None

        x_candidates = [el for el in elements if el.symbol in common_x_sites]
        if not x_candidates:
            return None, None, None

        x_candidates_sorted = sorted(x_candidates, key=lambda el: elements[el], reverse=True)
        primary_x = x_candidates_sorted[0].symbol
        x_sites = [primary_x]
        x_total = sum(elements[el] for el in elements if el.symbol == primary_x)
        elements = {el: amt for el, amt in elements.items() if el.symbol != primary_x}

        if not elements:
            return None, None, None

        non_x_total = sum(elements.values())
        ratio = x_total / non_x_total if non_x_total > 0 else 0

        a_sites = []
        b_sites = []

        if 1.2 <= ratio <= 1.8:
            if abs(non_x_total - 2) < 0.1:
                for el, amt in elements.items():
                    if el.symbol in common_a_sites:
                        a_sites.append(el.symbol)
                    else:
                        b_sites.append(el.symbol)

                if not a_sites and len(elements) == 2:
                    keys = list(elements.keys())
                    organic_a = [k.symbol for k in keys if k.symbol in ['MA','FA','CH3NH3','NH2CHNH2','CH(NH2)2','NH4','CH3','C2H5','C3H7','C4H9','C5H11','C6H13','DMA','TMA','EA','PMA','PEA','GA','AZ','IM']]
                    if len(organic_a) == 1:
                        a_sites = organic_a
                        b_sites = [k.symbol for k in keys if k.symbol not in a_sites]
                    else:
                        en1 = electronegativity.get(keys[0].symbol, 10)
                        en2 = electronegativity.get(keys[1].symbol, 10)
                        if en1 <= en2:
                            a_sites = [keys[0].symbol]
                            b_sites = [keys[1].symbol]
                        else:
                            a_sites = [keys[1].symbol]
                            b_sites = [keys[0].symbol]
                elif not a_sites and len(elements) == 1:
                    a_sites = [list(elements.keys())[0].symbol]
                    b_sites = [list(elements.keys())[0].symbol]
                elif a_sites and not b_sites:
                    if len(a_sites) > 1:
                        a_sites_sorted = sorted(a_sites, key=lambda sym: electronegativity.get(sym, 10), reverse=True)
                        b_sites = [a_sites_sorted[0]]
                        a_sites = a_sites_sorted[1:]
                    elif len(a_sites) == 1:
                        b_sites = a_sites.copy()

            elif abs(non_x_total - 4) < 0.1:
                a_candidates = []
                b_candidates = []
                for el, amt in elements.items():
                    if el.symbol in common_a_sites:
                        a_candidates.append(el.symbol)
                    else:
                        b_candidates.append(el.symbol)

                a_total = sum(elements[el] for el in elements if el.symbol in a_candidates)
                if abs(a_total - 2) < 0.1:
                    a_sites = list(set(a_candidates))
                    for el, amt in elements.items():
                        if el.symbol not in a_sites:
                            b_sites.append(el.symbol)
                else:
                    sorted_els = sorted(elements.items(), key=lambda x: x[1], reverse=True)
                    if abs(sorted_els[0][1] - 2) < 0.1:
                        a_sites = [sorted_els[0][0].symbol]
                        for el, amt in sorted_els[1:]:
                            b_sites.append(el.symbol)
                    else:
                        return None, None, None
            else:
                return None, None, None

        elif 0.2 <= ratio <= 0.35 and abs(non_x_total - 4) < 0.1:
            sorted_els = sorted(elements.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_els) == 2:
                a_sites = [sorted_els[0][0].symbol]
                b_sites = [sorted_els[1][0].symbol]
            else:
                a_candidates = [el for el, amt in sorted_els if amt == 3]
                b_candidates = [el for el, amt in sorted_els if amt == 1]
                if len(a_candidates) == 1 and len(b_candidates) == 1:
                    a_sites = a_candidates
                    b_sites = b_candidates
                else:
                    return None, None, None
        else:
            return None, None, None

        a_sites = list(set(a_sites))
        b_sites = list(set(b_sites))
        x_sites = list(set(x_sites))

        if a_sites and b_sites and x_sites:
            return (a_sites, b_sites, x_sites)
        else:
            return None, None, None

    def _analyze_dataset(self, data):
        print("\n" + "="*60)
        print("数据集统计信息")
        print("="*60)

        labels = ["Formation Energy", "Fermi Energy", "Bandgap"]
        units = ['eV', 'eV', 'eV']
        abx3_materials = [m for m in self.materials if m in self.abx3]
        a2bbx6_materials = [m for m in self.materials if m in self.a2bbx6]
        print(f"有效材料数：{len(abx3_materials)} ABX3, {len(a2bbx6_materials)} A2BBX6")

        categories = {}
        if data == 'abx3':
            categories = {'ABX3': abx3_materials}
        elif data == 'a2bbx6':
            categories = {'A2BBX6': a2bbx6_materials}
        elif data == 'all':
            categories = {'ABX3': abx3_materials, 'A2BBX6': a2bbx6_materials}
        categories = {k: v for k, v in categories.items() if v}

        stats_results = {}
        if data == 'all' and len(categories) == 2:
            category_data = {}  
            for cat_name, mids in categories.items():
                formation = [self.properties[m]['formation_energy'] for m in mids]
                fermi = [self.properties[m]['fermi_energy'] for m in mids]
                band = [self.properties[m]['band_gap'] for m in mids]
                if not (formation and fermi and band):
                    print(f"警告: {cat_name} 数据不完整，跳过")
                    continue
                category_data[cat_name] = [formation, fermi, band]

                print(f"\n{cat_name}数量：{len(mids)}")
                data_list = [formation, fermi, band]
                stats_names = ['形成能', '费米能', '能带']
                for name, d, unit in zip(stats_names, data_list, units):
                    arr = np.array(d)
                    min_val, max_val = np.min(arr), np.max(arr)
                    mean_val, var_val = np.mean(arr), np.var(arr)
                    median_val = np.median(arr)
                    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
                    print(f"\n{cat_name} - {name} ({unit}) 范围: [{min_val:.3f}, {max_val:.3f}]")
                    print(f"  均值±方差: {mean_val:.3f} ± {var_val:.3f}")
                    print(f"  中位数: {median_val:.3f}")
                    print(f"  四分位数: ({q1:.3f}, {q2:.3f}, {q3:.3f})")

                stats_results[cat_name] = {
                    prop: {'min': np.min(vals), 'max': np.max(vals),
                        'mean': np.mean(vals), 'var': np.var(vals),
                        'median': np.median(vals),
                        'q1': np.percentile(vals, 25),
                        'q2': np.percentile(vals, 50),
                        'q3': np.percentile(vals, 75)}
                    for prop, vals in zip(['formation_energy', 'fermi_energy', 'band_gap'],
                                        [formation, fermi, band])
                }

            if category_data:
                save_dir = 'results/'
                os.makedirs(save_dir, exist_ok=True)
                figures.plot_histograms_kde_combined(category_data, labels, units,
                                                    os.path.join(save_dir, "combined_histograms_kde.png"))
                figures.plot_boxplots_combined(category_data, os.path.join(save_dir, "combined_boxplots.png"))
                figures.plot_scatter_matrix_combined(category_data, os.path.join(save_dir, "combined_scatter_matrix.png"))
                figures.plot_pair_scatter_combined(category_data, os.path.join(save_dir, "combined_pair_scatter.png"))
                figures.plot_hexbin_density_combined(category_data, os.path.join(save_dir, "combined_hexbin_density.png"))

        else:
            for cat_name, mids in categories.items():
                print(f"{cat_name}数量：{len(mids)}")
                formation = [self.properties[m]['formation_energy'] for m in mids]
                fermi = [self.properties[m]['fermi_energy'] for m in mids]
                band = [self.properties[m]['band_gap'] for m in mids]

                data_list = [formation, fermi, band]
                stats_names = ['形成能', '费米能', '能带']
                for name, d, unit in zip(stats_names, data_list, units):
                    arr = np.array(d)
                    min_val, max_val = np.min(arr), np.max(arr)
                    mean_val, var_val = np.mean(arr), np.var(arr)
                    median_val = np.median(arr)
                    q1, q2, q3 = np.percentile(arr, [25, 50, 75])
                    print(f"\n{name} ({unit}) 范围: [{min_val:.3f}, {max_val:.3f}]")
                    print(f"  均值±方差: {mean_val:.3f} ± {var_val:.3f}")
                    print(f"  中位数: {median_val:.3f}")
                    print(f"  四分位数: ({q1:.3f}, {q2:.3f}, {q3:.3f})")

                stats_results[cat_name] = {
                    'formation_energy': {
                        'min': np.min(formation), 'max': np.max(formation),
                        'mean': np.mean(formation), 'var': np.var(formation),
                        'median': np.median(formation),
                        'q1': np.percentile(formation, 25),
                        'q2': np.percentile(formation, 50),
                        'q3': np.percentile(formation, 75)
                    },
                    'fermi_energy': {
                        'min': np.min(fermi), 'max': np.max(fermi),
                        'mean': np.mean(fermi), 'var': np.var(fermi),
                        'median': np.median(fermi),
                        'q1': np.percentile(fermi, 25),
                        'q2': np.percentile(fermi, 50),
                        'q3': np.percentile(fermi, 75)
                    },
                    'band_gap': {
                        'min': np.min(band), 'max': np.max(band),
                        'mean': np.mean(band), 'var': np.var(band),
                        'median': np.median(band),
                        'q1': np.percentile(band, 25),
                        'q2': np.percentile(band, 50),
                        'q3': np.percentile(band, 75)
                    }
                }

                save_dir = 'results/'
                base_name = f"{cat_name.lower()}_"
                figures.plot_histograms_kde(
                    [formation, fermi, band], labels, units,
                    os.path.join(save_dir, f"{base_name}histograms_kde.png")
                )
                figures.plot_boxplots(
                    formation, fermi, band,
                    os.path.join(save_dir, f"{base_name}boxplots.png")
                )
                figures.plot_scatter_matrix(
                    formation, fermi, band,
                    os.path.join(save_dir, f"{base_name}scatter_matrix.png")
                )
                figures.plot_pair_scatter(
                    formation, fermi, band,
                    os.path.join(save_dir, f"{base_name}pair_scatter.png")
                )
                figures.plot_hexbin_density(
                    formation, fermi, band,
                    os.path.join(save_dir, f"{base_name}hexbin_density.png")
                )
                print(f"  {base_name}统计数据已保存至: {save_dir}")

        print("\n" + "="*60)
        return stats_results
    
    def prepare_spectral_sequences(self, spectrum, cite, xas, des='cwt'):
        print("\n正在准备单一谱学序列数据...")

        sequences = []
        property_values = []
        material_ids = []
        formulas = []

        for material_id in tqdm(self.materials, desc="处理谱学数据"):
            try:
                spectral_data = self.spectral_data[material_id]
                props = self.properties[material_id]
                formula = self.properties[material_id]['formula']
                A_cites, B_cites, X_cites = self._get_perovskite_sites(formula)

                xrd_features = self.extract_xrd_features(spectral_data['xrd'], method='real_pattern') # XRD特征

                if cite == "A": cite_elements = A_cites
                elif cite == "B": cite_elements = B_cites
                elif cite == "X": cite_elements = X_cites
                else: cite_elements = None

                xas_features = create_des_features(spectral_data['xas'], des, 
                                                  simple_feature_names, cite_elements)

                combined_sequence = []
                if spectrum == "xrd":
                    target_length = len(xrd_features)
                    combined_sequence.extend(xrd_features)

                elif spectrum == f"{xas}":
                    target_length = len(xas_features)
                    combined_sequence.extend(xas_features)
                else:
                    print("multi-dimensional spectrums need to be considered")

                if len(combined_sequence) < target_length:
                    combined_sequence.extend([0.0] * (target_length - len(combined_sequence)))
                else:
                    combined_sequence = combined_sequence[:target_length]

                sequence_array = np.array(combined_sequence, dtype=np.float32)
                if np.std(sequence_array) > 0:
                    sequence_array = (sequence_array - np.mean(sequence_array)) / (np.std(sequence_array) + 1e-8)

                sequences.append(sequence_array)

                property_values.append([
                    props['formation_energy'],
                    props['fermi_energy'],
                    props['band_gap']
                ])
                material_ids.append(material_id)
                formulas.append(formula)

            except Exception as e:
                print(f"处理材料 {material_id} 时出错: {e}")
                continue

        sequences = np.array(sequences, dtype=np.float32)
        property_values = np.array(property_values, dtype=np.float32)

        print(f"谱学序列形状: {sequences.shape}")
        print(f"属性值形状: {property_values.shape}")

        return sequences, property_values, material_ids, formulas
    
    def prepare_multi_sequences(self, cite, fxas, xas_feat,
                                normalize_xrd='global',
                                normalize_xafs=True
                                ):  
        print(f"\n正在准备多谱数据...")
        
        xrd_sequences = []
        simple_features_list = []    
        xafs_dict = {name: [] for name in fxas}    
        property_values = []
        
        xrd_stats = {'sum': None, 'sum_sq': None, 'count': 0}
        abx3_used_character = {}
        
        for material_id in tqdm(self.materials, desc="处理多谱数据"):
            try:
                spectral_data = self.spectral_data[material_id]
                props = self.properties[material_id]
                formula = self.properties[material_id]['formula']
                A_cites, B_cites, X_cites = self._get_perovskite_sites(formula)
                abx3_used_character[material_id] = {"formula": formula, "A": A_cites, "B": B_cites, "X": X_cites, 
                                                    "formation_energy": props['formation_energy'], "fermi_energy": props['fermi_energy'], 
                                                    "band_gap": props['band_gap']}

                if xas_feat:
                    roles = {'A': A_cites, 'B': B_cites, 'X': X_cites}
                else:
                    roles = None
                
                xrd_features = self._extract_from_real_pattern(spectral_data['xrd'], maxseq=500)            
                xrd_features = np.array(xrd_features, dtype=np.float32)
                
                if xrd_stats['sum'] is None:
                    xrd_stats['sum'] = np.zeros_like(xrd_features)
                    xrd_stats['sum_sq'] = np.zeros_like(xrd_features)
                xrd_stats['sum'] += xrd_features
                xrd_stats['sum_sq'] += xrd_features ** 2
                xrd_stats['count'] += 1
                
                if cite == "A": 
                    cite_elements = A_cites  
                elif cite == "B": 
                    cite_elements = B_cites  
                elif cite == "X": 
                    cite_elements = X_cites
                else: 
                    cite_elements = None
                
                simple_features, xas_des_features = create_des_features(spectral_data['xas'], fxas, roles, 
                                                                      simple_feature_names, cite_elements)
                simple_features_list.append(simple_features)
                
                for des_name, des_vector in zip(fxas, xas_des_features):
                    des_vector = np.array(des_vector, dtype=np.float32)
                    
                    if normalize_xafs:  
                        norm = np.linalg.norm(des_vector) + 1e-8
                        des_vector = des_vector / norm
                    
                    xafs_dict[des_name].append(des_vector)
                
                xrd_sequences.append(xrd_features)

                band_gap_val = props['band_gap']
                property_values.append([
                    props['formation_energy'],
                    props['fermi_energy'],
                    band_gap_val
                ])
                
            except Exception as e:
                print(f"处理材料 {material_id} 时出错: {e}")
                continue
        
        xrd_sequences = np.array(xrd_sequences, dtype=np.float32)
        property_values = np.array(property_values, dtype=np.float32)
        
        for name in fxas:
            xafs_dict[name] = np.array(xafs_dict[name], dtype=np.float32)
        
        if normalize_xrd == 'global' and xrd_stats['count'] > 0:
            xrd_mean = xrd_stats['sum'] / xrd_stats['count']
            xrd_std = np.sqrt(xrd_stats['sum_sq'] / xrd_stats['count'] - xrd_mean ** 2) + 1e-8
            
            xrd_sequences = (xrd_sequences - xrd_mean) / xrd_std
        
        if xas_feat and simple_features_list:
            simple_features_array = np.array(simple_features_list, dtype=np.float32) 
            
            min_vals = np.min(simple_features_array, axis=0)
            max_vals = np.max(simple_features_array, axis=0)
            ranges = max_vals - min_vals
            ranges[ranges == 0] = 1.0
            
            normalized_simple = (simple_features_array - min_vals) / ranges
            
            for idx, name in enumerate(simple_feature_names):
                xafs_dict[name] = normalized_simple[:, idx]  

        print(f"\n多谱数据统计:")
        print(f"XRD序列形状: {xrd_sequences.shape}")
        print(f"属性值形状: {property_values.shape}")
        
        self._save_used_abx3(abx3_used_character, "abx3_used.txt")

        return xrd_sequences, xafs_dict, property_values
    
    def _save_used_abx3(self, data, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for material_id, item in data.items():
                line = '\t'.join([
                    str(material_id),
                    str(item.get('formula', '')),
                    str(item.get('A', '')),
                    str(item.get('B', '')),
                    str(item.get('X', '')),
                    str(item.get('formation_energy', '')),
                    str(item.get('fermi_energy', '')),
                    str(item.get('band_gap', ''))
                ])
                f.write(line + '\n')
        print(f"used abx3数据保存到{filename}")
    
    def extract_xrd_features(self, xrd_data, maxseq=500, method='real_pattern'):
        return self._extract_from_real_pattern(xrd_data, maxseq)

    def _extract_from_real_pattern(self, xrd_data, maxseq=500):
        features = np.zeros(maxseq, dtype=np.float32)
        if 'all_peaks' not in xrd_data:
            return features
        
        all_peaks = xrd_data['all_peaks']
        if 'two_theta' not in all_peaks or 'intensity' not in all_peaks:
            return features
        
        two_theta = np.array(all_peaks['two_theta'], dtype=np.float32)
        intensity = np.array(all_peaks['intensity'], dtype=np.float32)
        
        if len(two_theta) < 5 or len(intensity) < 5:
            return features

        theta_min = max(5.0, np.min(two_theta) - 5.0)
        theta_max = min(85.0, np.max(two_theta) + 5.0)
        
        two_theta_range = np.linspace(theta_min, theta_max, maxseq)
        
        if np.max(intensity) > 0:
            intensity = intensity / np.max(intensity)
        
        reconstructed = np.zeros_like(two_theta_range)
        
        for tth, intens in zip(two_theta, intensity):
            if intens > 0.01: 
                if 'd_spacing' in all_peaks:
                    idx = np.where(two_theta == tth)[0]
                    if len(idx) > 0:
                        d_spacing = all_peaks['d_spacing'][idx[0]]
                
                        fwhm = 0.1 + 0.3 * (5.0 / (d_spacing + 1.0))
                    else:
                        fwhm = 0.2
                else:
                    fwhm = 0.2
                
                sigma = fwhm / 2.355
                
                gaussian = intens * np.exp(-(two_theta_range - tth)**2 / (2 * sigma**2))
                reconstructed += gaussian
        
        if np.max(reconstructed) > 0:
            reconstructed = reconstructed / np.max(reconstructed)
            from scipy.ndimage import gaussian_filter1d
            reconstructed = gaussian_filter1d(reconstructed, sigma=1.0)
        
        if len(reconstructed) < maxseq:
            padded = np.zeros(maxseq, dtype=np.float32)
            start_idx = (maxseq - len(reconstructed)) // 2
            padded[start_idx:start_idx+len(reconstructed)] = reconstructed
            reconstructed = padded
        elif len(reconstructed) > maxseq:
            start_idx = (len(reconstructed) - maxseq) // 2
            reconstructed = reconstructed[start_idx:start_idx+maxseq]
        
        return reconstructed

    def _extract_peak_features(self, xrd_data, maxseq=500):
        features = []
        
        if 'significant_peaks' in xrd_data and len(xrd_data['significant_peaks']) > 0:
            sig_peaks = xrd_data['significant_peaks']
            
            num_peaks = len(sig_peaks)
            num_total = xrd_data.get('num_peaks', 0)
            num_sig = xrd_data.get('num_significant', 0)
            
            features.append(num_peaks)
            features.append(num_total)
            features.append(num_sig)
            features.append(num_sig / max(num_total, 1))  # 显著峰比例
            
            intensities = []
            two_thetas = []
            d_spacings = []
            
            for peak in sig_peaks:
                if 'intensity' in peak:
                    intensities.append(peak['intensity'])
                if 'two_theta' in peak:
                    two_thetas.append(peak['two_theta'])
                if 'd_spacing' in peak:
                    d_spacings.append(peak['d_spacing'])
            
            if intensities:
                intensities = np.array(intensities, dtype=np.float32)
                features.extend([
                    np.mean(intensities), np.std(intensities),
                    np.min(intensities), np.max(intensities),
                    np.percentile(intensities, 25), np.percentile(intensities, 50), np.percentile(intensities, 75)
                ])
            else:
                features.extend([0.0] * 7)
            
            if two_thetas:
                two_thetas = np.array(two_thetas, dtype=np.float32)
                features.extend([
                    np.mean(two_thetas), np.std(two_thetas),
                    np.min(two_thetas), np.max(two_thetas),
                    (np.max(two_thetas) - np.min(two_thetas)) 
                ])
            else:
                features.extend([0.0] * 5)
            
            if d_spacings:
                d_spacings = np.array(d_spacings, dtype=np.float32)
                features.extend([
                    np.mean(d_spacings), np.std(d_spacings),
                    np.min(d_spacings), np.max(d_spacings),
                    1.0 / np.mean(d_spacings) if np.mean(d_spacings) > 0 else 0  
                ])
            else:
                features.extend([0.0] * 5)
            
            if len(two_thetas) > 2:
                sorted_theta = np.sort(two_thetas)
                spacings = np.diff(sorted_theta)
                if len(spacings) > 1:
                    spacing_std = np.std(spacings)
                    spacing_mean = np.mean(spacings)
                    features.extend([
                        spacing_mean, spacing_std,
                        spacing_std / max(spacing_mean, 1e-6)  
                    ])
                    
                    autocorr = self._calculate_autocorrelation(spacings)
                    features.append(autocorr)
                else:
                    features.extend([0.0] * 4)
            else:
                features.extend([0.0] * 4)
            
            hkl_counts = []
            for peak in sig_peaks:
                if 'hkl' in peak and isinstance(peak['hkl'], list):
                    hkl_counts.append(len(peak['hkl']))
            
            if hkl_counts:
                features.extend([
                    np.mean(hkl_counts), np.max(hkl_counts),
                    sum(1 for c in hkl_counts if c > 1)  
                ])
            else:
                features.extend([0.0] * 3)
        
        else:
            features = [0.0] * (7 + 5 + 5 + 4 + 3)  
        
        if len(features) > maxseq:
            features = features[:maxseq]
        elif len(features) < maxseq:
            features.extend([0.0] * (maxseq - len(features)))
        
        return np.array(features, dtype=np.float32)
    
    def feature_engineering(self):  
        xas_features = [] 
        xrd_features = []  
        property_values = []
        feature_names = ['B_X_bond_length', 'coordination_number', 'disorder', 'A_site_displacement', 'X_X_average_distance', 
                         'B_X_covalency', 'xanes_pre_edge_integral', 'white_line_position', 'white_line_fwhm', 'ft_peak_ratio', 
                         'k_space_decay_rate' 
                         ]
        
        for material_id in tqdm(self.materials, desc="Featurization"):
            try:
                spectral_data = self.spectral_data[material_id]
                props = self.properties[material_id]
                formula = self.properties[material_id]['formula']
                A_cites, B_cites, X_cites = self._get_perovskite_sites(formula)
                roles = {'A': A_cites, 'B': B_cites, 'X': X_cites}
                
                xrd_feature = self._extract_peak_features(spectral_data['xrd'], maxseq=24)
                xas_feature = simple_descriptors_xas(spectral_data['xas'], roles, feature_names)
                xrd_features.append(xrd_feature)
                xas_features.append(xas_feature)
                
                property_values.append([
                    props['formation_energy'],
                    props['fermi_energy'],
                    props['band_gap']
                ])
                
            except Exception as e:
                print(f"处理材料 {material_id} 时出错: {e}")
                continue
        
        xrd_features = np.array(xrd_features, dtype=np.float32)
        xas_features = np.array(xas_features, dtype=np.float32)
        property_values = np.array(property_values, dtype=np.float32)

        min_vals = np.min(xrd_features, axis=0)
        max_vals = np.max(xrd_features, axis=0)
        ranges = max_vals - min_vals

        ranges[ranges == 0] = 1.0
        xrd_normalized = (xrd_features - min_vals) / ranges

        min_vals = np.min(xas_features, axis=0)
        max_vals = np.max(xas_features, axis=0)
        ranges = max_vals - min_vals

        ranges[ranges == 0] = 1.0
        xas_normalized = (xas_features - min_vals) / ranges
        
        return xrd_normalized, xas_normalized, property_values

    def _calculate_autocorrelation(self, spacings, max_lag=None):

        if len(spacings) < 3:
            return 0.0
        
        if max_lag is None:
            max_lag = min(5, len(spacings) // 2)
        
        try:
            spacings_norm = (spacings - np.mean(spacings)) / (np.std(spacings) + 1e-8)
            
            autocorr = []
            for lag in range(1, max_lag + 1):
                if lag < len(spacings_norm):
                    corr = np.corrcoef(spacings_norm[:-lag], spacings_norm[lag:])[0, 1]
                    autocorr.append(corr if not np.isnan(corr) else 0.0)
            
            return np.mean(autocorr) if autocorr else 0.0
        except:
            return 0.0

    def _extract_advanced_features(self, xrd_data):
        features = []
        
        if 'significant_peaks' not in xrd_data:
            return features
        
        sig_peaks = xrd_data['significant_peaks']
        if len(sig_peaks) < 3:
            return features
        
        try:
            d_spacings = []
            hkl_sets = []
            
            for peak in sig_peaks:
                if 'd_spacing' in peak and 'hkl' in peak and peak['hkl']:
                    d_spacings.append(peak['d_spacing'])
                    hkl = peak['hkl'][0]
                    hkl_sets.append((hkl.get('h', 0), hkl.get('k', 0), hkl.get('l', 0)))
            
            if len(d_spacings) >= 3 and len(hkl_sets) >= 3:
                lattice_params = []
                for d, (h, k, l) in zip(d_spacings[:5], hkl_sets[:5]):
                    if h**2 + k**2 + l**2 > 0:
                        a = d * np.sqrt(h**2 + k**2 + l**2)
                        lattice_params.append(a)
                
                if lattice_params:
                    features.append(np.mean(lattice_params))
                    features.append(np.std(lattice_params))
                else:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            intensities = [p.get('intensity', 0) for p in sig_peaks]
            if intensities:
                sorted_int = np.sort(intensities)[::-1]  
                
                if len(sorted_int) >= 2 and sorted_int[1] > 0:
                    features.append(sorted_int[0] / sorted_int[1])
                else:
                    features.append(1.0)
                
                if np.sum(sorted_int) > 0:
                    normalized = sorted_int / np.sum(sorted_int)
                    entropy = -np.sum(normalized * np.log(normalized + 1e-8))
                    features.append(entropy / np.log(len(normalized) + 1e-8))  
                else:
                    features.append(0.0)
            else:
                features.extend([1.0, 0.0])
            
            two_thetas = [p.get('two_theta', 0) for p in sig_peaks]
            if two_thetas:
                two_thetas = np.array(two_thetas)
                low_angle_ratio = np.sum(two_thetas < 30) / len(two_thetas)
                features.append(low_angle_ratio)
                high_angle_ratio = np.sum(two_thetas > 60) / len(two_thetas)
                features.append(high_angle_ratio)
            else:
                features.extend([0.0, 0.0])
        
        except Exception as e:
            print(f"提取高级特征时出错: {e}")
            features.extend([0.0] * 6)  
        
        return features
    
    def _save_elements_to_file(self, data_dict, filepath):
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                total_elements = 0
                element_counts = {}
                
                for values in data_dict.values():
                    total_elements += len(values)
                    for element in values:
                        element_counts[element] = element_counts.get(element, 0) + 1
                
                f.write("统计信息:\n")
                f.write(f"  总材料数: {len(data_dict)}\n")
                f.write(f"  总元素出现次数: {total_elements}\n")
                f.write(f"  不重复元素种类: {len(element_counts)}\n\n")
                
                f.write("=" * 60 + "\n")
                f.write("材料元素对应表:\n")
                f.write("=" * 60 + "\n\n")
                
                for i, (key, values) in enumerate(data_dict.items(), 1):
                    values_str = f'\t'.join(values)
                    f.write(f"{key:<20}\t{values_str}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("元素出现频率统计:\n")
                f.write("=" * 60 + "\n\n")
                
                for element, count in sorted(element_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(data_dict) * 100
                    f.write(f"  {element}: {count}次 ({percentage:.1f}%)\n")
            
            print(f"详细格式文件已保存到: {filepath}")
            return True
        
        except Exception as e:
            print(f"保存文件失败: {e}")
            return False





