
import torch.nn as nn


elements_K_energies = { 
    # 低Z元素 (Z<20)
    'O': 543.1, 'Mg': 1303, 'Al': 1559, 'Si': 1839, 'P': 2149, 'S': 2472,
    # 3d 过渡金属
    'K': 3607, 'Ca': 4038, 'Sc': 4492, 'Ti': 4966, 'V': 5465, 'Cr': 5989, 'Mn': 6539,
    'Fe': 7112, 'Co': 7709, 'Ni': 8333, 'Cu': 8979, 'Zn': 9659,
    # 4d/4p 元素
    'Ga': 10367, 'Ge': 11103, 'As': 11867, 'Se': 12658, 'Br': 13474, 'Rb': 15200, 'Sr': 16105,
    'Y': 17038, 'Zr': 17998, 'Nb': 18986, 'Mo': 20000, 'Tc': 21044, 'Ru': 22117, 'Rh': 23220,
    'Pd': 24350, 'Ag': 25514, 'Cd': 26711, 'In': 27940, 'Sn': 29200, 'Sb': 30491, 'Te': 31814, 'I': 33169,
    # 5d/5p/镧系后元素
    'Cs': 35985, 'Ba': 37444,
    # **镧系元素（Lanthanides）的K边能量非常高（>38 keV），通常不常用。更常用的是其L边。**
    'La': 38925, 'Ce': 40443, 'Pr': 41991, 'Nd': 43569, 'Pm': 45184, 'Sm': 46834,
    'Eu': 48519, 'Gd': 50239, 'Tb': 51996, 'Dy': 53789, 'Ho': 55618, 'Er': 57486,
    'Tm': 59390, 'Yb': 61332, 'Lu': 63314, 
    # 5d过渡金属及后过渡金属
    'Hf': 65351, 'Ta': 67416, 'W': 69525, 'Re': 71676, 'Os': 73871, 'Ir': 76111,
    'Pt': 78395, 'Au': 80725, 'Hg': 83102, 'Tl': 85530, 'Pb': 88005, 'Bi': 90524,
    # 锕系元素（Actinides）K边极高，实验罕见，多用L或M边
    'Th': 109651, 'U': 115606,
}

activation_map = {
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'leakyrelu': nn.LeakyReLU(0.1),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'prelu': nn.PReLU(),
    }

fxas_len_map = {'cdf': 50, 'cwt': 300,       
                'wacsf': 100, 'soap2': 200, 'pdos': 150, 
                'msr1': 100,  
    }

fxas_spectrum_type = {
    # XAFS全谱（适合整个能量范围）
    'cwt': 'XAFS', 'cdf': 'XAFS',
    # XANES区域
    'wacsf': 'XANES', 'soap2': 'XANES', 'pdos': 'XANES',
    # EXAFS区域
    'msr1': 'EXAFS', 
}

group_feat_map = {
    'cwt': 'Signal', 'cdf': 'Signal',
    'wacsf': 'XANES', 'soap2': 'XANES', 'pdos': 'XANES',
    'msr1': 'EXAFS', 
    'B_X_bond_length': 'Addition',         
    'A_site_displacement': 'Addition',      
    'X_X_average_distance': 'Addition',
}

xafs_group = ['cwt', 'cdf']
xanes_group = ['wacsf', 'soap2', 'pdos']  
exafs_group = ['msr1'] 

simple_feature_names = [
    'B_X_bond_length',          # B-X键长
    'A_site_displacement',      # A位阳离子位移
    'X_X_average_distance',     # X-X平均距离
    ]

simple_feature_source = {
    'B_X_bond_length': 'B',           
    'A_site_displacement': 'A',       
    'X_X_average_distance': 'X',      
}