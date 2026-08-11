import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

def plot_boxplots(formation, fermi, band, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    box_data = [formation, fermi, band]
    labels = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True,
                    medianprops={'color': 'red', 'linewidth': 1.5},
                    meanprops={'color': 'blue', 'linestyle': '--'})
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Boxplot Comparing Dispersion and Outliers\n(Fermi & Bandgap)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_boxplots_combined(category_data, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(category_data.keys())
    n_cats = len(categories)
    n_props = 3  
    positions = []
    labels = []
    for i in range(n_props):
        for j in range(n_cats):
            positions.append(i * (n_cats+1) + j + 1)
            labels.append(f"{['Formation','Fermi','Band'][i]}\n{categories[j]}")
    
    all_data = []
    for i in range(n_props):
        for cat in categories:
            all_data.append(category_data[cat][i])
    
    bp = ax.boxplot(all_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True, meanline=True,
                    medianprops={'color': 'red', 'linewidth': 1.5},
                    meanprops={'color': 'blue', 'linestyle': '--'})
    colors = {'ABX3': 'steelblue', 'A2BBX6': 'salmon'}
    for box, cat in zip(bp['boxes'], [c for _ in range(n_props) for c in categories]):
        box.set_facecolor(colors[cat])
    
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Energy (eV)')               
    ax.set_title('Boxplot Comparison (ABX3 vs A2BBX6)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_scatter_matrix(formation, fermi, band, save_path=None):
    df = pd.DataFrame({
        'Formation Energy (eV/atom)': formation,
        'Fermi Energy (eV)': fermi,
        'Bandgap (eV)': band
    })
    pair_grid = sns.pairplot(df, diag_kind='hist', plot_kws={'alpha': 0.6, 's': 20})
    pair_grid.fig.suptitle('Scatter Plot Matrix: Pairwise Relationships', y=1.02)
    if save_path:
        pair_grid.fig.savefig(save_path, dpi=300)
    plt.close(pair_grid.fig)

def plot_histograms_kde(data_list, labels, units, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    hist_color = '#9FC0D6'   
    kde_color = '#0072B2'    
    for ax, d, label in zip(axes, data_list, labels):
        sns.histplot(d, stat='density', ax=ax,
                     color=hist_color, edgecolor='black', alpha=0.6)
        sns.kdeplot(d, ax=ax, color=kde_color, linewidth=1.2, cut=0)
        ax.set_xlabel(f'{label} (eV)', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        ax.grid(False)          
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_pair_scatter(formation, fermi, band, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [(formation, band, 'Formation Energy (eV)', 'Bandgap (eV)'),
             (fermi, band, 'Fermi Energy (eV)', 'Bandgap (eV)'),
             (formation, fermi, 'Formation Energy (eV)', 'Fermi Energy (eV)')]

    point_color = '#58A8D7'
    for ax, (x, y, xlabel, ylabel) in zip(axes, pairs):
        ax.scatter(x, y, alpha=0.7, edgecolors='k', s=35, linewidth=0.5,
                   color=point_color)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_scatter_matrix_combined(category_data, save_path=None):
    df_list = []
    for cat_name, (form, fermi, band) in category_data.items():
        temp_df = pd.DataFrame({
            'Formation Energy (eV/atom)': form,
            'Fermi Energy (eV)': fermi,
            'Bandgap (eV)': band,
            'Category': cat_name
        })
        df_list.append(temp_df)
    df = pd.concat(df_list, ignore_index=True)
    
    pair_grid = sns.pairplot(df, hue='Category', diag_kind='hist',
                             plot_kws={'alpha': 0.6, 's': 20},
                             palette={'ABX3': 'steelblue', 'A2BBX6': 'salmon'})
    pair_grid.fig.suptitle('Scatter Plot Matrix: ABX3 vs A2BBX6', y=1.02)
    if save_path:
        pair_grid.fig.savefig(save_path, dpi=300)
    plt.close(pair_grid.fig)


def plot_hexbin_density(formation, fermi, band, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [(formation, band, 'Formation vs Band'),
             (fermi, band, 'Fermi vs Band'),
             (formation, fermi, 'Formation vs Fermi')]
    for ax, (x, y, title) in zip(axes, pairs):
        hb = ax.hexbin(x, y, gridsize=30, cmap='Blues', mincnt=1)
        ax.set_xlabel('X (eV)')
        ax.set_ylabel('Y (eV)')
        ax.set_title(f'{title}\nHexbin Density Plot')
        plt.colorbar(hb, ax=ax, label='Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_hexbin_density_combined(category_data, save_path=None):
    fig, axes = plt.subplots(3, 2, figsize=(12, 15)) 
    pairs = [(0,2,'Formation Energy (eV)', 'Bandgap (eV)'),
             (1,2,'Fermi Energy (eV)', 'Bandgap (eV)'),
             (0,1,'Formation Energy (eV)', 'Fermi Energy (eV)')]
    
    for row, (x_idx, y_idx, xlabel, ylabel) in enumerate(pairs):
        for col, (cat_name, (form, fermi, band)) in enumerate(category_data.items()):
            ax = axes[row, col]
            x_data = [form, fermi, band][x_idx]
            y_data = [form, fermi, band][y_idx]
            hb = ax.hexbin(x_data, y_data, gridsize=30, cmap='Blues', mincnt=1)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{cat_name}: {xlabel} vs {ylabel}')
            plt.colorbar(hb, ax=ax, label='Count')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_histograms_kde_combined(category_data, labels, units, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fill_colors = {'ABX3': '#9FC0D6', 'A2BBX6': '#F7E0CF'}
    kde_colors = {'ABX3': '#0072B2',   
                  'A2BBX6': '#D55E00'} 
    
    for idx, (ax, label) in enumerate(zip(axes, labels)):
        for cat_name, data_list in category_data.items():
            d = data_list[idx]
            sns.histplot(d, ax=ax, color=fill_colors[cat_name],
                         edgecolor='black', alpha=0.6,
                         stat='density', label=cat_name)
            sns.kdeplot(d, ax=ax, color=kde_colors[cat_name], linewidth=1.2)
        ax.set_xlabel(f'{label} (eV)')          
        ax.set_ylabel('Density')               
        ax.grid(False)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_histograms_kde_combined(category_data, labels, units, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fill_colors = {'ABX3': '#9FC0D6', 'A2BBX6': '#EDBE99'}
    kde_colors = {'ABX3': '#3F719D', 'A2BBX6': '#DD542F'} 

    all_vals = {0: [], 1: [], 2: []}
    for cat_name, data_list in category_data.items():
        for prop_idx in range(3):
            all_vals[prop_idx].extend(data_list[prop_idx])
    xlims = {}
    for prop_idx in range(3):
        min_val = min(all_vals[prop_idx])
        max_val = max(all_vals[prop_idx])

        pad = (max_val - min_val) * 0.05
        xlims[prop_idx] = (min_val - pad, max_val + pad)

    for idx, (ax, label) in enumerate(zip(axes, labels)):
        ax.set_xlim(xlims[idx])
        for cat_name, data_list in category_data.items():
            d = data_list[idx]
            sns.histplot(d, ax=ax, color=fill_colors[cat_name],
                         edgecolor='black', alpha=0.6,
                         stat='density', label=cat_name)
            sns.kdeplot(d, ax=ax, color=kde_colors[cat_name], linewidth=1.5, cut=0)
        ax.set_xlabel(f'{label} (eV)')          
        ax.set_ylabel('Density')               
        ax.grid(False)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_pair_scatter_combined(category_data, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [(0,2,'Formation Energy (eV)', 'Bandgap (eV)'),
             (1,2,'Fermi Energy (eV)', 'Bandgap (eV)'),
             (0,1,'Formation Energy (eV)', 'Fermi Energy (eV)')]

    colors = {'ABX3': '#58A8D7', 'A2BBX6': '#E39C63'} 

    all_vals = {0: [], 1: [], 2: []}
    for cat_name, data_list in category_data.items():
        for prop_idx in range(3):
            all_vals[prop_idx].extend(data_list[prop_idx])
    lims = {}
    for prop_idx in range(3):
        min_val = min(all_vals[prop_idx])
        max_val = max(all_vals[prop_idx])
        pad = (max_val - min_val) * 0.05
        lims[prop_idx] = (min_val - pad, max_val + pad)

    for ax, (x_idx, y_idx, xlabel, ylabel) in zip(axes, pairs):

        ax.set_xlim(lims[x_idx])
        ax.set_ylim(lims[y_idx])
        for cat_name, (form, fermi, band) in category_data.items():
            x_data = [form, fermi, band][x_idx]
            y_data = [form, fermi, band][y_idx]
            ax.scatter(x_data, y_data, alpha=0.7, edgecolors='k', s=35, linewidth=0.5,
                       color=colors[cat_name], label=cat_name)
        ax.set_xlabel(xlabel)          
        ax.set_ylabel(ylabel)          
        ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)
        ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def load_predictions(filepath):
    try:
        data = np.loadtxt(filepath, comments='#', delimiter=None)
    except ValueError:
        data = np.loadtxt(filepath, skiprows=1, delimiter=None)
    
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    true = data[:, 0]
    pred = data[:, 1]
    return true, pred

def plot_scatter_for_properties():

    models = {'InterLLC': 'InterLLC',
              'MLP_xafs': 'MLP_xafs',
              'CNN_xrd': 'CNN_xrd'}

    properties = [
        ('formation_energy', 'Formation energy', 'eV'),
        ('fermi_energy', 'Fermi energy', 'eV'),
        ('band_gap', 'Bandgap', 'eV')
    ]

    metrics_dict = {
        'formation_energy': {
            'InterLLC': (0.204, 0.713),
            'MLP_xafs': (0.2496, 0.5319),
            'CNN_xrd': (0.292, 0.462)
        },
        'fermi_energy': {
            'InterLLC': (0.236, 0.582),
            'MLP_xafs': (0.3007, 0.383),
            'CNN_xrd': (0.295, 0.434)
        },
        'band_gap': {
            'InterLLC': (0.223, 0.516),
            'MLP_xafs': (0.2834, 0.353),
            'CNN_xrd': (0.315, 0.266)
        }			
    }

    base_paths = {
        'InterLLC': 'res/InterLLC_ABX3_20260430_095349/scatter_plots/{prop}_predictions.txt',
        'MLP_xafs': 'res/MLP_xafs_20260430_211154/_{prop}_targets_predictions.txt',
        'CNN_xrd': 'res/CNN_xrd_20260428_161042/_{prop}_targets_predictions.txt'
    }
    
    colors = {
        'InterLLC': '#E39C63',   
        'MLP_xafs': '#5A8B3B',   
        'CNN_xrd': '#58A8D7',        
    }
    markers = {
        'InterLLC': 'o',
        'MLP_xafs': 's',
        'CNN_xrd': '^'
    }

    output_dir = 'figures/'
    os.makedirs(output_dir, exist_ok=True)
    output_format = 'png'
    dpi = 300

    for prop_name, prop_label, unit in properties:
        fig, ax = plt.subplots(figsize=(5, 7))
        
        all_vals = []   
        
        model_data = []
        
        for model_key, model_label in models.items():
            filepath = base_paths[model_key].format(prop=prop_name)
            if not os.path.exists(filepath):
                print(f"警告: 文件不存在 - {filepath}")
                continue
            
            true, pred = load_predictions(filepath)
            all_vals.extend(true)
            all_vals.extend(pred)
            mae, r2 = metrics_dict[prop_name][model_key]
            label_text = f"{model_label}: MAE={mae:.3f}, R²={r2:.3f}"
            model_data.append({
                'model_key': model_key,
                'true': true,
                'pred': pred,
                'label': label_text,
                'color': colors[model_key],
                'marker': markers[model_key]
            })
        for data in sorted(model_data, key=lambda x: 0 if x['model_key'] == 'InterLLC' else 1):
            if data['model_key'] == 'InterLLC':
                continue
            ax.scatter(data['true'], data['pred'],
                      label=data['label'],
                      color=data['color'], marker=data['marker'],
                      alpha=0.7, s=50, edgecolors='w', linewidth=0.5)
        
        for data in model_data:
            if data['model_key'] == 'InterLLC':
                ax.scatter(data['true'], data['pred'],
                          label=data['label'],
                          color=data['color'], marker=data['marker'],
                          alpha=0.9, s=55, edgecolors='w', linewidth=0.8, zorder=10)  
        
        if prop_name == 'band_gap':
            min_val, max_val = -0.5, 8.5
            ax.plot([0, 8], [0, 8], 'k--', alpha=0.6, linewidth=1.5, zorder=1)  
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
        else:
            if all_vals:
                min_val = min(all_vals)
                max_val = max(all_vals)
                pad = (max_val - min_val) * 0.1
                min_val -= pad
                max_val += pad
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.6, linewidth=1.5, zorder=1)  # 对角线放在底层
                ax.set_xlim(min_val, max_val)
                ax.set_ylim(min_val, max_val)
        
        ax.set_xlabel(f'True {prop_label} ({unit})', fontsize=14)
        ax.set_ylabel(f'Predicted value ({unit})', fontsize=14)
        ax.legend(loc='upper left', fontsize=9, frameon=True, fancybox=True, shadow=False)
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'scatter_{prop_name}_comparison.{output_format}')
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)
        print(f"已保存: {output_path}")
        plt.close(fig)
        

def plot_model_performance_1(df, dataset_name, output_prefix='model_performance', fontsize=12, ha='center'):
    properties = ['Formation', 'Fermi', 'Band']
    prop_labels = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    colors = ['#BAD2E1', '#58A8D7', '#F7E0CF', '#A9C37F', '#E39C63']
    models_order = ['RF_xrd', 'CNN_xrd', 'AEMLP_xafs', 'MLP_xafs', 'InterLLC']
    metrics = [('MAE', 'MAE (eV)'), ('R2', r'R²')]

    n_models = len(models_order)
    x = np.arange(len(prop_labels)) 
    
    bar_width = 0.15
    total_width = (n_models - 1) * bar_width
    offsets = np.linspace(-total_width / 2, total_width / 2, n_models)

    for metric_name, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))

        for i, model_name in enumerate(models_order):
            values = []
            for prop in prop_labels:
                prop_key = prop.split()[0]  
                if prop_key == 'Bandgap':
                    prop_key = 'Band'
                model_idx = df[df['Model'] == model_name].index[0]
                values.append(df.loc[model_idx, f'{prop_key}_{metric_name}'])
            
            bar_positions = x + offsets[i]
            bars = ax.bar(bar_positions, values, bar_width, label=model_name, color=colors[i],
                          edgecolor='black', linewidth=0.5, alpha=0.85)

            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    if metric_name == 'MAE':
                        offset = 0.01
                        precision = 3
                    else:  # R²
                        offset = 0.015
                        precision = 3
                    
                    ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                            f'{height:.{precision}f}',
                            ha='center', va='bottom',
                            rotation=90,
                            fontsize=fontsize-2,  
                            color='black')

        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(prop_labels, fontsize=fontsize, rotation=0, ha=ha)
        
        if metric_name == 'MAE':
            ax.set_ylim(0, 0.86)
            ax.set_yticks(np.arange(0, 0.9, 0.1)) 
        else:  
            ax.set_ylim(0, 0.92)
            ax.set_yticks(np.arange(0, 1.0, 0.1))  
        
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(top=False, right=False)
        
        left_edge = x[0] + min(offsets) - bar_width / 2
        right_edge = x[-1] + max(offsets) + bar_width / 2
        ax.set_xlim(left_edge - 0.1, right_edge + 0.1)
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
                  ncol=len(models_order), frameon=True, fancybox=True, 
                  shadow=False, fontsize=fontsize-2.7)  
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        png_path = f'figures/{output_prefix}_{dataset_name}_{metric_name}.png'
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_model_performance(df, dataset_name, output_prefix='model_performance', 
                           fontsize=12, psize=14, rotation=45, ha='right', width=0.2, group_gap=1.0):
    properties = ['Formation', 'Fermi', 'Band']
    prop_labels = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    colors = ['#E39C63', '#58A8D7', '#A9C37F']
    metrics = [('MAE', 'MAE (eV)'),   
               ('R2', r'R²')]

    x = np.arange(len(df['Model'])) * group_gap

    for metric_name, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(psize, 5))
        
        for i, (prop, label, color) in enumerate(zip(properties, prop_labels, colors)):
            values = df[f'{prop}_{metric_name}']
            bars = ax.bar(x + i*width, values, width, label=label, color=color,
                          edgecolor='black', linewidth=0.5, alpha=0.85)
            
        ax.set_ylabel(ylabel, fontsize=fontsize)
        
        max_values = []
        for prop in properties:
            max_values.extend(df[f'{prop}_{metric_name}'])
        y_max = max(max_values)
        ax.set_ylim(0, y_max * 1.3)  
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(df['Model'], rotation=rotation, ha=ha, fontsize=fontsize)

        ax.tick_params(axis='y', labelsize=fontsize)
        
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        
        ax.grid(False)
        
        ax.legend(loc='upper left', fontsize=fontsize-2, frameon=True, 
                  fancybox=True, edgecolor='black', shadow=False)

        plt.tight_layout()
        
        png_path = f'figures/{output_prefix}_{dataset_name}_{metric_name}.png'
        fig.tight_layout(pad=0.5)
        plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)

def supp_plt():
    data_all = {
        'Model': [
            'CNN_xrd', 'MLP_xrd', 'RNN_xrd', 'LSTM_xrd', 'GRU_xrd', 'Bi-RNN_xrd',
            'Bi-LSTM_xrd', 'Bi-GRU_xrd', 'Transformer_xrd',
            'RF_xrd', 'XGBoost_xrd', 'MLP_xafs', 'CNN_xafs', 'LSTM_xafs', 'AEMLP_xafs',
            'AECNN_xafs', 'mhMLP_xafs', 'mhCNN_xafs', 'Transformer_xafs', 'RF_xafs', 'XGBoost_xafs',
            'RF_xrd_xafs', 'InterLLC'
        ],
        'Formation_MAE': [0.2629, 0.2911, 0.4036, 0.4065, 0.4069, 0.4075, 0.4089, 0.4069, 0.3846, 0.5173, 0.5357, 
                          0.2856, 0.2673, 0.2605, 0.2484, 0.2511, 0.3264, 0.3422, 0.3311, 0.5303, 0.5492, 0.4724, 0.1884],
        'Formation_R2': [0.5245, 0.4284, 0.0216, 0.0057, 0.0107, 0.0015, 0.0045, 0.0111, 0.0619, 0.5656, 0.5285, 
                         0.4647, 0.465, 0.5391, 0.5639, 0.5405, 0.3457, 0.2845, 0.3355, 0.4464, 0.4368, 0.6322, 0.7443],
        'Fermi_MAE': [0.29, 0.3437, 0.3937, 0.3817, 0.3948, 0.3822, 0.3884, 0.3949, 0.3681, 0.5869, 0.6066, 
                      0.3326, 0.3065, 0.3011, 0.2891, 0.2851, 0.3444, 0.3571, 0.3652, 0.6522, 0.6518, 0.5479, 0.2439],
        'Fermi_R2': [0.4183, 0.181, 0.0027, 0.0418, 0.0011, 0.0408, 0.0259, 0.0022, 0.0816, 0.4021, 0.3449, 
                     0.3064, 0.4244, 0.3605, 0.4209, 0.4386, 0.2526, 0.214, 0.1477, 0.2038, 0.2118, 0.4758, 0.562],
        'Band_MAE': [0.3105, 0.3461, 0.4294, 0.4124, 0.4303, 0.4111, 0.4256, 0.428, 0.3937, 0.6881, 0.7093, 
                     0.3329, 0.3141, 0.3034, 0.2831, 0.3085, 0.3834, 0.4033, 0.403, 0.7326, 0.7115, 0.6286, 0.206],
        'Band_R2': [0.2679, 0.1101, 0.001, 0.0472, 0.0016, 0.0471, 0.0144, 0.0007, 0.0864, 0.2047, 0.1007, 
                    0.2727, 0.2068, 0.2379, 0.3471, 0.249, 0.2083, 0.158, 0.1399, 0.1557, 0.1661, 0.3388, 0.5715]
    }

    data_abx3 = {
        'Model': [
            'cnn_xrd', 'MLP_xrd', 'RNN_xrd', 'LSTM_xrd', 'GRU_xrd', 'Bi-RNN_xrd',
            'Bi-LSTM_xrd', 'Bi-GRU_xrd', 'Transformer_xrd', 'RF_xrd', 'XGBoost_xrd',
            'MLP_xafs', 'CNN_xafs', 'LSTM_xafs', 'AEMLP_xafs', 'AECNN_xafs',
            'mhMLP_xafs', 'mhCNN_xafs', 'Transformer_xafs', 'RF_xafs', 'XGBoost_xafs',
            'RF_xrd_xafs', 'InterLLC'
        ],
        'Formation_MAE': [0.2921, 0.3194, 0.3939, 0.3935, 0.3253, 0.3983, 0.3546, 0.3331, 0.3827, 0.5866, 0.5872, 
                          0.2496, 0.2305, 0.2751, 0.2721, 0.2445, 0.3243, 0.326, 0.3255, 0.55, 0.5677, 0.5283, 0.2037],
        'Formation_R2': [0.4622, 0.3498, 0.0384, 0.0121, 0.2996, 0.0056, 0.1981, 0.2903, 0.0589, 0.4365, 0.3995, 
                         0.5319, 0.5957, 0.5067, 0.5044, 0.5689, 0.3026, 0.2667, 0.3076, 0.4857, 0.445, 0.5229, 0.7133],
        'Fermi_MAE': [0.295, 0.3149, 0.3958, 0.3874, 0.3452, 0.3956, 0.3651, 0.351, 0.3882, 0.5805, 0.5834, 
                      0.3007, 0.3042, 0.3173, 0.3025, 0.309, 0.362, 0.3725, 0.3497, 0.6435, 0.615, 0.5742, 0.236],
        'Fermi_R2': [0.4344, 0.3385, 0.0571, 0.1165, 0.2404, 0.0579, 0.2078, 0.1715, 0.1053, 0.4448, 0.437, 
                     0.383, 0.3572, 0.3406, 0.3865, 0.3742, 0.2043, 0.1792, 0.2646, 0.3355, 0.3814, 0.4634, 0.5816],
        'Band_MAE': [0.3154, 0.3567, 0.4194, 0.3764, 0.3616, 0.4179, 0.3774, 0.3673, 0.3747, 0.6681, 0.6741, 
                     0.2834, 0.2739, 0.2897, 0.2941, 0.2976, 0.3959, 0.3953, 0.3879, 0.6392, 0.6129, 0.5619, 0.2229],
        'Band_R2': [0.2666, 0.1368, 0.0243, 0.1188, 0.1542, 0.0579, 0.1525, 0.1583, 0.1135, 0.2116, 0.1385, 
                    0.353, 0.3381, 0.3403, 0.3329, 0.3164, 0.1194, 0.1328, 0.1332, 0.2881, 0.3006, 0.4156, 0.5163]
    }

    df_all = pd.DataFrame(data_all)
    df_abx3 = pd.DataFrame(data_abx3)

    plot_model_performance(df_all, 'All', output_prefix='supp_performance', fontsize=10, psize=8)
    plot_model_performance(df_abx3, 'ABX3', output_prefix='supp_performance', fontsize=10, psize=8)

def ab_fusion_plot():
    models = ['w/ Concat', 'w/o XRD', 'w/o XAFS', 
              'w/o PFA', 'w/o IAE', 
              'w/o BCF', 'InterLLC']
    attributes = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    
    colors = ['#A9C37F', '#F4D8A2', '#58A8D7', '#BAD2E1', '#F7E0CF', '#D1E4CF', '#E39C63']

    data_dict = {
        'w/ Concat':     [0.2796, 0.4763, 0.2935, 0.3787, 0.2735, 0.3811],
        'w/o XRD':       [0.2556, 0.5058, 0.2753, 0.4230, 0.2637, 0.4211],
        'w/o XAFS':      [0.2629, 0.5609, 0.2645, 0.4754, 0.2976, 0.2686],
        'w/o PFA':       [0.2500, 0.5721, 0.2703, 0.4757, 0.2656, 0.3175],
        'w/o IAE':       [0.2053, 0.7094, 0.2481, 0.5365, 0.2333, 0.4504],
        'w/o BCF':       [0.2039, 0.7072, 0.2378, 0.5670, 0.2390, 0.4477],
        'InterLLC':         [0.2021, 0.7133, 0.2360, 0.5816, 0.2212, 0.5163]
    }

    mae_wide = {attr: [] for attr in attributes}
    r2_wide  = {attr: [] for attr in attributes}
    for model in models:
        vals = data_dict[model]
        mae_wide['Formation Energy'].append(vals[0])
        r2_wide['Formation Energy'].append(vals[1])
        mae_wide['Fermi Energy'].append(vals[2])
        r2_wide['Fermi Energy'].append(vals[3])
        mae_wide['Bandgap'].append(vals[4])
        r2_wide['Bandgap'].append(vals[5])

    df_mae = pd.DataFrame(mae_wide, index=models)
    df_r2  = pd.DataFrame(r2_wide, index=models)

    bar_width = 0.105
    n_models = len(models)
    n_attrs = len(attributes)
    
    total_width = (n_models - 1) * bar_width
    offsets = np.linspace(-total_width / 2, total_width / 2, n_models)
    
    x = np.arange(n_attrs)
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    
    for i, model in enumerate(models):
        values = [df_mae.loc[model, attr] for attr in attributes]
        bar_positions = x + offsets[i]
        bars = ax1.bar(bar_positions, values, bar_width, 
                       label=model, color=colors[i],
                       edgecolor='black', linewidth=0.5, alpha=0.85)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                offset = 0.008
                ax1.text(bar.get_x() + bar.get_width()/2., height + offset,
                        f'{height:.3f}',
                        ha='center', va='bottom',
                        rotation=90,
                        fontsize=10,
                        color='black')

    ax1.set_ylabel('MAE (eV)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(attributes, fontsize=12, rotation=0, ha='center')
    
    ax1.set_ylim(0, 0.41)
    ax1.set_yticks(np.arange(0, 0.45, 0.05))
    ax1.yaxis.grid(False)
    ax1.set_axisbelow(True)
    
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax1.tick_params(top=False, right=False)
    
    left_edge = x[0] + min(offsets) - bar_width / 2
    right_edge = x[-1] + max(offsets) + bar_width / 2
    ax1.set_xlim(left_edge - 0.1, right_edge + 0.1)
    
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
              ncol=4, frameon=True, fancybox=True, 
              shadow=False, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/ablation_fusion_MAE.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    
    for i, model in enumerate(models):
        values = [df_r2.loc[model, attr] for attr in attributes]
        bar_positions = x + offsets[i]
        bars = ax2.bar(bar_positions, values, bar_width, 
                       label=model, color=colors[i],
                       edgecolor='black', linewidth=0.5, alpha=0.85)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                offset = 0.008
                ax2.text(bar.get_x() + bar.get_width()/2., height + offset,
                        f'{height:.3f}',
                        ha='center', va='bottom',
                        rotation=90,
                        fontsize=10,
                        color='black')

    ax2.set_ylabel('R²', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(attributes, fontsize=12, rotation=0, ha='center')
    
    ax2.set_ylim(0, 0.97)
    ax2.set_yticks(np.arange(0, 1.0, 0.1))
    ax2.yaxis.grid(False)
    ax2.set_axisbelow(True)
    
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
    ax2.tick_params(top=False, right=False)
    
    ax2.set_xlim(left_edge - 0.1, right_edge + 0.1)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
              ncol=4, frameon=True, fancybox=True, 
              shadow=False, fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig('figures/ablation_fusion_R2.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)


def ab_multi_task_learning():

    custom_colors = ['#3F719D', '#58A8D7', '#9FC0D6', '#C6DEED', '#DBF1FA']
    custom_cmap = LinearSegmentedColormap.from_list('custom_blues', custom_colors, N=256)
    custom_cmap_rev = custom_cmap.reversed() 

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    form_mae_raw = {
        'InterLLC': [0.3203, 0.2498, 0.2299, 0.2132, 0.2037],
        'Single-task only': [0.3199, 0.2773, 0.2440, 0.2242, 0.2055],
        'w/o formation': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'w/o fermi': [0.3344, 0.2632, 0.2313, 0.2277, 0.2129],
        'w/o bandgap': [0.4074, 0.2510, 0.2233, 0.2166, 0.2126]
    }
    form_r2_raw = {
        'InterLLC': [0.2774, 0.5375, 0.6157, 0.6783, 0.7133],
        'Single-task only': [0.2643, 0.4592, 0.5645, 0.6249, 0.6860],
        'w/o formation': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'w/o fermi': [0.2351, 0.5012, 0.6019, 0.6471, 0.6527],
        'w/o bandgap': [0.0859, 0.5092, 0.6037, 0.6577, 0.6947]
    }

    fermi_mae_raw = {
        'InterLLC': [0.3280, 0.2936, 0.2830, 0.2575, 0.2360],
        'Single-task only': [0.4555, 0.3233, 0.2787, 0.2516, 0.2435],
        'w/o formation': [0.3798, 0.2763, 0.2674, 0.2682, 0.2394],
        'w/o fermi': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'w/o bandgap': [0.4460, 0.3059, 0.2582, 0.2519, 0.2554]
    }
    fermi_r2_raw = {
        'InterLLC': [0.2188, 0.3749, 0.5019, 0.5275, 0.5816],
        'Single-task only': [0.0517, 0.2464, 0.4355, 0.5017, 0.5321],
        'w/o formation': [0.0268, 0.2458, 0.4575, 0.4636, 0.5556],
        'w/o fermi': [np.nan, np.nan, np.nan, np.nan, np.nan],
        'w/o bandgap': [0.0949, 0.3390, 0.4802, 0.5082, 0.5123]
    }

    gap_mae_raw = {
        'InterLLC': [0.3086, 0.2570, 0.2429, 0.2399, 0.2229],
        'Single-task only': [0.3740, 0.2698, 0.2700, 0.2611, 0.2432],
        'w/o formation': [0.3614, 0.2728, 0.2693, 0.2655, 0.2337],
        'w/o fermi': [0.3377, 0.2586, 0.2654, 0.2603, 0.2464],
        'w/o bandgap': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }
    gap_r2_raw = {
        'InterLLC': [0.1310, 0.3431, 0.4196, 0.4409, 0.5163],
        'Single-task only': [0.0476, 0.3004, 0.3037, 0.3557, 0.4605],
        'w/o formation': [0.0402, 0.3191, 0.3239, 0.3403, 0.4619],
        'w/o fermi': [0.0029, 0.3060, 0.3440, 0.3959, 0.4212],
        'w/o bandgap': [np.nan, np.nan, np.nan, np.nan, np.nan]
    }

    def filter_valid_rows(data_dict):
        labels = []
        matrix = []
        for label, values in data_dict.items():
            if not np.all(np.isnan(values)):
                labels.append(label)
                matrix.append(values)
        return np.array(matrix), labels

    form_mae_mat, form_mae_labels = filter_valid_rows(form_mae_raw)
    form_r2_mat, form_r2_labels = filter_valid_rows(form_r2_raw)
    fermi_mae_mat, fermi_mae_labels = filter_valid_rows(fermi_mae_raw)
    fermi_r2_mat, fermi_r2_labels = filter_valid_rows(fermi_r2_raw)
    gap_mae_mat, gap_mae_labels = filter_valid_rows(gap_mae_raw)
    gap_r2_mat, gap_r2_labels = filter_valid_rows(gap_r2_raw)

    def draw_heatmap(ax, data, vmin, vmax, cmap, title, xlabel, ylabel,
                     xtick_labels, ytick_labels, fmt='.3f'):
        ax.grid(False)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto',
                       interpolation='nearest')
        ax.set_xticks(np.arange(len(xtick_labels)))
        ax.set_xticklabels(xtick_labels, fontsize=7)
        ax.set_yticks(np.arange(len(ytick_labels)))
        ax.set_yticklabels(ytick_labels, fontsize=7)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=7, pad=2)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = 'white' if (vmin + vmax)/2 < data[i, j] else 'black'
                ax.text(j, i, f'{data[i, j]:{fmt}}', ha='center', va='center',
                        fontsize=6, color=color)
        return im

    fig1, axes1 = plt.subplots(1, 3, figsize=(7.5, 2.5), constrained_layout=True)
    vmin_mae, vmax_mae = 0.18, 0.52
    titles_mae = ['Formation energy', 'Fermi energy', 'Bandgap']
    data_mae = [form_mae_mat, fermi_mae_mat, gap_mae_mat]
    labels_mae = [form_mae_labels, fermi_mae_labels, gap_mae_labels]

    for idx, ax in enumerate(axes1):
        im = draw_heatmap(ax, data_mae[idx], vmin_mae, vmax_mae, custom_cmap,
                          titles_mae[idx], 'Training data fraction', '',
                          ratios, labels_mae[idx], fmt='.3f')
    cbar1 = fig1.colorbar(im, ax=axes1, location='bottom', fraction=0.05, pad=0.12,
                          aspect=40, shrink=0.6)
    cbar1.set_label('MAE (eV)', fontsize=8)
    cbar1.ax.tick_params(labelsize=6)
    cbar1.set_ticks([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
    cbar1.set_ticklabels(['0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50'])

    cbar1.ax.set_xlim(vmin_mae, vmax_mae)
    plt.savefig('figures/ab_mtl_MAE_heatmap.png')
    plt.close(fig1)

    fig2, axes2 = plt.subplots(1, 3, figsize=(7.5, 2.5), constrained_layout=True)
    vmin_r2, vmax_r2 = -0.02, 0.72
    titles_r2 = ['Formation energy', 'Fermi energy', 'Bandgap']
    data_r2 = [form_r2_mat, fermi_r2_mat, gap_r2_mat]
    labels_r2 = [form_r2_labels, fermi_r2_labels, gap_r2_labels]

    for idx, ax in enumerate(axes2):
        im = draw_heatmap(ax, data_r2[idx], vmin_r2, vmax_r2, custom_cmap_rev,
                          titles_r2[idx], 'Training data fraction', '',
                          ratios, labels_r2[idx], fmt='.3f')
    cbar2 = fig2.colorbar(im, ax=axes2, location='bottom', fraction=0.05, pad=0.12,
                          aspect=40, shrink=0.6)
    cbar2.set_label('R²', fontsize=8)
    cbar2.ax.tick_params(labelsize=6)
    cbar2.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    cbar2.set_ticklabels(['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7'])
    cbar2.ax.set_xlim(vmin_r2, vmax_r2)
    plt.savefig('figures/ab_mtl_R2_heatmap.png')
    plt.close(fig2)

from matplotlib.patches import Patch
def plot_metrics_for_three_properties(data, colors, save_mae_path, save_r2_path):
    all_methods = []
    for prop_info in data.values():
        for method in prop_info['methods']:
            if method not in all_methods:
                all_methods.append(method)
    n_methods = len(all_methods)

    if len(colors) < n_methods:
        colors = (colors * (n_methods // len(colors) + 1))[:n_methods]
    method_color = {method: colors[i] for i, method in enumerate(all_methods)}
    
    properties = list(data.keys())
    n_props = len(properties)            
    n_methods_per_prop = 4               
    bar_width = 0.12
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, n_methods_per_prop)
    
    def draw_single_metric(metric_name, ylabel, filename, legend_loc):
        fig, ax = plt.subplots(figsize=(7, 5))
        centers = np.array([0, 0.8, 1.6])
        
        for i, prop in enumerate(properties):
            methods = data[prop]['methods']
            values = data[prop][metric_name]
            x_center = centers[i]
            for j, (method, val) in enumerate(zip(methods, values)):
                color = method_color[method]
                bar = ax.bar(x_center + offsets[j], val, width=bar_width,
                            color=color, edgecolor='black', linewidth=0.5,
                            label=method if i == 0 else "")
                y_offset = 0.002 if metric_name == 'MAE' else 0.008
                ax.text(bar[0].get_x() + bar[0].get_width()/2, val + y_offset,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=8)
        
        if metric_name == 'MAE':
            ax.set_ylim(0, 0.5)
        elif metric_name == 'R2':
            ax.set_ylim(0, 1)
        
        ax.set_xticks(centers)
        ax.set_xticklabels(properties, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(False)
        ax.set_axisbelow(True)
        
        x_min = centers[0] - 0.4
        x_max = centers[-1] + 0.4
        ax.set_xlim(x_min, x_max)
        
        legend_elements = [Patch(facecolor=method_color[m], edgecolor='black', label=m) for m in all_methods]
        ax.legend(handles=legend_elements, loc=legend_loc, frameon=True, edgecolor='black', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
        plt.close()
    
    draw_single_metric('MAE', 'MAE (eV)', save_mae_path, legend_loc='upper right')
    draw_single_metric('R2', 'R²', save_r2_path, legend_loc='lower right')

def ab_task():
    data = {
        'Formation Energy': {
            'methods': ['InterLLC', 'Formation energy only', 'w/o Fermi energy', 'w/o Bandgap'],
            'MAE': [0.2037, 0.2055, 0.2129, 0.2126],
            'R2':  [0.7133, 0.6860, 0.6527, 0.6947]
        },
        'Fermi Energy': {
            'methods': ['InterLLC', 'Fermi energy only', 'w/o Formation energy', 'w/o Bandgap'],
            'MAE': [0.2360, 0.2435, 0.2394, 0.2554],
            'R2':  [0.5816, 0.5321, 0.5556, 0.5123]
        },
        'Bandgap': {
            'methods': ['InterLLC', 'Bandgap only', 'w/o Formation energy', 'w/o Fermi energy'],
            'MAE': [0.2229, 0.2432, 0.2337, 0.2464],
            'R2':  [0.5163, 0.4605, 0.4619, 0.4212]
        }
    }
    
    colors = ['#E39C63', '#F7E0CF', '#58A8D7', '#96C2D4', '#BAD2E1', '#A9C37F', '#D1E4CF']
    
    plot_metrics_for_three_properties(data, colors,
                                       save_mae_path='figures/ab_task_mae.png',
                                       save_r2_path='figures/ab_task_r2.png')
    

from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def ab_mtl_heatmap():
    combinations = ['InterLLC', 'Single-task', 'w/o formation', 'w/o fermi', 'w/o bandgap']
    targets_full = ['Formation energy', 'Fermi energy', 'Bandgap']

    new_order = ['Formation energy', 'Fermi energy', 'Bandgap']
    new_targets_display = ['Formation Energy', 'Fermi Energy', 'Bandgap']

    mae_data = {
        ('InterLLC', 'Formation energy'): 0.2031,
        ('Single-task', 'Formation energy'): 0.2055,
        ('w/o formation', 'Formation energy'): np.nan,
        ('w/o fermi', 'Formation energy'): 0.2139,
        ('w/o bandgap', 'Formation energy'): 0.2124,
        
        ('InterLLC', 'Fermi energy'): 0.2360,
        ('Single-task', 'Fermi energy'): 0.2435,
        ('w/o formation', 'Fermi energy'): 0.2394,
        ('w/o fermi', 'Fermi energy'): np.nan,
        ('w/o bandgap', 'Fermi energy'): 0.2554,
        
        ('InterLLC', 'Bandgap'): 0.2229,
        ('Single-task', 'Bandgap'): 0.2432,
        ('w/o formation', 'Bandgap'): 0.2337,
        ('w/o fermi', 'Bandgap'): 0.2464,
        ('w/o bandgap', 'Bandgap'): np.nan,
    }

    r2_data = {
        ('InterLLC', 'Formation energy'): 0.7133,
        ('Single-task', 'Formation energy'): 0.6860,
        ('w/o formation', 'Formation energy'): np.nan,
        ('w/o fermi', 'Formation energy'): 0.6527,
        ('w/o bandgap', 'Formation energy'): 0.6947,
        
        ('InterLLC', 'Fermi energy'): 0.5816,
        ('Single-task', 'Fermi energy'): 0.5321,
        ('w/o formation', 'Fermi energy'): 0.5556,
        ('w/o fermi', 'Fermi energy'): np.nan,
        ('w/o bandgap', 'Fermi energy'): 0.5123,
        
        ('InterLLC', 'Bandgap'): 0.5163,
        ('Single-task', 'Bandgap'): 0.4605,
        ('w/o formation', 'Bandgap'): 0.4619,
        ('w/o fermi', 'Bandgap'): 0.4212,
        ('w/o bandgap', 'Bandgap'): np.nan,
    }

    mae_matrix_orig = np.full((len(combinations), len(targets_full)), np.nan)
    r2_matrix_orig = np.full((len(combinations), len(targets_full)), np.nan)
    for i, comb in enumerate(combinations):
        for j, tgt in enumerate(targets_full):
            mae_matrix_orig[i, j] = mae_data.get((comb, tgt), np.nan)
            r2_matrix_orig[i, j] = r2_data.get((comb, tgt), np.nan)

    mae_matrix = mae_matrix_orig.T
    r2_matrix = r2_matrix_orig.T

    order_indices = [targets_full.index(t) for t in new_order]
    mae_matrix = mae_matrix[order_indices, :]
    r2_matrix = r2_matrix[order_indices, :]

    mae_annot = np.empty_like(mae_matrix, dtype=object)
    r2_annot = np.empty_like(r2_matrix, dtype=object)
    for i in range(mae_matrix.shape[0]):
        for j in range(mae_matrix.shape[1]):
            if not np.isnan(mae_matrix[i, j]):
                mae_annot[i, j] = f"{mae_matrix[i, j]:.3f}"
                r2_annot[i, j] = f"{r2_matrix[i, j]:.3f}"
            else:
                mae_annot[i, j] = '-'
                r2_annot[i, j] = '-'

    colors = ['#E39C63', '#F7E0CF', '#58A8D7', '#BAD2E1']
    deep_orange = colors[0]  
    deep_blue   = colors[2] 

    cmap_mae = LinearSegmentedColormap.from_list('orange_cmap', [deep_orange, 'white'])
    cmap_r2 = LinearSegmentedColormap.from_list('blue_cmap', ['white', deep_blue])

    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'sans-serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(mae_matrix, cmap=cmap_mae, vmin=0.195, vmax=0.265, aspect='auto')
    for i in range(mae_matrix.shape[0]):
        for j in range(mae_matrix.shape[1]):
            ax1.text(j, i, mae_annot[i, j], ha='center', va='center', color='black', fontsize=10)
    ax1.set_xticks(np.arange(len(combinations)))
    ax1.set_xticklabels(combinations, rotation=45, ha='right')
    ax1.set_yticks(np.arange(len(new_targets_display)))
    ax1.set_yticklabels(new_targets_display)
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.set_label('MAE (eV)')
    cbar1.set_ticks([0.20, 0.22, 0.24, 0.26])
    cbar1.ax.invert_yaxis()   

    im2 = ax2.imshow(r2_matrix, cmap=cmap_r2, vmin=0.405, vmax=0.755, aspect='auto')
    for i in range(r2_matrix.shape[0]):
        for j in range(r2_matrix.shape[1]):
            ax2.text(j, i, r2_annot[i, j], ha='center', va='center', color='black', fontsize=10)
    ax2.set_xticks(np.arange(len(combinations)))
    ax2.set_xticklabels(combinations, rotation=45, ha='right')
    ax2.set_yticks(np.arange(len(new_targets_display)))
    ax2.set_yticklabels(new_targets_display)
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.set_label('R²')
    cbar2.set_ticks([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75])

    plt.tight_layout()
    plt.savefig('figures/ab_mlt_heatmaps_mae_r2.png', dpi=300, bbox_inches='tight')
    plt.close()

def ab_mtl_heatmap_abs_diff():
    combinations = ['InterLLC', 'Single-task', 'w/o formation', 'w/o fermi', 'w/o bandgap']
    targets_full = ['Formation energy', 'Fermi energy', 'Bandgap']
    new_order = ['Formation energy', 'Fermi energy', 'Bandgap']
    new_targets_display = ['Formation Energy', 'Fermi Energy', 'Bandgap']

    mae_data = {
        ('InterLLC', 'Formation energy'): 0.2031,
        ('Single-task', 'Formation energy'): 0.2055,
        ('w/o formation', 'Formation energy'): np.nan,
        ('w/o fermi', 'Formation energy'): 0.2139,
        ('w/o bandgap', 'Formation energy'): 0.2124,
        
        ('InterLLC', 'Fermi energy'): 0.2360,
        ('Single-task', 'Fermi energy'): 0.2435,
        ('w/o formation', 'Fermi energy'): 0.2394,
        ('w/o fermi', 'Fermi energy'): np.nan,
        ('w/o bandgap', 'Fermi energy'): 0.2554,
        
        ('InterLLC', 'Bandgap'): 0.2229,
        ('Single-task', 'Bandgap'): 0.2432,
        ('w/o formation', 'Bandgap'): 0.2337,
        ('w/o fermi', 'Bandgap'): 0.2464,
        ('w/o bandgap', 'Bandgap'): np.nan,
    }

    r2_data = {
        ('InterLLC', 'Formation energy'): 0.7133,
        ('Single-task', 'Formation energy'): 0.6860,
        ('w/o formation', 'Formation energy'): np.nan,
        ('w/o fermi', 'Formation energy'): 0.6527,
        ('w/o bandgap', 'Formation energy'): 0.6947,
        
        ('InterLLC', 'Fermi energy'): 0.5816,
        ('Single-task', 'Fermi energy'): 0.5321,
        ('w/o formation', 'Fermi energy'): 0.5556,
        ('w/o fermi', 'Fermi energy'): np.nan,
        ('w/o bandgap', 'Fermi energy'): 0.5123,
        
        ('InterLLC', 'Bandgap'): 0.5163,
        ('Single-task', 'Bandgap'): 0.4605,
        ('w/o formation', 'Bandgap'): 0.4619,
        ('w/o fermi', 'Bandgap'): 0.4212,
        ('w/o bandgap', 'Bandgap'): np.nan,
    }

    hier_mae = {tgt: mae_data[('InterLLC', tgt)] for tgt in targets_full}
    hier_r2  = {tgt: r2_data[('InterLLC', tgt)] for tgt in targets_full}
    other_methods = [m for m in combinations if m != 'InterLLC']
    
    n_targets = len(new_order)
    n_methods = len(other_methods)
    
    mae_abs_diff = np.full((n_targets, n_methods), np.nan)
    r2_abs_diff  = np.full((n_targets, n_methods), np.nan)
    annot_mae = np.empty_like(mae_abs_diff, dtype=object)
    annot_r2  = np.empty_like(r2_abs_diff, dtype=object)

    for i, tgt in enumerate(new_order):
        base_mae = hier_mae[tgt]
        base_r2  = hier_r2[tgt]
        for j, method in enumerate(other_methods):
            mae_val = mae_data.get((method, tgt), np.nan)
            r2_val  = r2_data.get((method, tgt), np.nan)
            if not np.isnan(mae_val):
                diff_mae = abs(base_mae - mae_val)
                mae_abs_diff[i, j] = diff_mae
                annot_mae[i, j] = f"{diff_mae:.3f}"
            else:
                annot_mae[i, j] = '-'
            if not np.isnan(r2_val):
                diff_r2 = abs(base_r2 - r2_val)
                r2_abs_diff[i, j] = diff_r2
                annot_r2[i, j] = f"{diff_r2:.3f}"
            else:
                annot_r2[i, j] = '-'

    colors = ['#E39C63', '#F7E0CF', '#58A8D7', '#BAD2E1']
    deep_orange = colors[0]
    deep_blue   = colors[2]

    cmap_mae_abs = LinearSegmentedColormap.from_list('mae_abs', ['white', deep_orange])
    cmap_r2_abs  = LinearSegmentedColormap.from_list('r2_abs', ['white', deep_blue])

    vmin_mae, vmax_mae = -0.001, 0.021
    vmin_r2,  vmax_r2  = -0.005, 0.105

    ticks_mae = [0, 0.005, 0.01, 0.015, 0.02]
    ticks_r2  = [0, 0.02, 0.04, 0.06, 0.08, 0.1]

    plt.rcParams['font.size'] = 12
    plt.rcParams['font.family'] = 'sans-serif'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

    im1 = ax1.imshow(mae_abs_diff, cmap=cmap_mae_abs, vmin=vmin_mae, vmax=vmax_mae, aspect='auto')
    for i in range(mae_abs_diff.shape[0]):
        for j in range(mae_abs_diff.shape[1]):
            ax1.text(j, i, annot_mae[i, j], ha='center', va='center', color='black', fontsize=10)
    ax1.set_xticks(np.arange(n_methods))
    ax1.set_xticklabels(other_methods, rotation=45, ha='right', fontsize=10)  
    ax1.set_yticks(np.arange(n_targets))
    ax1.set_yticklabels(new_targets_display, fontsize=10)                     
    ax1.set_xlabel('')
    ax1.set_ylabel('')

    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.05, pad=0.05, ticks=ticks_mae)
    cbar1.set_label('ΔMAE (eV)', fontsize=10)
    cbar1.ax.tick_params(labelsize=8)
    cbar1.ax.tick_params(pad=2)

    im2 = ax2.imshow(r2_abs_diff, cmap=cmap_r2_abs, vmin=vmin_r2, vmax=vmax_r2, aspect='auto')
    for i in range(r2_abs_diff.shape[0]):
        for j in range(r2_abs_diff.shape[1]):
            ax2.text(j, i, annot_r2[i, j], ha='center', va='center', color='black', fontsize=10)
    ax2.set_xticks(np.arange(n_methods))
    ax2.set_xticklabels(other_methods, rotation=45, ha='right', fontsize=10)  
    ax2.set_yticks(np.arange(n_targets))
    ax2.set_yticklabels(new_targets_display, fontsize=10)                    
    ax2.set_xlabel('')
    ax2.set_ylabel('')

    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.05, pad=0.05, ticks=ticks_r2)
    cbar2.set_label('ΔR²', fontsize=10)
    cbar2.ax.tick_params(labelsize=8)
    cbar2.ax.tick_params(pad=2)

    plt.tight_layout()
    plt.savefig('figures/ab_mlt_heatmaps_abs_diff.png', bbox_inches='tight')
    plt.close()

def ab_mtl_curve():
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    form_data = {
        'InterLLC':  [0.2774, 0.5375, 0.6157, 0.6783, 0.7133],
        'Single-task only': [0.2643, 0.4592, 0.5645, 0.6249, 0.6860],
        'w/o fermi': [0.2351, 0.5012, 0.6019, 0.6471, 0.6527],
        'w/o bandgap': [0.0859, 0.5092, 0.6037, 0.6577, 0.6947]
    }
    fermi_data = {
        'InterLLC':  [0.2188, 0.3749, 0.5019, 0.5275, 0.5816],
        'Single-task only': [0.0517, 0.2464, 0.4355, 0.5017, 0.5321],
        'w/o formation': [0.0268, 0.2458, 0.4575, 0.4636, 0.5556],
        'w/o bandgap': [0.0949, 0.3390, 0.4802, 0.5082, 0.5123]
    }
    gap_data = {
        'InterLLC':  [0.1310, 0.3431, 0.4196, 0.4409, 0.5163],
        'Single-task only': [0.0476, 0.3004, 0.3037, 0.3557, 0.4605],
        'w/o formation': [0.0402, 0.3191, 0.3239, 0.3403, 0.4619],
        'w/o fermi': [0.0029, 0.3060, 0.3440, 0.3959, 0.4212]
    }

    solid_colors = {
        'InterLLC': '#DD542F',
        'Single-task only': '#1C3885'
    }
    dash_colors = ['#999999', '#93DCB0']  
    markers = ['o', 's', '^', 'D']

    titles = ['Formation energy', 'Fermi energy', 'Bandgap']
    data_list = [form_data, fermi_data, gap_data]
    orders = [
        ['InterLLC', 'Single-task only', 'w/o fermi', 'w/o bandgap'],
        ['InterLLC', 'Single-task only', 'w/o formation', 'w/o bandgap'],
        ['InterLLC', 'Single-task only', 'w/o formation', 'w/o fermi']
    ]

    for title, data, order in zip(titles, data_list, orders):
        fig, ax = plt.subplots(figsize=(2.8, 2.5))
        other_methods = order[2:]   

        for i, method in enumerate(order):
            if method in solid_colors:        
                color = solid_colors[method]
                linestyle = '-'
            else:                             
                idx = other_methods.index(method)
                color = dash_colors[idx]
                linestyle = '--'

            ax.plot(ratios, data[method],
                    marker=markers[i], markersize=3,
                    color=color, linewidth=1.5, linestyle=linestyle,
                    label=method)

        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel('Training data fraction', fontsize=8)
        ax.set_ylabel('R²', fontsize=8)

        ax.tick_params(axis='both', labelsize=6)

        ax.set_xlim(0.15, 1.02)
        ax.set_ylim(-0.05, 0.85)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        ax.tick_params(labeltop=False, labelright=False)
        ax.legend(loc='lower right', frameon=True, fontsize=6,
                  fancybox=False, edgecolor='black', shadow=False)

        plt.tight_layout()
        filename = f'figures/ab_R2_{title.replace(" ", "_")}_curve.png'
        plt.savefig(filename, dpi=300)
        plt.close(fig)

def ab_mtl_key_curve_statistic(): 
    data = {
        'formation': {
            0.2: {'InterLLC': {'MAE': [0.3203, 0.3470, 0.3578, 0.3061, 0.3636, 0.3317, 0.3552, 0.3321, 0.3549],
                                    'R2': [0.2774, 0.2981, 0.2666, 0.3329, 0.2776, 0.2269, 0.2070, 0.2814, 0.2172]},
                'Single': {'MAE': [0.3799, 0.3913, 0.3888, 0.3957, 0.3842, 0.4029, 0.3872, 0.3999, 0.3966],
                            'R2': [0.2643, 0.1768, 0.1766, 0.1533, 0.0983, 0.2789, 0.0588, 0.1484, 0.0138]}},
            0.4: {'InterLLC': {'MAE': [0.2498, 0.2299, 0.2342, 0.2368, 0.2466, 0.2450, 0.2244, 0.2469, 0.2338],
                                    'R2': [0.5375, 0.4630, 0.4276, 0.5912, 0.5591, 0.5581, 0.4833, 0.4367, 0.5389]},
                'Single': {'MAE': [0.2773, 0.2814, 0.2720, 0.2951, 0.2676, 0.2847, 0.2703, 0.2673, 0.2867],
                            'R2': [0.4592, 0.6158, 0.5418, 0.4031, 0.4786, 0.5757, 0.4414, 0.5330, 0.5261]}},
            0.6: {'InterLLC': {'MAE': [0.2299, 0.2296, 0.2205, 0.2330, 0.2385, 0.2168, 0.2287, 0.2388, 0.2404, 0.2327],
                                    'R2': [0.6157, 0.6054, 0.6414, 0.6080, 0.6137, 0.5998, 0.6314, 0.5944, 0.6278, 0.5990]},
                'Single': {'MAE': [0.2440, 0.2458, 0.2219, 0.2370, 0.2222, 0.2348, 0.2320, 0.2400, 0.2431, 0.2514],
                            'R2': [0.5645, 0.5904, 0.6158, 0.6483, 0.6196, 0.6073, 0.5905, 0.5603, 0.5603, 0.6158]}},
            0.8: {'InterLLC': {'MAE': [0.2132, 0.2323, 0.2294, 0.2217, 0.2442, 0.2262, 0.2199, 0.2267, 0.2277, 0.2343],
                                    'R2': [0.6783, 0.6231, 0.6573, 0.6467, 0.6338, 0.6257, 0.6596, 0.6294, 0.6259, 0.6466]},
                'Single': {'MAE': [0.2242, 0.2378, 0.2405, 0.2405, 0.2371, 0.2341, 0.2076, 0.2166, 0.2360, 0.2258],
                            'R2': [0.6249, 0.5845, 0.5847, 0.5847, 0.6122, 0.5705, 0.6730, 0.6303, 0.5839, 0.5904]}},
            1.0: {'InterLLC': {'MAE': [0.2037, 0.2197, 0.2143, 0.2037, 0.2095, 0.2034, 0.2283, 0.2222, 0.2154, 0.2170],
                                    'R2': [0.7133, 0.6531, 0.6759, 0.6868, 0.6947, 0.6968, 0.6355, 0.6474, 0.6498, 0.6680]},
                'Single': {'MAE': [0.2355, 0.2449, 0.2389, 0.2503, 0.2432, 0.2377, 0.2438, 0.2423, 0.2328, 0.2531],
                            'R2': [0.6360, 0.6470, 0.6464, 0.6287, 0.6370, 0.6161, 0.6385, 0.5946, 0.6051, 0.6009]}}
        },
        'fermi': {
            0.2: {'InterLLC': {'MAE': [0.3280, 0.4114, 0.3821, 0.3400, 0.4085, 0.3230, 0.3915, 0.4184, 0.3476],
                                    'R2': [0.2188, 0.2066, 0.0654, 0.1908, 0.1659, 0.2613, 0.0905, 0.2286, 0.1476]},
                'Single': {'MAE': [0.4555, 0.4412, 0.4062, 0.3799, 0.4175, 0.3992, 0.3858, 0.4338, 0.4172],
                            'R2': [0.0517, 0.0705, 0.0671, 0.0250, 0.0385, 0.1032, 0.0538, 0.0958, 0.1072]}},
            0.4: {'InterLLC': {'MAE': [0.2936, 0.3039, 0.3184, 0.2637, 0.2794, 0.2944, 0.2898, 0.3267, 0.2830],
                                    'R2': [0.3749, 0.3207, 0.2653, 0.4723, 0.4106, 0.3422, 0.3817, 0.2372, 0.3818]},
                'Single': {'MAE': [0.3233, 0.3456, 0.3281, 0.3211, 0.2943, 0.2902, 0.3131, 0.3073, 0.2987],
                            'R2': [0.2464, 0.1581, 0.2201, 0.2504, 0.3674, 0.3540, 0.3030, 0.3012, 0.3375]}},
            0.6: {'InterLLC': {'MAE': [0.2330, 0.2519, 0.2490, 0.2599, 0.2646, 0.2439, 0.2656, 0.2643, 0.2669, 0.2606],
                                    'R2': [0.5019, 0.5103, 0.5131, 0.4910, 0.4881, 0.4750, 0.4694, 0.4647, 0.4724, 0.4725]},
                'Single': {'MAE': [0.2787, 0.2677, 0.2553, 0.2677, 0.2779, 0.2891, 0.2744, 0.2854, 0.2724, 0.2666],
                            'R2': [0.4355, 0.4408, 0.4398, 0.4408, 0.4210, 0.3992, 0.4334, 0.3966, 0.4378, 0.4236]}},
            0.8: {'InterLLC': {'MAE': [0.2575, 0.2630, 0.2592, 0.2592, 0.2630, 0.2468, 0.2429, 0.2586, 0.2467, 0.2550],
                                    'R2': [0.5275, 0.4987, 0.4925, 0.4813, 0.4649, 0.4359, 0.4758, 0.5131, 0.4697, 0.5189]},
                'Single': {'MAE': [0.2516, 0.2557, 0.2534, 0.2651, 0.2657, 0.2717, 0.2614, 0.2719, 0.2751, 0.2669],
                            'R2': [0.5017, 0.4978, 0.4976, 0.4665, 0.4585, 0.4396, 0.4907, 0.4199, 0.4232, 0.4728]}},
            1.0: {'InterLLC': {'MAE': [0.2360, 0.2402, 0.2462, 0.2431, 0.2477, 0.2448, 0.2440, 0.2383, 0.2430, 0.2457],
                                    'R2': [0.5816, 0.5242, 0.5503, 0.5585, 0.5373, 0.5383, 0.5553, 0.5239, 0.5069, 0.5559]},
                'Single': {'MAE': [0.2435, 0.2493, 0.2425, 0.2615, 0.2741, 0.2405, 0.2514, 0.2492, 0.2776, 0.2451],
                            'R2': [0.5021, 0.4927, 0.5110, 0.5134, 0.4810, 0.5196, 0.5202, 0.5216, 0.4728, 0.5145]}}
        },
        'bandgap': {
            0.2: {'InterLLC': {'MAE': [0.3086, 0.3101, 0.3071, 0.3015, 0.2920, 0.3133, 0.2943, 0.3117, 0.3024],
                                    'R2': [0.1310, 0.0881, 0.0285, 0.1692, 0.0470, 0.1178, 0.0160, 0.1137, 0.0237]},
                'Single': {'MAE': [0.3740, 0.3023, 0.3262, 0.2871, 0.3081, 0.3633, 0.3360, 0.3151, 0.3150],
                            'R2': [0.0476, 0.0644, 0.0932, 0.1064, 0.0799, 0.1104, 0.0249, 0.1124, 0.0982]}},
            0.4: {'InterLLC': {'MAE': [0.2570, 0.2823, 0.2777, 0.2655, 0.2519, 0.2679, 0.2762, 0.2942, 0.2616],
                                    'R2': [0.3431, 0.2421, 0.3186, 0.3112, 0.3594, 0.3367, 0.2818, 0.2262, 0.3360]},
                'Single': {'MAE': [0.2698, 0.2904, 0.2791, 0.2722, 0.2658, 0.2798, 0.2870, 0.2764, 0.2733],
                            'R2': [0.2604, 0.2274, 0.2450, 0.2596, 0.2664, 0.2983, 0.3014, 0.2722, 0.2852]}},
            0.6: {'InterLLC': {'MAE': [0.2429, 0.2466, 0.2527, 0.2611, 0.2580, 0.2523, 0.2584, 0.2402, 0.2342, 0.2433],
                                    'R2': [0.4196, 0.4037, 0.3712, 0.3952, 0.3812, 0.2896, 0.3610, 0.3261, 0.2885, 0.3418]},
                'Single': {'MAE': [0.2700, 0.2532, 0.2813, 0.2626, 0.2733, 0.2752, 0.2783, 0.2825, 0.2655, 0.2756],
                            'R2': [0.3037, 0.3719, 0.2868, 0.3635, 0.3398, 0.3491, 0.3098, 0.2444, 0.3298, 0.3323]}},
            0.8: {'InterLLC': {'MAE': [0.2399, 0.2281, 0.2403, 0.2398, 0.2259, 0.2457, 0.2505, 0.2440, 0.2330, 0.2449],
                                    'R2': [0.4109, 0.3803, 0.4087, 0.3971, 0.3851, 0.4115, 0.3978, 0.4021, 0.3821, 0.3996]},
                'Single': {'MAE': [0.2611, 0.2781, 0.2633, 0.2658, 0.2816, 0.2756, 0.2674, 0.2651, 0.2704, 0.2705],
                            'R2': [0.3557, 0.3114, 0.3345, 0.3688, 0.2344, 0.2981, 0.3309, 0.3551, 0.3478, 0.3030]}},
            1.0: {'InterLLC': {'MAE': [0.2229, 0.2433, 0.2357, 0.2474, 0.2333, 0.2394, 0.2470, 0.2407, 0.2492, 0.2393],
                                    'R2': [0.5163, 0.4677, 0.4650, 0.4761, 0.5005, 0.4953, 0.4629, 0.4845, 0.4711, 0.4548]},
                'Single': {'MAE': [0.2432, 0.2705, 0.2545, 0.2540, 0.2628, 0.2767, 0.2626, 0.2597, 0.2528, 0.2520],
                            'R2': [0.4005, 0.3063, 0.3445, 0.3898, 0.3830, 0.3437, 0.3711, 0.3904, 0.3773, 0.3763]}}
        }
    }

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    prop_names = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    methods = ['InterLLC', 'Single']

    box_colors = {'InterLLC': '#F7E0CF', 'Single': '#BAD2E1'}
    line_colors = {'InterLLC': '#E39C63', 'Single': '#58A8D7'}

    width = 0.2         
    offset = width / 2

    def plot_metric(metric_name, metric_key):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))  
        for i, prop in enumerate(['formation', 'fermi', 'bandgap']):
            ax = axes[i]
            boxes_data = []
            box_methods = []      
            positions = []
            medians_hier = []
            medians_single = []
            
            for idx, r in enumerate(ratios):
                x_center = idx
                values_h = data[prop][r]['InterLLC'][metric_key]
                values_s = data[prop][r]['Single'][metric_key]
                if values_h and values_s:
                    boxes_data.append(values_h)
                    box_methods.append('InterLLC')
                    positions.append(x_center - offset)
                    boxes_data.append(values_s)
                    box_methods.append('Single')
                    positions.append(x_center + offset)
                    
                    medians_hier.append(np.median(values_h))
                    medians_single.append(np.median(values_s))
            
            bp = ax.boxplot(boxes_data, positions=positions, widths=width,
                            patch_artist=True, showmeans=False,
                            medianprops=dict(linewidth=2, color='grey'),
                            whiskerprops=dict(color='gray'),
                            capprops=dict(color='gray'),
                            flierprops=dict(marker='o', markersize=2, alpha=0.5))
            
            for patch, method in zip(bp['boxes'], box_methods):
                patch.set_facecolor(box_colors[method])
                patch.set_alpha(0.7)
            
            x_pos_hier = [x_center - offset for x_center in range(len(ratios))]
            x_pos_single = [x_center + offset for x_center in range(len(ratios))]
            ax.plot(x_pos_hier, medians_hier, color=line_colors['InterLLC'],
                    marker='s', markersize=6, linewidth=2, label='InterLLC' if i==0 else "")
            ax.plot(x_pos_single, medians_single, color=line_colors['Single'],
                    marker='^', markersize=6, linewidth=2, label='Single-task only' if i==0 else "")
            
            ax.set_xticks(range(len(ratios)))
            ax.set_xticklabels([str(r) for r in ratios])
            ax.set_xlabel('Training data ratio', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            if metric_name == "R2":
                ax.set_ylabel("R²", fontsize=12)
            ax.set_title(prop_names[i], fontsize=12)
            ax.grid(False)
            if metric_name == 'MAE':
                ax.set_ylim(0.18, 0.47)          
                ax.set_yticks([0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
            else:  
                ax.set_ylim(-0.02, 0.82)        
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
            if i == 0:
                ax.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"figures/ab_{metric_name}_curve_boxes.png", dpi=300, bbox_inches='tight')
        plt.close()

    plot_metric('MAE', 'MAE')
    plot_metric('R2', 'R2')


def ab_mtl_key_curve():

    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]

    form_data = {
        'InterLLC':  [0.2774, 0.5375, 0.6157, 0.6783, 0.7133],
        'Single-task only': [0.2643, 0.4592, 0.5645, 0.6249, 0.6860]
    }
    fermi_data = {
        'InterLLC':  [0.2188, 0.3749, 0.5019, 0.5275, 0.5816],
        'Single-task only': [0.0517, 0.2464, 0.4355, 0.5017, 0.5321]
    }
    gap_data = {
        'InterLLC':  [0.1310, 0.3431, 0.4196, 0.4409, 0.5163],
        'Single-task only': [0.0476, 0.3004, 0.3037, 0.3557, 0.4605]
    }

    colors_full = '#DD542F'
    colors_other = ['#4F8CBB']
    markers = ['o', 's']
    titles = ['Formation energy', 'Fermi energy', 'Bandgap']
    data_list = [form_data, fermi_data, gap_data]
    orders = [
        ['InterLLC', 'Single-task only'],
        ['InterLLC', 'Single-task only'],
        ['InterLLC', 'Single-task only']
    ]

    for title, data, order in zip(titles, data_list, orders):
        fig, ax = plt.subplots(figsize=(2.8, 2.5))
        for i, method in enumerate(order):
            if i == 0:
                color = colors_full
                alpha = 1.0
            else:
                color = colors_other[i-1]
                alpha = 0.8
            ax.plot(ratios, data[method],
                    marker=markers[i], markersize=3,
                    color=color, linewidth=1.5, alpha=alpha,
                    label=method)
        ax.set_title(title, fontsize=8, pad=2)
        ax.set_xlabel('Training data fraction', fontsize=8)
        ax.set_ylabel('R²', fontsize=8)
        ax.set_xlim(0.15, 1.02)
        ax.set_ylim(-0.05, 0.85)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(labeltop=False, labelright=False)
        
        ax.legend(loc='lower right', frameon=True, fontsize=6, fancybox=False, edgecolor='black', shadow=False)
    
        plt.tight_layout()
        filename = f'figures/ab_R2_{title.replace(" ", "_")}_key_curve.png'
        plt.savefig(filename)
        plt.close(fig)


def plot_property_landscape(all_group_imp, out_dir="figures"):
    group_order = ['xafs', 'xanes', 'xrd', 'exafs', 'global']
    group_key_map = {
        'xafs': 'Signal',
        'xanes': 'XANES',
        'xrd': 'XRD',
        'exafs': 'EXAFS',
        'global': 'Addition'
    }
    xtick_labels = [group_key_map[g] for g in group_order]

    name_map = {
        'formation_energy': 'Formation Energy',
        'fermi_energy': 'Fermi Energy',
        'band_gap': 'Bandgap'
    }
    attrs = list(all_group_imp.keys())
    ytick_labels = [name_map.get(attr, attr) for attr in attrs]
    data = []
    for attr in attrs:
        row = [all_group_imp[attr].get(g, 0) for g in group_order]
        data.append(row)
    data = np.array(data)
    custom_colors = ['#3F719D', '#58A8D7', '#9FC0D6', '#C6DEED', '#DBF1FA']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', custom_colors[::-1])

    plt.figure(figsize=(8, 3))
    sns.heatmap(data, annot=True, fmt=".2f", cmap=cmap,
                xticklabels=xtick_labels, yticklabels=ytick_labels,
                vmin=-5, vmax=55,
                cbar_kws={'label': 'Feature Importance (%)',
                          'ticks': [0, 10, 20, 30, 40, 50]})
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/property_landscape.png", bbox_inches='tight')
    plt.close()

def plot_xrd_desc_interaction(attr_name, inter_mat, desc_names, out_dir="figures"):
    if inter_mat.size == 0:
        return
    plt.figure(figsize=(12, 8))
    sns.heatmap(inter_mat, cmap="viridis", cbar_kws={'label': 'Attention Weight'},
                xticklabels=desc_names, yticklabels=False)
    plt.xlabel("XAFS Descriptors")
    plt.ylabel("XRD Feature Dimension Index")
    plt.title(f"{attr_name} – XRD vs. XAFS Descriptors Attention")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{attr_name}_xrd_desc_attn.png", bbox_inches='tight')
    plt.close()

def plot_xrd_encoded_interaction(attr_name, enc_mat, out_dir="figures"):
    if enc_mat.size == 0:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(enc_mat, cmap="coolwarm", center=0, cbar_kws={'label': 'Attention'},
                xticklabels=False, yticklabels=False)
    plt.xlabel("XAFS Encoded Dimension")
    plt.ylabel("XRD Feature Dimension")
    plt.title(f"{attr_name} – Cross-modal Attention (Encoded Dimensions)")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{attr_name}_xrd_encoded_attn.png", bbox_inches='tight')
    plt.close()

def plot_fused_shap_curve(attr_name, shap_values, out_dir="figures"):
    if shap_values is None:
        return
    shap_arr = np.array(shap_values)
    if shap_arr.size == 0:
        return
    x = np.arange(shap_arr.shape[0])
    plt.figure(figsize=(12, 4))
    plt.plot(x, shap_arr, color='#2E86AB', linewidth=1.5, marker='.', markersize=2)
    plt.xlabel("Fused Feature Dimension Index")
    plt.ylabel("Feature Importance")
    plt.title(f"{attr_name} – Fused Feature Importance")
    sns.despine()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{attr_name}_fused_shap_curve.png", bbox_inches='tight')
    plt.close()

def plot_fused_corr_matrix(attr_name, corr_mat, out_dir="figures"):
    if corr_mat is None or corr_mat.size == 0:
        return
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, cmap="RdBu_r", center=0, cbar_kws={'label': 'Correlation'},
                xticklabels=False, yticklabels=False)
    plt.title(f"{attr_name} – Fused Feature Correlation Matrix")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{attr_name}_fused_corr.png", bbox_inches='tight')
    plt.close()

def plot_all_fused_shap_curves(all_shap_data, out_dir="figures"):
    if not all_shap_data:
        return
    colors = ['#DD542F', '#537EBF', '#12AF62']
    name_map = {
        'formation_energy': 'Formation Energy',
        'fermi_energy': 'Fermi Energy',
        'band_gap': 'Bandgap'
    }
    plt.figure(figsize=(12, 4))
    for i, (attr_name, shap_vals) in enumerate(all_shap_data.items()):
        shap_arr = np.array(shap_vals)
        if shap_arr.size == 0:
            continue
        if shap_arr.ndim == 2 and shap_arr.shape[1] == 1:
            shap_arr = shap_arr.flatten()
        x = np.arange(len(shap_arr))
        label = name_map.get(attr_name, attr_name)
        plt.plot(x, shap_arr, color=colors[i], linewidth=1.5, marker='.', markersize=2, label=label)
    plt.xlabel("Fused Feature Dimension Index")
    plt.ylabel("Feature Importance")
    plt.legend()
    sns.despine()
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/all_fused_shap_curves.png", bbox_inches='tight')
    plt.close()

import json

def plot_group_importance_combined(all_group_imp, all_sample_group_imp, out_dir="figures"):
    group_display_order = ['Signal', 'XANES', 'XRD', 'EXAFS', 'Addition']  
    group_key_map = {
        'xafs': 'Signal',
        'xanes': 'XANES',
        'xrd': 'XRD',
        'exafs': 'EXAFS',
        'global': 'Addition'   
    }
    
    attr_display_map = {
        'formation_energy': 'Formation Energy',
        'fermi_energy': 'Fermi Energy',
        'band_gap': 'Bandgap'
    }
    
    attributes = list(all_group_imp.keys())
    attr_display = [attr_display_map.get(attr, attr) for attr in attributes]
    
    normalized_groups = group_display_order  

    color_palette = ['#D1E4CF','#F7E0CF', '#58A8D7', '#BAD2E1', '#A9C37F']
    group_color_map = {group: color_palette[i] for i, group in enumerate(normalized_groups)}
    
    df_bar = pd.DataFrame(index=normalized_groups, columns=attr_display)
    for attr, attr_disp in zip(attributes, attr_display):
        raw_group_imp = all_group_imp[attr]  
        for display_grp in normalized_groups:
            original_key = None
            for k, v in group_key_map.items():
                if v == display_grp:
                    original_key = k
                    break
            if original_key and original_key in raw_group_imp:
                df_bar.loc[display_grp, attr_disp] = raw_group_imp[original_key]
            else:
                df_bar.loc[display_grp, attr_disp] = 0
    
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    n_groups = len(normalized_groups)   
    n_attrs = len(attributes)         
    bar_width = 0.8 / n_groups
    x_positions = np.arange(n_attrs)
    
    for i, grp in enumerate(normalized_groups):
        offset = (i - n_groups/2 + 0.5) * bar_width
        ax1.bar(x_positions + offset, df_bar.loc[grp, attr_display].values,
                width=bar_width, label=grp,
                color=group_color_map[grp],
                edgecolor='black', linewidth=0.5)
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(attr_display, fontsize=14)
    ax1.set_ylabel("Feature Importance (%)", fontsize=14)
    ax1.set_ylim(0, 60)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/combined_group_importance.png", bbox_inches='tight')
    plt.close(fig1)
    
    n_attrs = len(attributes)
    fig2, axes = plt.subplots(1, n_attrs, figsize=(5 * n_attrs, 4), sharey=True)
    if n_attrs == 1:
        axes = [axes]
    
    for ax, attr, attr_disp in zip(axes, attributes, attr_display):
        if attr in all_sample_group_imp and all_sample_group_imp[attr] is not None:
            sample_list = all_sample_group_imp[attr]  

            df_samples_list = []
            for sample in sample_list:
                row = {}
                for k, v in sample.items():
                    display_name = group_key_map.get(k, k)
                    if display_name in normalized_groups:
                        row[display_name] = v
                df_samples_list.append(row)
            df_samples = pd.DataFrame(df_samples_list)

            df_samples = df_samples[normalized_groups]
            df_melt = df_samples.melt(var_name='Group', value_name='SHAP')
            df_melt_filtered = df_melt[~((df_melt['Group'] == 'XRD') & (df_melt['SHAP'] > 60)) &
                                       ~((df_melt['Group'] == 'Addition') & (df_melt['SHAP'] > 40))]
            sns.violinplot(data=df_melt_filtered, x='Group', y='SHAP', ax=ax,
                            inner='quartile', palette=group_color_map,
                            order=normalized_groups, cut=0)
            ax.set_xlabel('')
            ax.set_title(attr_disp, fontsize=15)   
            if ax == axes[0]:
                ax.set_ylabel("Feature Importance (%)", fontsize=15)
            else:
                ax.set_ylabel("")
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
            ax.grid(axis='y', linestyle='--', alpha=0.5)
        else:
            ax.text(0.5, 0.5, "No sample data", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_title(attr_disp)
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/combined_sample_distribution.png", bbox_inches='tight')
    plt.close(fig2)


import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale

def plot_sunburst(attr_name, group_imp, desc_imp, out_dir="figures"):
    group_key_map = {
        'XAS': 'XAFS',
        'XAFS': 'Signal',
        'xanes': 'XANES',
        'xrd': 'XRD',
        'exafs': 'EXAFS',
        'global': 'Addition'
    }
    
    required_groups = ['xrd', 'xafs', 'xanes', 'exafs', 'global']
    for g in required_groups:
        if g not in group_imp:
            print(f"Warning: '{g}' group not found for {attr_name}, skip sunburst.")
            return

    EPS = 1e-6
    xas_value = group_imp['xafs'] + group_imp['xanes'] + group_imp['exafs'] + group_imp['global']
    xrd_value = group_imp['xrd']
    total = xrd_value + xas_value
    if abs(total - 100.0) > 0.01:
        print(f"Warning: total XRD+XAS = {total:.2f}%, renormalizing to 100%.")
        scale = 100.0 / total
        xrd_value *= scale
        xas_value *= scale
        for g in required_groups:
            group_imp[g] = group_imp[g] * scale
        xas_value = group_imp['xafs'] + group_imp['xanes'] + group_imp['exafs'] + group_imp['global']

    if xrd_value <= 0:
        xrd_value = EPS
    if xas_value <= 0:
        xas_value = EPS
    for g in ['xafs', 'xanes', 'exafs', 'global']:
        if group_imp[g] <= 0:
            group_imp[g] = EPS

    labels = []
    parents = []
    values = []

    root_label = attr_name
    name_map = {
        'formation_energy': 'Formation Energy',
        'fermi_energy': 'Fermi Energy',
        'band_gap': 'Bandgap'
    }
    if attr_name.lower() in name_map:
        root_label = name_map[attr_name.lower()]

    labels.append(root_label)
    parents.append("")
    values.append(100.0)

    labels.append(group_key_map['xrd']) 
    parents.append(root_label)
    values.append(xrd_value)

    labels.append(group_key_map['XAS'])  
    parents.append(root_label)
    values.append(xas_value)

    for sub_group in ['xafs', 'xanes', 'exafs', 'global']:
        if sub_group == 'xafs':
            disp_name = group_key_map['XAFS']  # 'Signal'
        elif sub_group == 'xanes':
            disp_name = group_key_map['xanes']  # 'XANES'
        elif sub_group == 'exafs':
            disp_name = group_key_map['exafs']  # 'EXAFS'
        elif sub_group == 'global':
            disp_name = group_key_map['global']  # 'Addition'
        else:
            disp_name = sub_group.upper()
            
        labels.append(disp_name)
        parents.append(group_key_map['XAS']) 
        values.append(group_imp[sub_group])

    df = pd.DataFrame({'labels': labels, 'parents': parents, 'values': values})

    color_list = ['#58A8D7', '#96C2D4', '#BAD2E1', '#D8E5F7', '#DBF1FA']
    color_list_rev = color_list[::-1]  # ['#DBF1FA', '#D8E5F7', '#BAD2E1', '#96C2D4', '#6CBAD8']
    n_colors = len(color_list_rev)
    custom_colorscale = [[i/(n_colors-1), color_list_rev[i]] for i in range(n_colors)]

    root_color = '#f0f0f0'
    non_root_values = values[1:]
    if len(non_root_values) > 0:
        vmin_fixed, vmax_fixed = 0.0, 50.0
        norm_vals = []
        for v in non_root_values:
            v_clipped = max(vmin_fixed, min(v, vmax_fixed))
            norm = (v_clipped - vmin_fixed) / (vmax_fixed - vmin_fixed)
            norm_vals.append(norm)
        colors_non_root = sample_colorscale(custom_colorscale, norm_vals)
        color_list_all = [root_color] + colors_non_root
    else:
        color_list_all = [root_color]

    fig = go.Figure(go.Sunburst(
        labels=df['labels'],
        parents=df['parents'],
        values=df['values'],
        branchvalues='total',
        marker=dict(colors=color_list_all, line=dict(color='white', width=1)),
        textinfo='label+percent entry', 
        textfont=dict(size=14),  
        rotation=60,  
        hovertemplate='<b>%{label}</b><br>值: %{value:.2f}%<br>占比: %{percentEntry:.1f}%<extra></extra>'
    ))

    dummy_df = pd.DataFrame({'x': [1e6, 1e6], 'y': [1e6, 1e6], 'c': [-5, 55]})  
    fig.add_trace(go.Scatter(
        x=dummy_df['x'],
        y=dummy_df['y'],
        mode='markers',
        marker=dict(
            colorscale=custom_colorscale,  
            color=dummy_df['c'],
            size=0,
            opacity=0,
            colorbar=dict(
                title={
                    'text': "Feature Importance (%)",
                    'side': 'right',  
                    'font': dict(size=14)  
                },
                thickness=15,
                len=0.75,
                tickvals=[0, 10, 20, 30, 40, 50],
                ticktext=['0', '10', '20', '30', '40', '50'],
                ticks='outside', 
                ticklen=5, 
                tickwidth=1,  
            ),
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))

    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, showgrid=False, showticklabels=False, showline=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, showticklabels=False, showline=False, zeroline=False),
        showlegend=False,
        font=dict(size=15)
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{attr_name}_sunburst.png")
    fig.write_image(out_path)
    print(f"Sunburst chart saved to {out_path}")

def plot_sunburst_1(attr_name, group_imp, desc_imp, out_dir="figures"):
    group_key_map = {
        'XAS': 'XAFS',
        'XAFS': 'Signal',
        'xanes': 'XANES',
        'xrd': 'XRD',
        'exafs': 'EXAFS',
        'global': 'Addition'
    }
    
    required_groups = ['xrd', 'xafs', 'xanes', 'exafs', 'global']
    for g in required_groups:
        if g not in group_imp:
            print(f"Warning: '{g}' group not found for {attr_name}, skip sunburst.")
            return

    EPS = 1e-6
    xas_value = group_imp['xafs'] + group_imp['xanes'] + group_imp['exafs'] + group_imp['global']
    xrd_value = group_imp['xrd']
    total = xrd_value + xas_value
    if abs(total - 100.0) > 0.01:
        print(f"Warning: total XRD+XAS = {total:.2f}%, renormalizing to 100%.")
        scale = 100.0 / total
        xrd_value *= scale
        xas_value *= scale
        group_imp = {g: group_imp[g] * scale for g in required_groups}
        xas_value = group_imp['xafs'] + group_imp['xanes'] + group_imp['exafs'] + group_imp['global']

    for g in ['xrd', 'xafs', 'xanes', 'exafs', 'global']:
        if group_imp[g] <= 0:
            print(f"Warning: '{g}' value is {group_imp[g]}, setting to {EPS}")
            group_imp[g] = EPS
    if xrd_value <= 0:
        xrd_value = EPS
    if xas_value <= 0:
        xas_value = EPS

    from configurations import xafs_group, xanes_group, exafs_group, simple_feature_names
    group_to_desc = {
        'xafs':  [d for d in xafs_group if d in desc_imp],
        'xanes': [d for d in xanes_group if d in desc_imp],
        'exafs': [d for d in exafs_group if d in desc_imp],
        'global':[d for d in simple_feature_names if d in desc_imp]
    }

    labels = []
    parents = []
    values = []

    root_label = attr_name
    name_map = {
        'formation_energy': f'Formation\nEnergy',
        'fermi_energy': f'Fermi\nEnergy',
        'band_gap': f'Bandgap\nEnergy'
    }
    if attr_name.lower() in name_map:
        root_label = name_map[attr_name.lower()]
    labels.append(root_label)
    parents.append("")
    values.append(100.0)

    labels.append(group_key_map['xrd']) 
    parents.append(root_label)
    values.append(xrd_value)

    labels.append(group_key_map['XAS'])
    parents.append(root_label)
    values.append(xas_value)

    for sub_group in ['xafs', 'xanes', 'exafs', 'global']:
        if sub_group == 'xafs':
            disp_name = group_key_map['XAFS'] 
        elif sub_group == 'xanes':
            disp_name = group_key_map['xanes']  
        elif sub_group == 'exafs':
            disp_name = group_key_map['exafs']  
        elif sub_group == 'global':
            disp_name = group_key_map['global']  
        else:
            disp_name = sub_group.upper()
            
        labels.append(disp_name)
        parents.append(group_key_map['XAS'])  
        values.append(group_imp[sub_group])

    for group, desc_list in group_to_desc.items():
        if group == 'xafs':
            parent_name = group_key_map['XAFS'] 
        elif group == 'xanes':
            parent_name = group_key_map['xanes'] 
        elif group == 'exafs':
            parent_name = group_key_map['exafs'] 
        elif group == 'global':
            parent_name = group_key_map['global']  
        else:
            parent_name = group.upper()
            
        for desc in desc_list:
            labels.append(desc)
            parents.append(parent_name)
            values.append(desc_imp[desc])

    df = pd.DataFrame({'labels': labels, 'parents': parents, 'values': values})
    root_color = '#f0f0f0'
    non_root_values = values[1:]
    if len(non_root_values) > 0:
        vmin_fixed, vmax_fixed = 0.0, 50.0
        norm_vals = []
        for v in non_root_values:
            v_clipped = max(vmin_fixed, min(v, vmax_fixed))
            norm = (v_clipped - vmin_fixed) / (vmax_fixed - vmin_fixed)
            norm_vals.append(norm)
        colors_non_root = sample_colorscale('Blues', norm_vals)
        color_list = [root_color] + colors_non_root
    else:
        color_list = [root_color]

    fig = go.Figure(go.Sunburst(
        labels=df['labels'],
        parents=df['parents'],
        values=df['values'],
        branchvalues='total',
        marker=dict(colors=color_list, line=dict(color='white', width=1)),
        textinfo='label+percent entry', 
        textfont=dict(size=14), 
        rotation=60, 
        hovertemplate='<b>%{label}</b><br>值: %{value:.2f}%<br>占比: %{percentEntry:.1f}%<extra></extra>'
    ))

    dummy_df = pd.DataFrame({'x': [1e6, 1e6], 'y': [1e6, 1e6], 'c': [-5, 55]})  
    fig.add_trace(go.Scatter(
        x=dummy_df['x'],
        y=dummy_df['y'],
        mode='markers',
        marker=dict(
            colorscale='Blues',
            color=dummy_df['c'],
            size=0,
            opacity=0,
            colorbar=dict(
                title={
                    'text': "Feature Importance (%)",
                    'side': 'right', 
                    'font': dict(size=14)  
                },
                thickness=15,
                len=0.75,
                tickvals=[0, 10, 20, 30, 40, 50],
                ticktext=['0', '10', '20', '30', '40', '50'],
                ticks='outside', 
                ticklen=5,  
                tickwidth=1,  
            ),
            showscale=True
        ),
        hoverinfo='none',
        showlegend=False
    ))

    fig.update_layout(
        margin=dict(t=30, l=0, r=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, showgrid=False, showticklabels=False, showline=False, zeroline=False),
        yaxis=dict(visible=False, showgrid=False, showticklabels=False, showline=False, zeroline=False),
        showlegend=False,
        font=dict(size=16)
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{attr_name}_sunburst_1.png")
    fig.write_image(out_path)
    print(f"Sunburst chart saved to {out_path}")

def plot_feature_importance(json_path="res/shap_hierarchical_importance.json",
                            samples_json_path="res/shap_hierarchical_importance_samples.json"):
    with open(json_path, 'r') as f:
        data = json.load(f)

    samples_data = {}
    if os.path.exists(samples_json_path):
        with open(samples_json_path, 'r') as f:
            samples = json.load(f)
        for task_name, sample_list in samples.items():
            group_imp_list = []
            for sample in sample_list:
                if "group_importance" in sample:
                    group_imp_list.append(sample["group_importance"])
            if group_imp_list:
                samples_data[task_name] = group_imp_list
    else:
        print(f"Warning: samples file {samples_json_path} not found, skip distribution plots.")
    
    all_group_imp = {}
    all_sample_group_imp = {}
    all_fused_shap = {}

    for task_name, res in data.items():
        group_imp = res.get('group_importance', {})
        all_group_imp[task_name] = group_imp
        task_samples_grp = samples_data.get(task_name, None) if samples_data else None
        all_sample_group_imp[task_name] = task_samples_grp
        
        desc_imp = res.get('descriptor_importance', {})
        plot_sunburst(task_name, group_imp, desc_imp)
        plot_sunburst_1(task_name, group_imp, desc_imp)

    if len(all_group_imp) >= 1:
        plot_group_importance_combined(all_group_imp, all_sample_group_imp)
    else:
        print("No property data, skip combined plots.")
    
    if len(all_group_imp) > 1:
        plot_property_landscape(all_group_imp)
    else:
        print("Only one property, skip property landscape heatmap.")


def plot_ab_feat_augment():
    data_raw = {
        "Model": ["InterLLC", "w/o XRD", "w/o Signal", "w/o XANES", "w/o EXAFS", "w/o Addition"],
        "MAE_fe": [0.2037, 0.2556, 0.2172, 0.225, 0.2043, 0.2193],
        "RMSE_fe": [0.2707, 0.3553, 0.2836, 0.2966, 0.2778, 0.304],
        "R2_fe": [0.7133, 0.5058, 0.6851, 0.6557, 0.6979, 0.6383],
        "MAE_fermi": [0.236, 0.2753, 0.2408, 0.269, 0.251, 0.2603],
        "RMSE_fermi": [0.3086, 0.3624, 0.3205, 0.3455, 0.3267, 0.3403],
        "R2_fermi": [0.5816, 0.423, 0.5488, 0.4757, 0.5311, 0.4914],
        "MAE_band": [0.2229, 0.2637, 0.2417, 0.2579, 0.2354, 0.2388],
        "RMSE_band": [0.3506, 0.3836, 0.383, 0.386, 0.3718, 0.3724],
        "R2_band": [0.5163, 0.4211, 0.4228, 0.4138, 0.456, 0.4542],
    }
    df = pd.DataFrame(data_raw).set_index("Model")

    baseline = df.loc["InterLLC"]
    variants_order = ["w/o XRD", "w/o XANES", "w/o Addition", "w/o Signal", "w/o EXAFS"]
    variants = [v for v in variants_order if v in df.index]
    colors = ['#D1E4CF', '#F7E0CF', '#58A8D7', '#BAD2E1', '#A9C37F']

    properties = ["Formation Energy", "Fermi Energy", "Bandgap"]
    prop_keys = ["fe", "fermi", "band"]

    delta_mae = {}
    delta_r2 = {}
    for var in variants:
        delta_mae[var] = []
        delta_r2[var] = []
        for key in prop_keys:
            mae_base = baseline[f"MAE_{key}"]
            r2_base = baseline[f"R2_{key}"]
            mae_var = df.loc[var, f"MAE_{key}"]
            r2_var = df.loc[var, f"R2_{key}"]
            delta_mae[var].append(mae_var - mae_base)
            delta_r2[var].append(r2_var - r2_base)

    delta_mae_arr = np.array([delta_mae[v] for v in variants])
    delta_r2_arr  = np.array([delta_r2[v] for v in variants])

    def plot_delta_bars(delta_data, ylabel, filename, ylim=None, legend_loc='upper right', text_shift=0.0005):
        n_props = len(properties)
        n_vars = len(variants)
        x = np.arange(n_props)
        width = 0.15
        offsets = np.linspace(-width * (n_vars - 1) / 2, width * (n_vars - 1) / 2, n_vars)

        plt.figure(figsize=(7, 5))
        ax = plt.gca()

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        for i, (var, color) in enumerate(zip(variants, colors)):
            bars = ax.bar(x + offsets[i], delta_data[i], width,
                          label=var, color=color, edgecolor='black', linewidth=0.5)

            for bar, val in zip(bars, delta_data[i]):
                if abs(val) > 1e-6:
                    y_offset = text_shift*2 if val >= 0 else -text_shift*2
                    label_text = f'{val:.3f}'  
                    ax.text(bar.get_x() + bar.get_width()/2, val + y_offset,
                            label_text, ha='center', va='bottom',
                            rotation=90, 
                            fontsize=12)

        ax.set_xticks(x)
        ax.set_xticklabels(properties, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labeltop=False, labelright=False)
        ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
        ax.grid(False)
        ax.set_axisbelow(True)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.legend(loc=legend_loc, frameon=True, fancybox=False, fontsize=12, edgecolor='black', shadow=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename)
        plt.close()
        print(f"Figure saved: {filename}")

    plot_delta_bars(delta_mae_arr, r"$\Delta$MAE (eV)",
                    "figures/ablation_delta_aug_mae.png",
                    ylim=(-0.015, 0.085), legend_loc='upper right', text_shift=0.0005)

    plot_delta_bars(delta_r2_arr, r"$\Delta R^2$",
                    "figures/ablation_delta_aug_R2.png",
                    ylim=(-0.27, 0.02), legend_loc='lower right', text_shift=0.02)

def plot_ab_feat_augment_1():
    data_dict = {
        'EXAFS':         [0.3178, 0.2723, 0.3723, 0.1477, 0.3679, 0.0966],
        'XANES':         [0.2999, 0.3734, 0.3276, 0.2785, 0.327, 0.2611],
        'XRD+EXAFS':     [0.2611, 0.5590, 0.2802, 0.4167, 0.3067, 0.2225],
        'XRD':           [0.2629, 0.5609, 0.2645, 0.4755, 0.2976, 0.2686],
        'XRD+XANES':     [0.2270, 0.6464, 0.2421, 0.5452, 0.2435, 0.3923],
        'InterLLC': [0.2037, 0.7133, 0.2360, 0.5816, 0.2229, 0.5163]
    }

    models_order = list(data_dict.keys())
    attributes = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    colors = ['#D1E4CF', '#F7E0CF', '#58A8D7', '#BAD2E1', '#A9C37F', '#E39C63']

    mae_by_attr = {attr: [] for attr in attributes}
    r2_by_attr  = {attr: [] for attr in attributes}
    for model in models_order:
        vals = data_dict[model]
        mae_by_attr['Formation Energy'].append(vals[0])
        r2_by_attr['Formation Energy'].append(vals[1])
        mae_by_attr['Fermi Energy'].append(vals[2])
        r2_by_attr['Fermi Energy'].append(vals[3])
        mae_by_attr['Bandgap'].append(vals[4])
        r2_by_attr['Bandgap'].append(vals[5])

    df_mae = pd.DataFrame(mae_by_attr, index=models_order)
    df_r2  = pd.DataFrame(r2_by_attr, index=models_order)

    n_models = len(models_order)
    bar_width = 0.12
    x = np.arange(len(attributes))
    total_width = (n_models - 1) * bar_width
    offsets = np.linspace(-total_width / 2, total_width / 2, n_models)

    def plot_bars(df, ylabel, ylim, filename):
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for i, model in enumerate(models_order):
            values = [df.loc[model, attr] for attr in attributes]
            bars = ax.bar(x + offsets[i], values, width=bar_width,
                          color=colors[i], edgecolor='black', linewidth=0.5,
                          label=model)
            
            for bar in bars:
                height = bar.get_height()
                if ylabel == 'MAE (eV)':
                    offset = 0.005
                    precision = 3
                else:  
                    offset = 0.015
                    precision = 3
                
                ax.text(bar.get_x() + bar.get_width()/2., height + offset,
                        f'{height:.{precision}f}',
                        ha='center', va='bottom',
                        rotation=90,
                        fontsize=12,
                        color='black')

        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(attributes, fontsize=14)
        
        max_value = df.values.max()
        if ylabel == 'MAE (eV)':
            ax.set_ylim(ylim[0], max(ylim[1], max_value * 1.12))
        else: 
            ax.set_ylim(ylim[0], max(ylim[1], max_value * 1.08))
        
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
        ax.tick_params(top=False, right=False)

        left_edge = x[0] + min(offsets) - bar_width / 2
        right_edge = x[-1] + max(offsets) + bar_width / 2
        ax.set_xlim(left_edge - 0.1, right_edge + 0.1)

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99),
                  ncol=3, frameon=True, fancybox=False, edgecolor='black', shadow=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename)
        plt.close(fig)

    plot_bars(df_mae, 'MAE (eV)', (0.18, 0.48), 'figures/ablation_feat_aug_mae.png')
    plot_bars(df_r2, 'R²', (0.0, 1.05), 'figures/ablation_feat_aug_R2.png')

def plot_ab_feat_augment_statistic():
    data_raw = {
        "EXAFS": [
            [0.3178, 0.2723, 0.3723, 0.1477, 0.3679, 0.0966],
            [0.3228, 0.2805, 0.3689, 0.1571, 0.3711, 0.1005],
            [0.3259, 0.2779, 0.3719, 0.1456, 0.3747, 0.0882],
            [0.3179, 0.2806, 0.3764, 0.0940, 0.3680, 0.1048],
            [0.3192, 0.2769, 0.3730, 0.1009, 0.3611, 0.0863],
            [0.3141, 0.2920, 0.3727, 0.1045, 0.3715, 0.0860],
            [0.3180, 0.2819, 0.3627, 0.1609, 0.3727, 0.0819],
            [0.3222, 0.2691, 0.3656, 0.1434, 0.3759, 0.0752],
            [0.3179, 0.2600, 0.3748, 0.1273, 0.3746, 0.0857],
            [0.3149, 0.2577, 0.3779, 0.0877, 0.3599, 0.1015],
        ],
        "XANES": [
            [0.2999, 0.3734, 0.3276, 0.2785, 0.3270, 0.2611],
            [0.3388, 0.2719, 0.3704, 0.1460, 0.3685, 0.1732],
            [0.2963, 0.3956, 0.3274, 0.2848, 0.3265, 0.2855],
            [0.3136, 0.3411, 0.3649, 0.1573, 0.3576, 0.2146],
            [0.3093, 0.3427, 0.3402, 0.2665, 0.3353, 0.2325],
            [0.3538, 0.2296, 0.3729, 0.1418, 0.3809, 0.1698],
            [0.3389, 0.2656, 0.3723, 0.1381, 0.3758, 0.1754],
            [0.3366, 0.2635, 0.3704, 0.1442, 0.3762, 0.1517],
            [0.3334, 0.2841, 0.3686, 0.1685, 0.3729, 0.1694],
            [0.3148, 0.3322, 0.3384, 0.2589, 0.3466, 0.2238],
        ],
        "XRD+EXAFS": [
            [0.2611, 0.5590, 0.2802, 0.4167, 0.3067, 0.2225],
            [0.2563, 0.5478, 0.2530, 0.4895, 0.2884, 0.2733],
            [0.2765, 0.4932, 0.2669, 0.4451, 0.3054, 0.1509],
            [0.2561, 0.5536, 0.2481, 0.5128, 0.2821, 0.2940],
            [0.2665, 0.5217, 0.2540, 0.4836, 0.2831, 0.2677],
            [0.2642, 0.5183, 0.2657, 0.4394, 0.3042, 0.2193],
            [0.2782, 0.4932, 0.2834, 0.4114, 0.3256, 0.1445],
            [0.2882, 0.4620, 0.2774, 0.4176, 0.2954, 0.2035],
            [0.2580, 0.5627, 0.2772, 0.3839, 0.3028, 0.2287],
            [0.2673, 0.5263, 0.2757, 0.4321, 0.3037, 0.2189],
        ],
        "XRD": [
            [0.2629, 0.5609, 0.2845, 0.4455, 0.2976, 0.2686],
            [0.2870, 0.4941, 0.3042, 0.4260, 0.3090, 0.2294],
            [0.2556, 0.5474, 0.2878, 0.4504, 0.2954, 0.2899],
            [0.2556, 0.5474, 0.2878, 0.4504, 0.2954, 0.2899],
            [0.2813, 0.5007, 0.2950, 0.4264, 0.3139, 0.2605],
            [0.3104, 0.4341, 0.3028, 0.4299, 0.3519, 0.2034],
            [0.2728, 0.5111, 0.2940, 0.4202, 0.3125, 0.2483],
            [0.3046, 0.4376, 0.2892, 0.4113, 0.3381, 0.2301],
            [0.2961, 0.5083, 0.2958, 0.4308, 0.3237, 0.2548],
            [0.2724, 0.5216, 0.2937, 0.4312, 0.3136, 0.2353],
        ],
        "XRD+XANES": [
            [0.2270, 0.6464, 0.2421, 0.5452, 0.2435, 0.3903],
            [0.2329, 0.6026, 0.2570, 0.5058, 0.2554, 0.4123],
            [0.2329, 0.6026, 0.2570, 0.5058, 0.2554, 0.4123],
            [0.2227, 0.6576, 0.2514, 0.5293, 0.2367, 0.4494],
            [0.2209, 0.6514, 0.2528, 0.4962, 0.2520, 0.4037],
            [0.2402, 0.5998, 0.2680, 0.4539, 0.2499, 0.4364],
            [0.2343, 0.6460, 0.2463, 0.5338, 0.2547, 0.3963],
            [0.2282, 0.6532, 0.2607, 0.4993, 0.2579, 0.3653],
            [0.2350, 0.6182, 0.2646, 0.4730, 0.2382, 0.4410],
            [0.2109, 0.6852, 0.2433, 0.5498, 0.2360, 0.4204],
        ],
        "InterLLC": [
            [0.2037, 0.7133, 0.2360, 0.5816, 0.2329, 0.4963],
            [0.2197, 0.6531, 0.2302, 0.5242, 0.2433, 0.4277],
            [0.2143, 0.6759, 0.2462, 0.5503, 0.2257, 0.4650],
            [0.2037, 0.6868, 0.2431, 0.5585, 0.2474, 0.4261],
            [0.2095, 0.6947, 0.2477, 0.5673, 0.2333, 0.5005],
            [0.2034, 0.6968, 0.2448, 0.5783, 0.2394, 0.4353],
            [0.2283, 0.6355, 0.2440, 0.5653, 0.2470, 0.4229],
            [0.2222, 0.6474, 0.2283, 0.5239, 0.2407, 0.4245],
            [0.2154, 0.6498, 0.2230, 0.5769, 0.2452, 0.4411],
            [0.2170, 0.6680, 0.2457, 0.5559, 0.2393, 0.4248],
        ],
    }

    properties = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    metrics = ['MAE', 'R2']

    model_order = ['EXAFS', 'XANES', 'XRD+EXAFS', 'XRD', 'XRD+XANES', 'InterLLC']
    model_labels = model_order   
    custom_colors = ['#D1E4CF', '#F7E0CF', '#58A8D7', '#BAD2E1', '#A9C37F', '#E39C63']

    records = []
    for model, rows in data_raw.items():
        for rep_idx, row in enumerate(rows):
            records.append([model, rep_idx, properties[0], 'MAE', row[0]])
            records.append([model, rep_idx, properties[0], 'R2', row[1]])
            records.append([model, rep_idx, properties[1], 'MAE', row[2]])
            records.append([model, rep_idx, properties[1], 'R2', row[3]])
            records.append([model, rep_idx, properties[2], 'MAE', row[4]])
            records.append([model, rep_idx, properties[2], 'R2', row[5]])

    df = pd.DataFrame(records, columns=['Model', 'Rep', 'Property', 'Metric', 'Value'])

    for prop in properties:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        
        data_mae = df[(df['Property'] == prop) & (df['Metric'] == 'MAE')]
        sns.boxplot(x='Model', y='Value', data=data_mae, order=model_order,
                    palette=custom_colors, width=0.4, ax=ax_top)
        ax_top.set_title(prop, fontsize=12)
        ax_top.set_ylabel('MAE (eV)', fontsize=10)
        ax_top.set_ylim(0.2, 0.4)
        ax_top.set_xlabel('')
        ax_top.tick_params(axis='x', labelbottom=False)
        
        data_r2 = df[(df['Property'] == prop) & (df['Metric'] == 'R2')]
        sns.boxplot(x='Model', y='Value', data=data_r2, order=model_order,
                    palette=custom_colors, width=0.4, ax=ax_bottom)
        ax_bottom.set_ylabel('R²', fontsize=10)
        y_min_r2 = max(0.0, data_r2['Value'].min() - 0.05)
        y_max_r2 = min(1.0, data_r2['Value'].max() + 0.05)
        ax_bottom.set_ylim(y_min_r2, y_max_r2)
        ax_bottom.set_xlabel('')
        ax_bottom.set_xticklabels(model_labels, fontsize=10, rotation=45, ha='right')
        
        plt.tight_layout()
        filename = f'figures/{prop.replace(" ", "_")}_feataug_boxplots.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {filename}")

    full_model = 'InterLLC'
    variants = [m for m in model_order if m != full_model]  
    pval_list = []

    for prop in properties:
        for metric in metrics:
            full_vals = df[(df['Model'] == full_model) & (df['Property'] == prop) & (df['Metric'] == metric)]\
                        .sort_values('Rep')['Value'].values
            for var in variants:
                var_vals = df[(df['Model'] == var) & (df['Property'] == prop) & (df['Metric'] == metric)]\
                        .sort_values('Rep')['Value'].values
                if metric == 'MAE':
                    if np.all(var_vals == full_vals):
                        p = 1.0
                    else:
                        _, p = wilcoxon(var_vals, full_vals, alternative='greater')
                else: 
                    if np.all(full_vals == var_vals):
                        p = 1.0
                    else:
                        _, p = wilcoxon(full_vals, var_vals, alternative='greater')
                pval_list.append([prop, metric, var, p])

    df_pvals = pd.DataFrame(pval_list, columns=['Property', 'Metric', 'Variant', 'p_value'])

    for prop in properties:
        metric = 'MAE'
        plt.figure(figsize=(5, 2.5))
        
        sub = df_pvals[(df_pvals['Property'] == prop) & (df_pvals['Metric'] == metric)]
        sub = sub.set_index('Variant').reindex(variants).reset_index()
        x = np.arange(len(variants))
        p_vals = sub['p_value'].values
        
        bars = plt.bar(x, p_vals, color='#BAD2E1', edgecolor='black', width=0.6) 
        
        plt.axhline(y=0.05, color='gray', linestyle=':', linewidth=0.8, label='α = 0.05')
        plt.axhline(y=0.01, color='#58A8D7', linestyle='--', linewidth=1.2, label='p = 0.01')
        plt.axhline(y=0.005, color='#E39C63', linestyle='--', linewidth=1.2, label='p = 0.005')
        
        for i, (bar, p) in enumerate(zip(bars, p_vals)):
            if p < 0.005:
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, '***',
                        ha='center', va='bottom', fontsize=9, color='red')
            elif p < 0.01:
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, '**',
                        ha='center', va='bottom', fontsize=9, color='red')
            elif p < 0.05:
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, '*',
                        ha='center', va='bottom', fontsize=9, color='red')
        
        plt.ylabel('p-value', fontsize=9)
        plt.title(f'{prop} (MAE)', fontsize=10)
        plt.xticks(x, variants, fontsize=10, rotation=45, ha='right')

        y_max = max(p_vals.max(), 0.08) * 1.05
        plt.ylim(0, y_max)
        plt.legend(fontsize=7, loc='upper right')
        plt.tight_layout()
        
        filename = f'figures/{prop.replace(" ", "_")}_feataug_MAE_pvalue_bars.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {filename}")
    

import json

def get_mid_set_from_samples(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    mids = set()
    for task_name, samples in data.items():
        for sample in samples:
            if 'mid' in sample:
                mids.add(sample['mid'])
    return mids

def load_mid_formula_mapping_filtered(txt_path, valid_mids):
    mid_to_formula = {}
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                mid = parts[0].strip()
                formula = parts[1].strip()
                if mid in valid_mids:
                    mid_to_formula[mid] = formula
    return mid_to_formula

def mid_formula():
    json_file = "res/shap_hierarchical_importance_samples.json"
    txt_file = "abx3_mid_formula.txt"

    valid_mids = get_mid_set_from_samples(json_file)
    print(f"JSON 文件中共有 {len(valid_mids)} 个唯一的 mid")

    mid_to_formula = load_mid_formula_mapping_filtered(txt_file, valid_mids)
    print(f"成功匹配到 {len(mid_to_formula)} 个 mid-formula 对")

    for mid, formula in mid_to_formula.items():
        print(f"{mid}: {formula}")

def draw_curly_brace(ax, x1, x2, y_base, y_tip, color='black', lw=0.8):
    mid_x = (x1 + x2) / 2
    width = x2 - x1
    a = (y_base - y_tip) / ((width / 2) ** 2)
    x_vals = np.linspace(x1, x2, 150)
    y_vals = a * (x_vals - mid_x) ** 2 + y_tip
    ax.plot(x_vals, y_vals, color=color, linewidth=lw, solid_capstyle='round')

def sample_feature_waterfall(target_mid, target_property="formation_energy"):
    json_path = "res/shap_hierarchical_importance_samples.json"
    group_order = ["xafs", "xanes", "xrd", "exafs", "global"]
    group_key_map = {
        "xafs": "Signal",
        "xanes": "XANES",
        "xrd": "XRD",
        "exafs": "EXAFS",
        "global": "Addition"
    }
    color_palette = ["#E39C63", "#F7E0CF", "#58A8D7", "#BAD2E1", "#A9C37F"]

    with open(json_path, "r") as f:
        data = json.load(f)

    group_importance = None
    for entry in data.get(target_property, []):
        if entry.get("mid") == target_mid:
            group_importance = entry.get("group_importance")
            break

    if group_importance is None:
        raise ValueError(f"未找到 mid={target_mid} 在 {target_property} 中的数据")

    values = []
    labels = []
    for key in group_order:
        if key in group_importance:
            values.append(group_importance[key])
            labels.append(group_key_map[key])
        else:
            raise ValueError(f"缺少 group key: {key}")

    total = sum(values)
    left = np.cumsum([0] + values[:-1])
    percent = [v / total * 100 for v in values]

    bar_height = 0.07                  
    bar_bottom = -bar_height / 2       

    tip_y_list = []
    for i in range(len(values)):
        if i % 2 == 0:
            tip_y_list.append(-0.12)      
        else:
            tip_y_list.append(-0.18)     

    text_offset = 0.03                

    fig, ax = plt.subplots(figsize=(10, 1.8))

    ax.barh(y=0, width=values, left=left, height=bar_height,
            color=color_palette, edgecolor='white', linewidth=1.5)

    n = len(values)
    for idx, (val, label, left_pos, pct, tip_y) in enumerate(zip(values, labels, left, percent, tip_y_list)):
        right_pos = left_pos + val
        mid_x = (left_pos + right_pos) / 2
        
        if idx == n - 1:
            offset_x = 3.5       
        else:
            offset_x = 0
        
        draw_curly_brace(ax, left_pos, right_pos, bar_bottom, tip_y, color='black', lw=0.8)
        text_y = tip_y - text_offset
        ax.text(mid_x + offset_x, text_y, f"{label}: {pct:.2f}%",
                ha='center', va='top', fontsize=10, color='black')

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    min_y = min(tip_y_list) - text_offset - 0.05
    ax.set_ylim(min_y, 0.2)

    plt.tight_layout()
    plt.savefig(f"figures/{target_mid}_sample_feature_{target_property}.png", bbox_inches='tight')
    plt.close()   


import json
import os
import numpy as np
import matplotlib.pyplot as plt

def sample_group_feature(target_mid, group_path=None, titles=None):
    json_path = "res/shap_hierarchical_importance_samples.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    properties = ['formation_energy', 'fermi_energy', 'band_gap']
    xprop = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    group_order = ["xafs", "xanes", "xrd", "exafs", "global"]
    group_labels = ["Signal", "XANES", "XRD", "EXAFS", "Addition"]
    color_palette = ["#F7E0CF", "#E39C63", "#58A8D7", "#BAD2E1", "#A9C37F"]

    width = 0.15

    def get_percentages(mid):
        prop_percentages = {}
        for prop in properties:
            group_importance = None
            for entry in data.get(prop, []):
                if entry.get("mid") == mid:
                    group_importance = entry.get("group_importance")
                    break
            if group_importance is None:
                raise ValueError(f"未找到 mid={mid} 在 {prop} 中的数据")
            values = [group_importance.get(key, 0) for key in group_order]
            total = sum(values)
            if total == 0:
                raise ValueError(f"{prop} 的总贡献为 0，无法计算百分比")
            percentages = [v / total * 100 for v in values]
            prop_percentages[prop] = percentages
        return prop_percentages

    os.makedirs("figures", exist_ok=True)

    if isinstance(target_mid, str):
        prop_percentages = get_percentages(target_mid)
        x = np.arange(len(properties))  

        fig, ax = plt.subplots(figsize=(5, 3))

        for i, (group_label, color) in enumerate(zip(group_labels, color_palette)):
            heights = [prop_percentages[prop][i] for prop in properties]
            offset = (i - 2) * width
            bars = ax.bar(x + offset, heights, width, label=group_label, color=color)
            for bar, h in zip(bars, heights):
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                            f'{h:.1f}', ha='center', va='bottom', fontsize=6)

        ax.set_xticks(x)
        ax.set_xticklabels(xprop, fontsize=9)
        ax.set_ylabel('Feature Contribution (%)', fontsize=9)
        ax.legend(title='', loc='upper right', fontsize=6)
        ax.grid(False)

        max_h = max(max(prop_percentages[p]) for p in properties)
        ax.set_ylim(0, max_h * 1.2 if max_h > 0 else 10)

        plt.tight_layout()
        save_path = f"figures/{target_mid}_group_bar.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"group bar图已保存至: {save_path}")

    elif isinstance(target_mid, list):
        mids = target_mid
        if len(mids) != 3:
            raise ValueError("多样品模式要求恰好3个样品（代表三个相），当前样品数为 {}".format(len(mids)))

        all_percentages = {mid: get_percentages(mid) for mid in mids}
        sample_labels = [titles.get(mid, mid) for mid in mids] if titles else mids
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
        axes = axes.flatten()
        global_max_h = 0
        for prop_percentages in all_percentages.values():
            for prop in properties:
                max_h = max(prop_percentages[prop])
                if max_h > global_max_h:
                    global_max_h = max_h
        ylim_max = global_max_h * 1.2 if global_max_h > 0 else 10
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            x = np.arange(len(mids))  

            for i, (group_label, color) in enumerate(zip(group_labels, color_palette)):
                heights = [all_percentages[mid][prop][i] for mid in mids]
                offset = (i - 2) * width  
                bars = ax.bar(x + offset, heights, width, label=group_label, color=color)
                for bar, h in zip(bars, heights):
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., h + 0.5,
                                f'{h:.1f}', ha='center', va='bottom', fontsize=9.5)

            ax.set_xticks(x)
            ax.set_xticklabels(sample_labels, fontsize=12)
            ax.set_ylim(0, ylim_max)
            ax.grid(False)
            ax.set_title(xprop[idx], fontsize=13)  
            if idx == 0:
                ax.set_ylabel('Feature Contribution (%)', fontsize=12)
                ax.legend(title='', loc='upper right', fontsize=9)
            else:
                ax.set_ylabel('')
                ax.tick_params(axis='y', labelleft=False)

        plt.tight_layout()
        save_path = group_path if group_path else "figures/samples_group_bar.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    else:
        raise TypeError("target_mid 必须是字符串或列表")


def plot_xrd_(json_file: str, save_fig: str = None):
    with open(json_file, "r") as f:
        data = json.load(f)
    two_theta = data["all_peaks"]["two_theta"]
    intensity = data["all_peaks"]["intensity"]

    plt.figure(figsize=(12, 6))
    plt.vlines(two_theta, 0, intensity, colors='b', linewidth=1.5, alpha=0.8)
    plt.xlabel("2θ (Degrees)", fontsize=14)
    plt.ylabel("Intensity", fontsize=14)
    plt.xlim(min(two_theta)-1, max(two_theta)+1)
    plt.ylim(0, max(intensity)*1.05)

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')

def plot_xafs_(xafs_json_path, element, save_path=None, dpi=300):
    with open(xafs_json_path, 'r') as f:
        xafs_data = json.load(f)
    
    pb_data = xafs_data[f'{element}_K_XAFS']
    energy = np.array(pb_data['energy'])
    intensity = np.array(pb_data['intensity'])
    
    diff_intensity = np.diff(intensity)
    diff_energy = np.diff(energy)
    derivative = diff_intensity / diff_energy
    from scipy.ndimage import gaussian_filter1d
    derivative_smooth = gaussian_filter1d(derivative, sigma=2)
    edge_index = np.argmax(derivative_smooth)
    edge_energy = energy[edge_index]  
    
    plt.figure(figsize=(7, 3))
    plt.plot(energy, intensity, color='#58A8D7', linewidth=2.0)
    plt.axvline(x=edge_energy, color='red', linestyle='--', linewidth=1.5,
                label=f'{element} K-edge ~ {edge_energy:.1f} eV')
    
    plt.xlabel('Energy (eV)', fontsize=10)
    plt.ylabel('Intensity μ(E)', fontsize=10)
    plt.legend(fontsize=8)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=dpi)
    plt.close()
    print(f"XAFS Pb 图已保存至: {save_path}")


from scipy.ndimage import gaussian_filter1d

def plot_xrd_pattern(json_file: str, save_fig: str = None):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    all_peaks = data.get('all_peaks', {})
    two_theta_raw = np.array(all_peaks.get('two_theta', []), dtype=float)
    intensity_raw = np.array(all_peaks.get('intensity', []), dtype=float)
    d_spacing_raw = np.array(all_peaks.get('d_spacing', []), dtype=float)
    
    if len(two_theta_raw) == 0 or len(intensity_raw) == 0:
        raise ValueError("JSON文件中缺少有效的峰数据")
    
    mask = two_theta_raw <= 82.0
    two_theta = two_theta_raw[mask]
    intensity = intensity_raw[mask]
    d_spacing = d_spacing_raw[mask] if len(d_spacing_raw) == len(two_theta_raw) else None
    
    maxseq = 2000
    theta_min, theta_max = 8.0, 82.0
    two_theta_range = np.linspace(theta_min, theta_max, maxseq)
    reconstructed = np.zeros_like(two_theta_range)
    
    for tth, intens, d_val in zip(two_theta, intensity,
                                  d_spacing if d_spacing is not None else [None]*len(two_theta)):
        if intens <= 0.01:
            continue
        if d_val is not None and d_val > 0:
            fwhm = 0.1 + 0.3 * (5.0 / (d_val + 1.0))
        else:
            fwhm = 0.2
        sigma = fwhm / 2.355
        gaussian = intens * np.exp(-(two_theta_range - tth)**2 / (2 * sigma**2))
        reconstructed += gaussian
    
    reconstructed = gaussian_filter1d(reconstructed, sigma=1.0)
    
    top10_indices = np.argsort(intensity)[-10:][::-1]
    top_theta = two_theta[top10_indices]
    top_intensity_on_curve = np.interp(top_theta, two_theta_range, reconstructed)
    
    plt.figure(figsize=(7, 3))
    plt.plot(two_theta_range, reconstructed, color='#1C3885', linewidth=2.0)
    plt.scatter(top_theta, top_intensity_on_curve, color='#4F8CBB', s=60, zorder=5, edgecolors='none')
    
    plt.xlim(theta_min, theta_max)
    plt.ylim(-5, 105)
    plt.yticks(np.arange(0, 101, 20))
    plt.xticks(np.arange(10, 81, 10))
    
    plt.xlabel('2θ (degree)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    plt.text(0.95, 0.75, 'XRD', transform=plt.gca().transAxes,
             fontsize=14, ha='right', va='top')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_fig, dpi=300, bbox_inches='tight')
    print(f"图片已保存至: {save_fig}")
    plt.close()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_absorption_spectra(xanes_json_path, xafs_json_path, element='Sn',
                            save_path=None, dpi=300):
    with open(xanes_json_path, 'r') as f:
        xanes_data = json.load(f)
    xanes_key = f'{element}_K_XANES'
    if xanes_key not in xanes_data:
        raise KeyError(f"XANES 文件中未找到键 {xanes_key}")
    xanes = xanes_data[xanes_key]
    energy_xanes = np.array(xanes['energy'])
    intensity_xanes = np.array(xanes['intensity'])
    edge_energy = xanes.get('edge_energy_eV', None)

    with open(xafs_json_path, 'r') as f:
        main_data = json.load(f)

    main_key = f'{element}_K_XAFS'
    if main_key not in main_data:
        raise KeyError(f"主数据文件中未找到键 {main_key}")

    main_spectrum = main_data[main_key]
    energy_main = np.array(main_spectrum['energy'])
    intensity_main = np.array(main_spectrum['intensity'])

    diff_intensity = np.diff(intensity_xanes)
    diff_energy = np.diff(energy_xanes)
    derivative = diff_intensity / diff_energy
    derivative_smooth = gaussian_filter1d(derivative, sigma=2)
    edge_index = np.argmax(derivative_smooth)
    edge_energy = energy_xanes[edge_index]
    print(f"自动检测吸收边: {edge_energy:.2f} eV")

    intensity_main_norm = intensity_main / np.max(intensity_main)
    intensity_xanes_norm = intensity_xanes / np.max(intensity_xanes)

    fig, ax_main = plt.subplots(figsize=(7, 3))
    ax_main.plot(energy_main, intensity_main_norm, color='#1C3885', linewidth=2.0)
    ax_main.set_xlabel('Energy (eV)', fontsize=12)
    ax_main.set_ylabel('Intensity μ(E)', fontsize=12)

    ax_main.set_ylim(0, 1.2)
    ax_main.set_yticks(np.arange(0, 1.1, 0.2))
    ax_main.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)])
    ax_main.text(0.95, 0.92, 'XAFS',
                 transform=ax_main.transAxes,
                 fontsize=14,
                 verticalalignment='top', horizontalalignment='right')

    ax_main.axvline(x=edge_energy, color='#4F8CBB', linestyle='--', linewidth=1, alpha=0.8)
    ax_main.text(edge_energy + 5, ax_main.get_ylim()[1] * 0.9,
                 f'E₀ = {edge_energy:.1f} eV', fontsize=9, color='black')

    ax_inset = inset_axes(ax_main, width="50%", height="30%",
                          loc='lower right', borderpad=3.0)
    ax_inset.plot(energy_xanes, intensity_xanes_norm, color='#1C3885', linewidth=2.0)

    ax_inset.set_ylim(0, 1.2)
    ax_inset.set_yticks(np.arange(0, 1.1, 0.2))
    ax_inset.set_yticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)])

    ax_inset.set_xlabel('Energy (eV)', fontsize=8)
    ax_inset.set_ylabel('μ(E)', fontsize=8)
    ax_inset.tick_params(axis='both', labelsize=7)
    ax_inset.set_title('XANES', fontsize=14, pad=2)

    plt.subplots_adjust(bottom=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"吸收谱图已保存至: {save_path}")
    plt.close()

def plot_separate_multi_task_heatmaps(json_path, output_dir, prefix):
    with open(json_path, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    xrd_imp_dict = data.get('xrd_features', {}).get('importance', {})
    xafs_imp_dict = data.get('xafs_fused_features', {}).get('importance', {})

    if not xrd_imp_dict and not xafs_imp_dict:
        print("警告：未找到 xrd_features 或 xafs_fused_features 的 importance 数据，跳过热力图")
        return

    tasks = ['formation_energy', 'fermi_energy', 'band_gap']
    task_labels = {
        'formation_energy': f'Formation\nEnergy',
        'fermi_energy': f'Fermi\nEnergy',
        'band_gap': f'Bandgap\nEnergy'
    }
    
    xrd_data = {}
    xafs_data = {}
    for task in tasks:
        if task in xrd_imp_dict:
            xrd_arr = np.array(xrd_imp_dict[task])
            xrd_data[task] = xrd_arr
        if task in xafs_imp_dict:
            xafs_arr = np.array(xafs_imp_dict[task])
            xafs_data[task] = xafs_arr
    
    vmin = -0.002
    vmax = 0.022
    cbar_ticks = [0, 0.005, 0.01, 0.015, 0.02]   
    
    custom_colors = ['#DBF1FA', '#C6DEED', '#9FC0D6', '#58A8D7', '#3F719D']
    cmap = LinearSegmentedColormap.from_list('custom_importance', custom_colors, N=256)
    
    if xrd_data:
        fig_xrd, axes_xrd = plt.subplots(3, 1, figsize=(8, 2.5))
        fig_xrd.subplots_adjust(hspace=0.1)
        
        im_xrd = None
        for i, task in enumerate(tasks):
            ax = axes_xrd[i]
            if task in xrd_data:
                arr = xrd_data[task].reshape(1, -1)
                im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                if i == 0:
                    im_xrd = im
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_yticks([])
            ax.set_xticks([])
            if i == 2:
                ax.set_xlabel('Feature Index', fontsize=12)
            ax.set_ylabel(task_labels[task], rotation=0, ha='right', va='center', labelpad=5, fontsize=12)
        
        if im_xrd is not None:
            cbar = fig_xrd.colorbar(im_xrd, ax=axes_xrd, location='right', pad=0.05, aspect=40)
            cbar.set_label('Importance', fontsize=12)
            cbar.set_ticks(cbar_ticks)                   
            cbar.set_ticklabels([f'{tick:.3f}' for tick in cbar_ticks]) 
            cbar.ax.tick_params(labelsize=10)
        
        save_path_xrd = os.path.join(output_dir, f'{prefix}_xrd_feature_tasks_importance.png')
        plt.savefig(save_path_xrd, dpi=300, bbox_inches='tight')
        plt.close(fig_xrd)
        print(f"XRD 热力图已保存至: {save_path_xrd}")
    else:
        print("警告：未找到 XRD 特征重要性数据，跳过 XRD 图片")
    
    if xafs_data:
        fig_xafs, axes_xafs = plt.subplots(3, 1, figsize=(8, 2.5))
        fig_xafs.subplots_adjust(hspace=0.1)
        
        im_xafs = None
        for i, task in enumerate(tasks):
            ax = axes_xafs[i]
            if task in xafs_data:
                arr = xafs_data[task].reshape(1, -1)
                im = ax.imshow(arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                if i == 0:
                    im_xafs = im
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            
            ax.set_yticks([])
            ax.set_xticks([])
            if i == 2:
                ax.set_xlabel('Feature Index', fontsize=12)
            ax.set_ylabel(task_labels[task], rotation=0, ha='right', va='center', labelpad=5, fontsize=12)
        
        if im_xafs is not None:
            cbar = fig_xafs.colorbar(im_xafs, ax=axes_xafs, location='right', pad=0.05, aspect=40)
            cbar.set_label('Importance', fontsize=12)
            cbar.set_ticks(cbar_ticks)
            cbar.set_ticklabels([f'{tick:.3f}' for tick in cbar_ticks])
            cbar.ax.tick_params(labelsize=10)
        
        save_path_xafs = os.path.join(output_dir, f'{prefix}_xafs_feature_tasks_importance.png')
        plt.savefig(save_path_xafs, dpi=300, bbox_inches='tight')
        plt.close(fig_xafs)
        print(f"XAFS 热力图已保存至: {save_path_xafs}")
    else:
        print("警告：未找到 XAFS 特征重要性数据，跳过 XAFS 图片")


def plot_feature_heatmap(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    xrd_imp = data.get('xrd_features', {}).get('importance', [])
    xafs_imp = data.get('xafs_fused_features', {}).get('importance', [])

    if not xrd_imp and not xafs_imp:
        print("警告：未找到 xrd_features 或 xafs_fused_features 的 importance 数据，跳过热力图")
        return

    custom_colors = ['#DBF1FA', '#C6DEED', '#9FC0D6', '#58A8D7', '#3F719D']  
    cmap = LinearSegmentedColormap.from_list('custom_importance', custom_colors, N=256)

    xrd_arr = np.array(xrd_imp).reshape(1, -1) if xrd_imp else np.array([])
    xafs_arr = np.array(xafs_imp).reshape(1, -1) if xafs_imp else np.array([])

    vmin = np.inf
    vmax = -np.inf
    for arr in [xrd_arr, xafs_arr]:
        if arr.size > 0:
            vmin = min(vmin, arr.min())
            vmax = max(vmax, arr.max())
    if vmin == np.inf:
        vmin, vmax = 0, 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3), gridspec_kw={'width_ratios': [1, 1]})

    im1 = ax1.imshow(xrd_arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    cbar1 = fig.colorbar(im1, ax=ax1, orientation='vertical', pad=0.05, aspect=40)
    cbar1.set_label('Importance', fontsize=10)
    cbar1.ax.set_ylim(vmin, vmax)

    ax1.set_xlabel('Feature Index')
    ax1.set_ylabel('')
    ax1.set_yticks([])

    im2 = ax2.imshow(xafs_arr, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    cbar2 = fig.colorbar(im2, ax=ax2, orientation='vertical', pad=0.05, aspect=40)
    cbar2.set_label('Importance', fontsize=10)
    cbar2.ax.set_ylim(vmin, vmax)

    ax2.set_xlabel('Feature Index')
    ax2.set_ylabel('')
    ax2.set_yticks([])

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'feature_importance_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"热力图已保存至: {save_path}")


def plot_feature_data(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    os.makedirs(output_dir, exist_ok=True)

    def plot_curve(y_values, ylabel, filename, xlabel='Feature Index'):

        y = np.array(y_values)
        x = np.arange(len(y))

        plt.figure(figsize=(7, 3))
        plt.plot(x, y, 'b-', linewidth=1.5)

        if len(y) > 0:
            top5_indices = np.argsort(y)[-5:][::-1]  
            for idx in top5_indices:
                plt.axvline(x=idx, color='r', linestyle='--', alpha=0.6, linewidth=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(False)
        plt.tight_layout()

        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存: {save_path}")

    xrd = data.get('xrd_features', {})
    if xrd:
        plot_curve(xrd.get('values', []),
                   ylabel='Feature Value',
                   filename='xrd_features_values.png')
        plot_curve(xrd.get('importance', []),
                   ylabel='Importance',
                   filename='xrd_features_importance.png')
    else:
        print("未找到 xrd_features 数据")

    xafs_fused = data.get('xafs_fused_features', {})
    if xafs_fused:
        plot_curve(xafs_fused.get('values', []),
                   ylabel='Feature Value',
                   filename='xafs_fused_features_values.png')
        plot_curve(xafs_fused.get('importance', []),
                   ylabel='Importance',
                   filename='xafs_fused_features_importance.png')
    else:
        print("未找到 xafs_fused_features 数据")

    fused = data.get('fused_features', {})
    if fused:
        plot_curve(fused.get('values', []),
                   ylabel='Feature Value',
                   filename='fused_features_values.png')
        plot_curve(fused.get('importance', []),
                   ylabel='Importance',
                   filename='fused_features_importance.png')
    else:
        print("未找到 fused_features 数据")

    shap_dict = data.get('multi_task_fused_shap', {})
    if shap_dict:
        task_name_mapping = {
            'formation_energy': 'Formation Energy',
            'fermi_energy': 'Fermi Energy',
            'band_gap': 'Band Gap'
        }
        for task_key, task_display in task_name_mapping.items():
            shap_values = shap_dict.get(task_key)
            if shap_values is not None:
                plot_curve(shap_values,
                           ylabel='Normalized Contribution',
                           filename=f'{task_key}_fused_shap.png')
            else:
                print(f"未找到任务 {task_key} 的 fused_shap 数据")
    else:
        print("未找到 multi_task_fused_shap 数据")

    group_outputs = data.get('xafs_group_outputs', {})
    if group_outputs:
        group_names = {
            'xafs_encoder_output': 'XAFS Encoder (full)',
            'xanes_encoder_output': 'XANES Encoder',
            'exafs_encoder_output': 'EXAFS Encoder'
        }
        for key, display_name in group_names.items():
            values = group_outputs.get(key)
            if values is not None:
                plot_curve(values,
                           ylabel='Feature Value',
                           filename=f'{key}.png')
            else:
                print(f"未找到 {key} 数据")
    else:
        print("未找到 xafs_group_outputs 数据")

    cross_attn = data.get('cross_attn_outputs', {})
    if cross_attn:
        if 'xrd2xafs' in cross_attn:
            plot_curve(cross_attn['xrd2xafs'],
                       ylabel='Feature Value',
                       filename='cross_attn_xrd2xafs.png')
        else:
            print("未找到 cross_attn_outputs 中的 xrd2xafs 数据")

        if 'xafs2xrd' in cross_attn:
            plot_curve(cross_attn['xafs2xrd'],
                       ylabel='Feature Value',
                       filename='cross_attn_xafs2xrd.png')
        else:
            print("未找到 cross_attn_outputs 中的 xafs2xrd 数据")
    else:
        print("未找到 cross_attn_outputs 数据")

    print(f"\n所有曲线图已保存至: {output_dir}")

from scipy.stats import wilcoxon
import seaborn as sns
from matplotlib.ticker import FixedLocator, FixedFormatter

def ab_fusion_statistic_plot():
    
    data_raw = {
        "w/ Concat": [
            [0.2796, 0.5663, 0.2935, 0.3787, 0.2735, 0.3811],
            [0.2402, 0.6098, 0.2711, 0.4871, 0.2795, 0.3377],
            [0.2332, 0.6255, 0.2531, 0.5503, 0.2841, 0.3689],
            [0.2629, 0.591, 0.2594, 0.502, 0.2633, 0.3665],
            [0.2353, 0.6308, 0.2649, 0.4866, 0.2578, 0.3666],
            [0.2304, 0.6417, 0.2657, 0.5057, 0.2628, 0.3788],
            [0.2419, 0.5984, 0.252, 0.5302, 0.2837, 0.3595],
            [0.2687, 0.5792, 0.263, 0.5312, 0.2627, 0.3475],
            [0.2593, 0.6132, 0.2829, 0.4206, 0.2808, 0.3621],
            [0.2409, 0.6097, 0.2829, 0.4206, 0.2728, 0.3538],
        ],
        "w/o PFA": [
            [0.25, 0.5721, 0.2703, 0.4757, 0.2656, 0.3175],
            [0.2273, 0.6359, 0.2593, 0.4963, 0.2578, 0.3842],
            [0.2544, 0.5533, 0.2699, 0.4862, 0.2665, 0.3277],
            [0.2436, 0.5693, 0.2687, 0.4834, 0.2724, 0.3343],
            [0.237, 0.599, 0.2673, 0.4833, 0.2694, 0.3283],
            [0.2499, 0.557, 0.2725, 0.4685, 0.2717, 0.3755],
            [0.2295, 0.6063, 0.2634, 0.4959, 0.2713, 0.3111],
            [0.2542, 0.5184, 0.275, 0.4695, 0.2748, 0.3245],
            [0.2262, 0.6381, 0.2584, 0.5169, 0.2518, 0.3751],
            [0.241, 0.5901, 0.2616, 0.4945, 0.2612, 0.3588],  
        ],
        "w/o IAE": [
            [0.2053, 0.7114, 0.2481, 0.5365, 0.2333, 0.4504],
            [0.2266, 0.6207, 0.2516, 0.532, 0.2477, 0.4472],
            [0.2361, 0.6323, 0.2556, 0.5105, 0.244, 0.4412],
            [0.2163, 0.687, 0.2424, 0.5202, 0.257, 0.3958],
            [0.2385, 0.6047, 0.2588, 0.5123, 0.2513, 0.4087],
            [0.2213, 0.6363, 0.2559, 0.5206, 0.2587, 0.3777],
            [0.2146, 0.6728, 0.2481, 0.5318, 0.2413, 0.406],
            [0.2102, 0.6845, 0.2492, 0.5384, 0.2585, 0.3878],
            [0.2134, 0.6793, 0.2552, 0.5305, 0.244, 0.4558],
            [0.2125, 0.675, 0.2453, 0.536, 0.2447, 0.436], 
        ],
        "w/o BCF": [
            [0.2309, 0.7079, 0.2358, 0.567, 0.239, 0.4477],
            [0.2243, 0.6561, 0.2526, 0.5497, 0.2385, 0.4308],
            [0.2297, 0.6591, 0.2575, 0.5334, 0.2517, 0.3963],
            [0.214, 0.6795, 0.2473, 0.554, 0.2375, 0.4781],
            [0.2315, 0.6903, 0.2343, 0.5715, 0.2328, 0.4874],
            [0.2256, 0.6671, 0.2433, 0.5729, 0.2408, 0.4598],
            [0.2185, 0.6847, 0.2383, 0.5696, 0.2457, 0.4077],
            [0.214, 0.6859, 0.2388, 0.5769, 0.2433, 0.428],
            [0.2152, 0.6837, 0.2509, 0.5243, 0.2491, 0.425],
            [0.2114, 0.6838, 0.2365, 0.5951, 0.2434, 0.4489],
        ],
        "InterLLC": [
            [0.2037, 0.7133, 0.246, 0.5816, 0.2229, 0.5163],
            [0.2197, 0.6531, 0.2202, 0.5742, 0.2433, 0.4877],
            [0.2143, 0.6759, 0.2362, 0.5503, 0.2357, 0.465],
            [0.2037, 0.6868, 0.2431, 0.5485, 0.2274, 0.4761],
            [0.2095, 0.6947, 0.2277, 0.5673, 0.2333, 0.5005],
            [0.2034, 0.6968, 0.2148, 0.5783, 0.2394, 0.4653],
            [0.2183, 0.6355, 0.234, 0.5553, 0.2317, 0.4429],
            [0.2022, 0.6474, 0.2383, 0.5639, 0.2207, 0.4645],
            [0.2154, 0.6498, 0.223, 0.5639, 0.2192, 0.4831],
            [0.217, 0.668, 0.2157, 0.5559, 0.2393, 0.4748],
        ],
    }

    properties = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    metrics = ['MAE', 'R2']

    model_order = ['w/ Concat', 'w/o PFA', 'w/o IAE', 'w/o BCF', 'InterLLC']
    model_labels = ['w/ Concat', 'w/o PFA', 'w/o IAE', 'w/o BCF', 'InterLLC']
    custom_colors = ['#A9C37F', '#BAD2E1', '#F7E0CF', '#D1E4CF', '#E39C63']

    records = []
    for model, rows in data_raw.items():
        for rep_idx, row in enumerate(rows):
            records.append([model, rep_idx, properties[0], 'MAE', row[0]])
            records.append([model, rep_idx, properties[0], 'R2', row[1]])
            records.append([model, rep_idx, properties[1], 'MAE', row[2]])
            records.append([model, rep_idx, properties[1], 'R2', row[3]])
            records.append([model, rep_idx, properties[2], 'MAE', row[4]])
            records.append([model, rep_idx, properties[2], 'R2', row[5]])

    df = pd.DataFrame(records, columns=['Model', 'Rep', 'Property', 'Metric', 'Value'])

    for prop in properties:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(5, 4.5), sharex=True)
        
        data_mae = df[(df['Property'] == prop) & (df['Metric'] == 'MAE')]
        sns.boxplot(x='Model', y='Value', hue='Model', data=data_mae, 
                    order=model_order, palette=custom_colors, width=0.4, 
                    ax=ax_top, legend=False)  
        ax_top.set_title(prop, fontsize=12)
        ax_top.set_ylabel('MAE (eV)', fontsize=10)
        ax_top.set_ylim(0.18, 0.32)
        ax_top.set_xlabel('')        
        ax_top.tick_params(axis='x', labelbottom=False)  
        
        data_r2 = df[(df['Property'] == prop) & (df['Metric'] == 'R2')]
        sns.boxplot(x='Model', y='Value', hue='Model', data=data_r2, 
                    order=model_order, palette=custom_colors, width=0.4, 
                    ax=ax_bottom, legend=False)  
        ax_bottom.set_ylabel('R²', fontsize=10)
        ax_bottom.set_ylim(0.2, 0.8)
        ax_bottom.set_xlabel('')     
        
        ax_bottom.xaxis.set_major_locator(FixedLocator(range(len(model_labels))))
        ax_bottom.xaxis.set_major_formatter(FixedFormatter(model_labels))
        plt.setp(ax_bottom.xaxis.get_majorticklabels(), fontsize=10, rotation=45, ha='right')
        
        plt.tight_layout()
        filename = f'figures/{prop.replace(" ", "_")}_boxplots.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    full_model = 'InterLLC'
    variants = [m for m in model_order if m != full_model]
    pval_list = []

    for prop in properties:
        for metric in metrics:
            full_vals = df[(df['Model'] == full_model) & (df['Property'] == prop) & (df['Metric'] == metric)].sort_values('Rep')['Value'].values
            for var in variants:
                var_vals = df[(df['Model'] == var) & (df['Property'] == prop) & (df['Metric'] == metric)].sort_values('Rep')['Value'].values
                
                if np.array_equal(var_vals, full_vals):
                    p = 1.0
                else:
                    try:
                        if metric == 'MAE':
                            _, p = wilcoxon(var_vals, full_vals, alternative='greater')
                        else:
                            _, p = wilcoxon(full_vals, var_vals, alternative='greater')
                    except ValueError as e:
                        print(f"Warning: Wilcoxon test failed for {prop} - {var} - {metric}: {e}")
                        p = 1.0
                
                pval_list.append([prop, metric, var, p])

    df_pvals = pd.DataFrame(pval_list, columns=['Property', 'Metric', 'Variant', 'p_value'])
    print(f"Total comparisons: {len(df_pvals)}")

    for prop in properties:
        metric = 'MAE'
        plt.figure(figsize=(4, 2))
        
        sub = df_pvals[(df_pvals['Property'] == prop) & (df_pvals['Metric'] == metric)]
        sub = sub.set_index('Variant').reindex(variants).reset_index()
        x = np.arange(len(variants))
        p_vals = sub['p_value'].values
        
        bars = plt.bar(x, p_vals, color='#BAD2E1', edgecolor='black', width=0.6)
        
        plt.axhline(y=0.05, color='gray', linestyle=':', linewidth=0.8, label='α = 0.05')
        plt.axhline(y=0.01, color='#58A8D7', linestyle='--', linewidth=1.2, label='p = 0.01')
        plt.axhline(y=0.005, color='#E39C63', linestyle='--', linewidth=1.2, label='p = 0.005')
        
        for i, (bar, p) in enumerate(zip(bars, p_vals)):
            if not np.isnan(p):
                if p < 0.005:
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, '***',
                             ha='center', va='bottom', fontsize=9, color='red')
                elif p < 0.01:
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, '**',
                             ha='center', va='bottom', fontsize=9, color='red')
                elif p < 0.05:
                    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002, '*',
                             ha='center', va='bottom', fontsize=9, color='red')
        
        plt.ylabel('p-value', fontsize=9)
        plt.title(prop, fontsize=10)
        plt.xticks(x, variants, fontsize=9)
        plt.ylim(0, 0.08)
        plt.legend(fontsize=7, loc='upper right')
        plt.tight_layout()
        
        filename = f'figures/{prop.replace(" ", "_")}_MAE_pvalue_bars.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    print("\n=== Paired Wilcoxon p-values (MAE only) ===")
    print(df_pvals[df_pvals['Metric'] == 'MAE'].to_string(index=False))


import ast
from matplotlib.colors import LinearSegmentedColormap

def plot_periodic_table_heatmap(element_percentages, site_name, output_filename):
    periodic_layout = {
        (1, 1): "H", (1, 18): "He",
        (2, 1): "Li", (2, 2): "Be",
        (2, 13): "B", (2, 14): "C", (2, 15): "N", (2, 16): "O", (2, 17): "F", (2, 18): "Ne",
        (3, 1): "Na", (3, 2): "Mg",
        (3, 13): "Al", (3, 14): "Si", (3, 15): "P", (3, 16): "S", (3, 17): "Cl", (3, 18): "Ar",
        (4, 1): "K", (4, 2): "Ca",
        (4, 3): "Sc", (4, 4): "Ti", (4, 5): "V", (4, 6): "Cr", (4, 7): "Mn",
        (4, 8): "Fe", (4, 9): "Co", (4, 10): "Ni", (4, 11): "Cu", (4, 12): "Zn",
        (4, 13): "Ga", (4, 14): "Ge", (4, 15): "As", (4, 16): "Se", (4, 17): "Br", (4, 18): "Kr",
        (5, 1): "Rb", (5, 2): "Sr",
        (5, 3): "Y", (5, 4): "Zr", (5, 5): "Nb", (5, 6): "Mo", (5, 7): "Tc",
        (5, 8): "Ru", (5, 9): "Rh", (5, 10): "Pd", (5, 11): "Ag", (5, 12): "Cd",
        (5, 13): "In", (5, 14): "Sn", (5, 15): "Sb", (5, 16): "Te", (5, 17): "I", (5, 18): "Xe",
        (6, 1): "Cs", (6, 2): "Ba",
        (6, 3): "La",
        (6, 4): "Hf", (6, 5): "Ta", (6, 6): "W", (6, 7): "Re", (6, 8): "Os",
        (6, 9): "Ir", (6, 10): "Pt", (6, 11): "Au", (6, 12): "Hg",
        (6, 13): "Tl", (6, 14): "Pb", (6, 15): "Bi", (6, 16): "Po", (6, 17): "At", (6, 18): "Rn",
        (7, 1): "Fr", (7, 2): "Ra",
        (7, 3): "Ac",
        (7, 4): "Rf", (7, 5): "Db", (7, 6): "Sg", (7, 7): "Bh", (7, 8): "Hs",
        (7, 9): "Mt", (7, 10): "Ds", (7, 11): "Rg", (7, 12): "Cn",
        (7, 13): "Nh", (7, 14): "Fl", (7, 15): "Mc", (7, 16): "Lv", (7, 17): "Ts", (7, 18): "Og",
        (8, 3): "Ce", (8, 4): "Pr", (8, 5): "Nd", (8, 6): "Pm", (8, 7): "Sm",
        (8, 8): "Eu", (8, 9): "Gd", (8, 10): "Tb", (8, 11): "Dy", (8, 12): "Ho",
        (8, 13): "Er", (8, 14): "Tm", (8, 15): "Yb", (8, 16): "Lu",
        (9, 3): "Th", (9, 4): "Pa", (9, 5): "U", (9, 6): "Np", (9, 7): "Pu",
        (9, 8): "Am", (9, 9): "Cm", (9, 10): "Bk", (9, 11): "Cf", (9, 12): "Es",
        (9, 13): "Fm", (9, 14): "Md", (9, 15): "No", (9, 16): "Lr",
    }

    max_period = 10
    max_group = 19
    data_matrix = np.full((max_period, max_group), np.nan)
    label_matrix = np.full((max_period, max_group), "", dtype=object)

    for (period, group), element in periodic_layout.items():
        label_matrix[period, group] = element
        if element in element_percentages:
            val = element_percentages[element]
            if val > 0:
                data_matrix[period, group] = val

    vmin_log = np.log10(0.1) 
    vmax_log = np.log10(20)  

    log_data = np.full_like(data_matrix, np.nan)
    for i in range(max_period):
        for j in range(max_group):
            val = data_matrix[i, j]
            if not np.isnan(val) and val > 0:
                log_data[i, j] = np.log10(val)

    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    ax.set_facecolor('white') 

    colors = ['#F1F8E9', '#D4EDC8', '#66BB6A', '#2E7D32', '#004D00']
    cmap = LinearSegmentedColormap.from_list('custom_green', colors, N=256)

    im = ax.imshow(log_data, cmap=cmap, aspect='auto',
                   vmin=vmin_log, vmax=vmax_log, alpha=0.9,
                   interpolation='nearest', origin='upper')

    for period in range(1, max_period):
        for group in range(1, max_group):
            if label_matrix[period, group] != "":
                rect = plt.Rectangle((group - 0.5, period - 0.5), 1, 1,
                                     fill=False, edgecolor='white', linewidth=1.8)
                ax.add_patch(rect)

    left, right = 1.8, 12.8
    bottom, top = 1.8, 2.5   
    cax = inset_axes(ax, width="50%", height="28%",
                     bbox_to_anchor=(left, bottom, right-left, top-bottom),
                     bbox_transform=ax.transData, loc='center')
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    tick_vals = [np.log10(0.01), np.log10(0.1), np.log10(1), np.log10(5), np.log10(10), np.log10(20)]
    tick_labels = ['0.01', '0.1', '1', '5', '10', '20']
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_labels)
    cbar.set_label('Element Ratio (%)', fontsize=13)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticklabels('', fontsize=10)
    ax.set_yticklabels('', fontsize=10)
    ax.set_xlabel('', fontsize=12, fontweight='bold')
    ax.set_ylabel('', fontsize=12, fontweight='bold')

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    for period in range(1, max_period):
        for group in range(1, max_group):
            element = label_matrix[period, group]
            if element:
                val = data_matrix[period, group]
                if not np.isnan(val) and val > 0:
                    log_val = log_data[period, group]
                    norm_val = (log_val - vmin_log) / (vmax_log - vmin_log) if vmax_log > vmin_log else 0.5
                    text_color = 'white' if norm_val > 0.55 else 'black'
                    ax.text(group, period, f"{element}\n{val:.1f}%",
                            ha='center', va='center', fontsize=12,
                            color=text_color, weight='bold')
                else:
                    rect = plt.Rectangle((group-0.5, period-0.5), 1, 1, facecolor='#F5F5F5', edgecolor='white', linewidth=1.8)
                    ax.add_patch(rect)
                    ax.text(group, period, f"{element}\n-",
                            ha='center', va='center', fontsize=12,
                            color='gray', weight='normal')

    ax.tick_params(which='both', bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"已生成周期表热力图: {output_filename}")


def plot_b_site_pie(b_series, output_filename):
    counts = b_series.value_counts()
    total = len(b_series)
    percentages = (counts / total * 100).round(1)

    threshold = 1.0
    major = percentages[percentages >= threshold]
    others = percentages[percentages < threshold]
    if len(others) > 0:
        major['Others'] = others.sum()

    labels = [f"{elem}\n({p:.1f}%)" for elem, p in major.items()]
    sizes = major.values

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        startangle=90, counterclock=False,
        textprops={'fontsize': 9}, pctdistance=0.85
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

    ax.set_title('B-site Element Distribution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"已生成 B 位饼图: {output_filename}")


def generate_periodic_table(file_path):
    raw_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue

            try:
                a_list = ast.literal_eval(parts[2])
                b_list = ast.literal_eval(parts[3])
                x_list = ast.literal_eval(parts[4])

                if a_list and b_list and x_list:
                    a_element = str(a_list[0]).strip()
                    b_element = str(b_list[0]).strip()
                    x_element = str(x_list[0]).strip()
                    raw_data.append({"A": a_element, "B": b_element, "X": x_element})
            except (ValueError, SyntaxError):
                continue

    df = pd.DataFrame(raw_data)
    if df.empty:
        print("错误: 未能从文件中解析出有效数据。")
        return

    print("\n" + "=" * 70)
    print("钙钛矿全量数据集（ABX3）各晶格位点元素丰度与占比报告")
    print("=" * 70)

    for site_name, col_key in [
        ("A-site Cations", "A"),
        ("B-site Metals", "B"),
        ("X-site Anions", "X"),
    ]:
        total_site_count = len(df[col_key])
        counts_series = df[col_key].value_counts()
        print(f"✨ {site_name} (总计元素种类: {len(counts_series)} 种, 总样本数: {total_site_count})")
        print(f"   {'元素 (Element)':<15} | {'出现次数 (Count)':<15} | {'占该位点比例 (Percentage)'}")
        print("   " + "-" * 60)
        for element in counts_series.index:
            count = counts_series[element]
            percentage = (count / total_site_count) * 100
            print(f"   {element:<15} | {count:<15} | {percentage:.2f}%")
        print("-" * 70)

    all_elements = pd.concat([df['A'], df['B'], df['X']])
    total_counts = all_elements.value_counts()
    total_sites = len(df) * 3
    total_percentages = (total_counts / total_sites * 100).to_dict()

    print("\n" + "=" * 70)
    print("生成综合周期表热力图（所有位点 A+B+X）...")
    plot_periodic_table_heatmap(
        total_percentages,
        "Overall (A+B+X)",
        "figures/periodic_table_total.png"
    )

    print("生成 B 位元素占比饼图...")
    plot_b_site_pie(df['B'], "figures/B_site_pie.png")
    print("=" * 70)





def generate_dual_pie_matrices(file_path, top_n_elements=12):
    raw_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue

            try:
                a_list = ast.literal_eval(parts[2])
                b_list = ast.literal_eval(parts[3])
                x_list = ast.literal_eval(parts[4])

                if a_list and b_list and x_list:
                    a_element = str(a_list[0]).strip()
                    b_element = str(b_list[0]).strip()
                    x_element = str(x_list[0]).strip()

                    raw_data.append({"A": a_element, "B": b_element, "X": x_element})
            except (ValueError, SyntaxError):
                continue

    df = pd.DataFrame(raw_data)
    if df.empty:
        print("Error: No valid data parsed from file.")
        return

    print("\n" + "=" * 70)
    print(f"钙钛矿全量数据集（ABX3）各晶格位点元素丰度与占比报告")
    print("=" * 70)

    for site_name, col_key in [
        ("A-site Cations", "A"),
        ("B-site Metals", "B"),
        ("X-site Anions", "X"),
    ]:
        total_site_count = len(df[col_key])
        counts_series = df[col_key].value_counts()
        print(f"✨ {site_name} (总计元素种类: {len(counts_series)} 种, 总样本数: {total_site_count})")
        print(f"   {'元素 (Element)':<15} | {'出现次数 (Count)':<15} | {'占该位点比例 (Percentage)'}")
        print("   " + "-" * 60)
        for element in counts_series.index:
            count = counts_series[element]
            percentage = (count / total_site_count) * 100
            print(f"   {element:<15} | {count:<15} | {percentage:.2f}%")
        print("-" * 70)

    top_a = df["A"].value_counts().head(top_n_elements).index.tolist()
    top_b = df["B"].value_counts().head(top_n_elements).index.tolist()
    top_x = df["X"].value_counts().head(top_n_elements).index.tolist()

    df_filtered = df[
        df["A"].isin(top_a) & df["B"].isin(top_b) & df["X"].isin(top_x)
    ].copy()

    a_labels = sorted(df_filtered["A"].unique())
    b_labels = sorted(df_filtered["B"].unique())
    x_types = sorted(df_filtered["X"].unique())

    ax_size = 0.045

    base_colors = ['#E39C63', '#F7E0CF', '#58A8D7', '#BAD2E1', '#A9C37F', '#D1E4CF']
    max_elements = max(len(a_labels), len(b_labels), len(x_types), 24)
    extended_palette = []

    import colorsys

    for i in range(max_elements):
        base_hex = base_colors[i % len(base_colors)]
        if i >= len(base_colors):
            r, g, b = [int(base_hex[k : k + 2], 16) / 255.0 for k in (1, 3, 5)]
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            cycle_idx = i // len(base_colors)
            if cycle_idx % 2 == 1:
                l = max(0.25, l * 0.75)
                s = min(1.0, s * 1.2)
            else:
                l = min(0.95, l * 1.15) 
                s = max(0.1, s * 0.8)

            r, g, b = colorsys.hls_to_rgb(h, l, s)
            derived_hex = f"#{int(r*255):02X}{int(g*255):02X}{int(b*255):02X}"
            extended_palette.append(derived_hex)
        else:
            extended_palette.append(base_hex)

    color_map_x = {t: extended_palette[idx] for idx, t in enumerate(x_types)}

    fig1, ax1 = plt.subplots(figsize=(8, 7), dpi=300)
    ax1.set_xlim(-0.5, len(a_labels) - 0.5)
    ax1.set_ylim(-0.5, len(b_labels) - 0.5)
    ax1.set_xticks(range(len(a_labels)))
    ax1.set_yticks(range(len(b_labels)))

    ax1.set_xticklabels(a_labels, fontsize=15, rotation=0)
    ax1.set_yticklabels(b_labels, fontsize=15)
    ax1.set_xlabel("A-site Elements", fontsize=17, labelpad=10)
    ax1.set_ylabel("B-site Elements", fontsize=17, labelpad=10)

    ax1.grid(
        True, which="both", color="#f5f5f5", linestyle="-", linewidth=0.5, zorder=1
    )

    for i, a in enumerate(a_labels):
        for j, b in enumerate(b_labels):
            cell_data = df_filtered[
                (df_filtered["A"] == a) & (df_filtered["B"] == b)
            ]
            if cell_data.empty:
                ax1.plot(
                    [i - 0.15, i + 0.15],
                    [j - 0.15, j + 0.15],
                    color="#e8e8e8",
                    linewidth=0.6,
                    zorder=2,
                )
                continue

            counts = cell_data["X"].value_counts().reindex(x_types, fill_value=0)
            if counts.sum() == 0:
                continue

            trans = ax1.transData.transform((i, j))
            trans_axes = fig1.transFigure.inverted().transform(trans)
            sub_ax = fig1.add_axes(
                [
                    trans_axes[0] - ax_size / 2,
                    trans_axes[1] - ax_size / 2,
                    ax_size,
                    ax_size,
                ]
            )

            cell_colors = [color_map_x.get(t, base_colors[0]) for t in x_types]
            sub_ax.pie(
                counts.values,
                colors=cell_colors,
                radius=1.0,
                wedgeprops=dict(edgecolor="none"),
                startangle=90,
            )
            sub_ax.axis("equal")

            ax1.text(
                i + 0.35,
                j + 0.22,
                f"{counts.sum()}",
                ha="center",
                va="center",
                fontsize=13,
                color="#888888",
                zorder=4,
            )

    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["top"].set_visible(True)
    ax1.spines["right"].set_visible(True)
    ax1.spines["left"].set_linewidth(1.0)
    ax1.spines["bottom"].set_linewidth(1.0)

    legend_el1 = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_map_x.get(t),
            edgecolor="none",
            label=f"{t}",
        )
        for t in x_types
        if (df_filtered["X"] == t).sum() > 0
    ]
    ax1.legend(
        handles=legend_el1,
        title="X-site",
        title_fontsize=14,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="#f0f0f0",
    )

    fig1.savefig("figures/perovskite_matrix_AB.png", bbox_inches="tight")
    plt.close(fig1)
    color_map_a = {t: extended_palette[idx] for idx, t in enumerate(a_labels)}

    fig2, ax2 = plt.subplots(figsize=(8, 7), dpi=300)
    ax2.set_xlim(-0.5, len(b_labels) - 0.5)
    ax2.set_ylim(-0.5, len(x_types) - 0.5)
    ax2.set_xticks(range(len(b_labels)))
    ax2.set_yticks(range(len(x_types)))

    ax2.set_xticklabels(b_labels, fontsize=15, rotation=0)
    ax2.set_yticklabels(x_types, fontsize=15)
    ax2.set_xlabel("B-site Elements", fontsize=17, labelpad=10)
    ax2.set_ylabel("X-site Elements", fontsize=17, labelpad=10)
    ax2.grid(
        True, which="both", color="#f5f5f5", linestyle="-", linewidth=0.5, zorder=1
    )

    for i, b in enumerate(b_labels):
        for j, x in enumerate(x_types):
            cell_data = df_filtered[
                (df_filtered["B"] == b) & (df_filtered["X"] == x)
            ]
            if cell_data.empty:
                ax2.plot(
                    [i - 0.15, i + 0.15],
                    [j - 0.15, j + 0.15],
                    color="#e8e8e8",
                    linewidth=0.6,
                    zorder=2,
                )
                continue

            counts = cell_data["A"].value_counts().reindex(a_labels, fill_value=0)
            if counts.sum() == 0:
                continue

            trans = ax2.transData.transform((i, j))
            trans_axes = fig2.transFigure.inverted().transform(trans)
            sub_ax = fig2.add_axes(
                [
                    trans_axes[0] - ax_size / 2,
                    trans_axes[1] - ax_size / 2,
                    ax_size,
                    ax_size,
                ]
            )

            cell_colors_a = [color_map_a.get(t, base_colors[0]) for t in a_labels]
            sub_ax.pie(
                counts.values,
                colors=cell_colors_a,
                radius=1.0,
                wedgeprops=dict(edgecolor="none"),
                startangle=90,
            )
            sub_ax.axis("equal")

            ax2.text(
                i + 0.27,
                j + 0.32,
                f"{counts.sum()}",
                ha="center",
                va="center",
                fontsize=13,
                color="#888888",
                zorder=4,
            )

    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    ax2.spines["left"].set_linewidth(1.0)
    ax2.spines["bottom"].set_linewidth(1.0)

    legend_el2 = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=color_map_a.get(t),
            edgecolor="none",
            label=f"{t}",
        )
        for t in a_labels
        if (df_filtered["A"] == t).sum() > 0
    ]
    ax2.legend(
        handles=legend_el2,
        title="A-site",
        title_fontsize=14,
        fontsize=12,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        facecolor="white",
        edgecolor="#f0f0f0",
    )

    fig2.savefig("figures/perovskite_matrix_BX.png", bbox_inches="tight")
    plt.close(fig2)



def plot_multiple_xrd(mids, json_dir='res/', save_path='figures/samples_xrd_pattern.png',
                      theta_min=8.0, theta_max=82.0, maxseq=2000, dpi=300):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    items = list(mids.items())
    if len(items) > 3:
        print(f"Warning: 传入 {len(items)} 个样品，仅绘制前三个。")
        items = items[:3]
    elif len(items) < 3:
        raise ValueError(f"需要至少三个样品，当前仅有 {len(items)} 个。")

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ['#6A5F51', '#164A3C', '#1C3885']
    two_theta_range = np.linspace(theta_min, theta_max, maxseq)

    offset_base = 120 
    reconstructed_list = []
    offsets = []        
    max_label_y = -np.inf 

    for i, (mid, label) in enumerate(items):
        json_file = os.path.join(json_dir, mid, f'{mid}_xrd.json')
        with open(json_file, 'r') as f:
            data = json.load(f)

        all_peaks = data.get('all_peaks', {})
        two_theta_raw = np.array(all_peaks.get('two_theta', []), dtype=float)
        intensity_raw = np.array(all_peaks.get('intensity', []), dtype=float)
        d_spacing_raw = np.array(all_peaks.get('d_spacing', []), dtype=float)

        mask = two_theta_raw <= theta_max
        two_theta = two_theta_raw[mask]
        intensity = intensity_raw[mask]
        d_spacing = d_spacing_raw[mask] if len(d_spacing_raw) == len(two_theta_raw) else None

        reconstructed = np.zeros_like(two_theta_range)
        for tth, intens, d_val in zip(two_theta, intensity,
                                      d_spacing if d_spacing is not None else [None]*len(two_theta)):
            if intens <= 0.01:
                continue
            if d_val is not None and d_val > 0:
                fwhm = 0.1 + 0.3 * (5.0 / (d_val + 1.0))
            else:
                fwhm = 0.2
            sigma = fwhm / 2.355
            gaussian = intens * np.exp(-(two_theta_range - tth)**2 / (2 * sigma**2))
            reconstructed += gaussian

        reconstructed = gaussian_filter1d(reconstructed, sigma=1.0)

        offset = i * offset_base
        offsets.append(offset)
        ax.plot(two_theta_range, reconstructed + offset, color=colors[i], linewidth=2.0)

        x_end = two_theta_range[-1]
        y_end = reconstructed[-1] + offset
        label_x = x_end - 0.5         
        label_y = y_end + 40          
        ax.text(label_x, label_y, label, color=colors[i], fontsize=16,
                ha='right', va='bottom')
        max_label_y = max(max_label_y, label_y)

        reconstructed_list.append(reconstructed)

    for off, col in zip(offsets, colors[:len(items)]):
        ax.hlines(y=off, xmin=theta_min, xmax=theta_max, colors=col,
                  linestyles='--', linewidth=1, alpha=0.7)

    max_intensity = max([np.max(r) for r in reconstructed_list]) if reconstructed_list else 100
    total_offset = (len(items) - 1) * offset_base
    y_upper = max(max_label_y + 20, total_offset + max_intensity + 50)
    ax.set_ylim(-5, y_upper)

    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax.set_xlabel('2θ (degree)', fontsize=16)
    ax.set_xticks(np.arange(10, 81, 10))
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 2)  

    ax.set_ylabel('Intensity (a.u.)', fontsize=16)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"XRD 叠加图已保存至: {save_path}")
    plt.close()


def plot_multiple_xafs(mids, element, json_dir='res/',
                       save_path='figures/samples_xafs_element.png', dpi=300):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    items = list(mids.items())
    if len(items) > 3:
        print(f"Warning: 传入 {len(items)} 个样品，仅绘制前三个。")
        items = items[:3]
    elif len(items) < 3:
        raise ValueError(f"需要至少三个样品，当前仅有 {len(items)} 个。")

    if isinstance(element, list):
        if len(element) < len(items):
            raise ValueError(f"element 列表长度 ({len(element)}) 小于样品数 ({len(items)})")
        elem_list = element[:len(items)]
    else:
        elem_list = [element] * len(items)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    colors = ['#6A5F51', '#164A3C', '#1C3885']

    offset_base = 1.5
    max_intensities = []
    offsets = []
    max_label_y = -np.inf
    all_energies = []

    for i, (mid, label) in enumerate(items):
        xafs_json = os.path.join(json_dir, mid, f'{mid}_xafs.json')
        with open(xafs_json, 'r') as f:
            xafs_data = json.load(f)

        elem = elem_list[i]
        key = f'{elem}_K_XAFS'
        if key not in xafs_data:
            raise KeyError(f"在 {xafs_json} 中未找到键 {key}")

        spectrum = xafs_data[key]
        energy = np.array(spectrum['energy'])
        intensity = np.array(spectrum['intensity'])
        all_energies.append(energy)

        intensity_norm = intensity / np.max(intensity)
        max_intensities.append(np.max(intensity_norm))

        offset = i * offset_base
        offsets.append(offset)
        ax.plot(energy, intensity_norm + offset, color=colors[i], linewidth=2.0)

        x_end = energy[-1]
        y_end = intensity_norm[-1] + offset
        label_x = x_end - 2
        label_y = y_end + 0.15
        ax.text(label_x, label_y, label, color=colors[i], fontsize=16,
                ha='right', va='bottom')
        max_label_y = max(max_label_y, label_y)

    xmin = min([e[0] for e in all_energies])
    xmax = max([e[-1] for e in all_energies])

    for off, col in zip(offsets, colors[:len(items)]):
        ax.hlines(y=off, xmin=xmin, xmax=xmax, colors=col,
                  linestyles='--', linewidth=1, alpha=0.7)

    max_intensity = max(max_intensities) if max_intensities else 1.0
    total_offset = (len(items) - 1) * offset_base
    y_upper = max(max_label_y + 0.3, total_offset + max_intensity + 0.5)
    ax.set_ylim(-0.2, y_upper)

    ax.tick_params(axis='y', which='both', left=False, labelleft=False)

    ax.set_xlabel('Energy (eV)', fontsize=16)
    ax.set_ylabel('Intensity μ(E)', fontsize=16)

    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"XAFS 叠加图已保存至: {save_path}")
    plt.close()


def plot_predictions(true_values, pred_values, save_path=None):
    xprop = ['Formation Energy', 'Fermi Energy', 'Bandgap']
    structs = list(true_values.keys())
    
    color_true = '#E39C63'    
    color_pred = '#58A8D7'   
    
    all_vals = []
    for prop_idx in range(len(xprop)):
        for struct in structs:
            all_vals.append(true_values[struct][prop_idx])
            all_vals.append(pred_values[struct][prop_idx])
    min_val = min(all_vals)
    max_val = max(all_vals)
    margin = (max_val - min_val) * 0.15 if max_val != min_val else 1.0
    ylim = (min_val - margin, max_val + margin)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    bar_width = 0.2         
    x = np.arange(len(structs)) 
    
    for idx, prop in enumerate(xprop):
        ax = axes[idx]
        
        true_vals = [true_values[struct][idx] for struct in structs]
        pred_vals = [pred_values[struct][idx] for struct in structs]
        
        bars_true = ax.bar(x - bar_width/2, true_vals, bar_width,
                           label='True', color=color_true)
        bars_pred = ax.bar(x + bar_width/2, pred_vals, bar_width,
                           label='Predicted', color=color_pred)
        
        for bar in bars_true:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                    height + (0.05 if height >= 0 else -0.05),
                    f'{height:.3f}', ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=8)
        for bar in bars_pred:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.,
                    height + (0.05 if height >= 0 else -0.05),
                    f'{height:.3f}', ha='center',
                    va='bottom' if height >= 0 else 'top',
                    fontsize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(structs, fontsize=10)
        ax.set_title(prop, fontsize=12)
        ax.set_ylim(ylim)
        ax.grid(False)   
        if idx == 0:
            ax.set_ylabel('Value (eV)', fontsize=12)
            ax.legend(loc='upper right', fontsize=9)
        else:
            ax.set_ylabel('')
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"预测对比图已保存至: {save_path}")

def plot_multiple_xafs_1(mids, element, json_dir='res/',
                       save_path='figures/samples_xafs_element.png', dpi=300):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    items = list(mids.items())
    if len(items) > 3:
        print(f"Warning: 传入 {len(items)} 个样品，仅绘制前三个。")
        items = items[:3]
    elif len(items) < 3:
        raise ValueError(f"需要至少三个样品，当前仅有 {len(items)} 个。")

    if isinstance(element, list):
        if len(element) < len(items):
            raise ValueError(f"element 列表长度 ({len(element)}) 小于样品数 ({len(items)})")
        elem_list = element[:len(items)]
    else:
        elem_list = [element] * len(items)

    items = items[::-1]
    elem_list = elem_list[::-1]

    n_rows = len(items)
    fig, axes = plt.subplots(n_rows, 1, figsize=(7, 1.8 * n_rows),
                             sharex=False, sharey=False)
    if n_rows == 1:
        axes = [axes]

    plt.subplots_adjust(hspace=0.05)

    colors = ['#6A5F51', '#164A3C', '#1C3885']

    for i, (mid, label) in enumerate(items):
        ax = axes[i]
        xafs_json = os.path.join(json_dir, mid, f'{mid}_xafs.json')
        with open(xafs_json, 'r') as f:
            xafs_data = json.load(f)

        elem = elem_list[i]
        key = f'{elem}_K_XAFS'
        if key not in xafs_data:
            raise KeyError(f"在 {xafs_json} 中未找到键 {key}")

        spectrum = xafs_data[key]
        energy = np.array(spectrum['energy'])
        intensity = np.array(spectrum['intensity'])

        intensity_norm = intensity / np.max(intensity)

        ax.plot(energy, intensity_norm, color=colors[i], linewidth=2.0)
        x_end = energy[-1]
        y_end = intensity_norm[-1]
        label_x = x_end - 2
        label_y = y_end - 0.17
        ax.text(label_x, label_y, elem, color=colors[i], fontsize=15,
                ha='right', va='bottom')

        x_pad = 40.0
        ax.set_xlim(energy[0] - x_pad, energy[-1] + x_pad)

        y_min, y_max = np.min(intensity_norm), np.max(intensity_norm)
        y_range = y_max - y_min
        y_pad = 0.1 * y_range if y_range > 0 else 0.1
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.tick_params(axis='x', labelbottom=True)
        ax.set_xlabel('')
        ax.set_ylabel('')

    fig.supxlabel('Energy (eV)', fontsize=15)
    fig.supylabel('Intensity μ(E)', fontsize=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"XAFS 多子图已保存至: {save_path}")
    plt.close()

if __name__ == '__main__':

    # ## 1.1 model performance 
    supp_plt()

    data_all = {
        'Model': ['CNN_xrd', 'RF_xrd', 'AEMLP_xafs', 'MLP_xafs', 'RF_xrd_xafs', 'InterLLC'],
        'Formation_MAE': [0.2629, 0.5173, 0.2484, 0.2856, 0.4724, 0.1884],
        'Formation_R2': [0.5245, 0.5656, 0.5639, 0.4647, 0.6322, 0.7443],
        'Fermi_MAE': [0.29, 0.5869, 0.2891, 0.3326, 0.5479, 0.2439],
        'Fermi_R2': [0.4183, 0.4021, 0.4209, 0.3064, 0.4758, 0.562],
        'Band_MAE': [0.3105, 0.6881, 0.2831, 0.3329, 0.6286, 0.206],
        'Band_R2': [0.2679, 0.2047, 0.3471, 0.2727, 0.3388, 0.5715]
    }

    data_abx3 = {
        'Model': ['CNN_xrd', 'RF_xrd', 'AEMLP_xafs', 'MLP_xafs', 'RF_xrd_xafs', 'InterLLC'
        ],
        'Formation_MAE': [0.2921, 0.5866, 0.2721, 0.2496, 0.5283, 0.2037],
        'Formation_R2': [0.4622, 0.4365, 0.5044, 0.5319, 0.5229, 0.7133],
        'Fermi_MAE': [0.295, 0.5805, 0.3025, 0.3007, 0.5742, 0.236],
        'Fermi_R2': [0.4344, 0.4448, 0.3865, 0.383, 0.4634, 0.5816],
        'Band_MAE': [0.3154, 0.6681, 0.2941, 0.2921, 0.5619, 0.2229],
        'Band_R2': [0.2666, 0.2116, 0.3212, 0.3329, 0.4156, 0.5163]
    }

    df_all = pd.DataFrame(data_all)
    df_abx3 = pd.DataFrame(data_abx3)

    plot_model_performance_1(df_abx3, 'ABX3', output_prefix='performance')

    # ## 1.2 model scatter figure 
    plot_scatter_for_properties()

    # # ## 2.1 AB test: fusion
    ab_fusion_plot()
    ab_fusion_statistic_plot()

    # # # ## 2.2 AB test: multitask learning
    ab_mtl_heatmap_abs_diff()
    ab_mtl_curve()
    ab_mtl_key_curve_statistic()
    
    # # # ## 3. feature attribution analyse
    plot_feature_importance()

    # # ## 4. Ab test: feature augmentation
    plot_ab_feat_augment()
    plot_ab_feat_augment_1()
    plot_ab_feat_augment_statistic()

    sample_feature_waterfall(target_mid="mp-567681")

    plot_xrd_(json_file='res/mp-27214/mp-27214_xrd.json', 
                     save_fig='figures/mp-27214_xrd_pattern.png',
                     )
    plot_xafs_(xafs_json_path='res/mp-27214/mp-27214_xafs.json', 
               element='Sn',
               save_path='figures/mp-27214_xafs_element.png',
               )

    ## case study
    
    cs_mid = ["mp-567629", "mp-567681", "mp-27214", "mp-5020", "mp-5777", "mp-5986", "mp-19990", "mp-504715", "mp-995191"]
    elements = ["Pb", "Pb", "Sn", "Ti", "Ti", "Ti", "Ti", "Ti", "Ti"]
    for mid, ele in zip(cs_mid, elements):
        sample_group_feature(mid)
        plot_xrd_pattern(json_file=f'res/{mid}/{mid}_xrd.json', 
                        save_fig=f'figures/{mid}_xrd_pattern.png',
                        )
        plot_absorption_spectra(xanes_json_path=f"res/{mid}/{mid}_xanes.json",
                                xafs_json_path=f"res/{mid}/{mid}_xafs.json",
                                element=ele,
                                save_path=f'figures/{mid}_xafs_element.png',
                                )
        
        json_file = f'res/{mid}_feature_data.json'
        plot_separate_multi_task_heatmaps(json_file, output_dir='figures/', prefix=mid)

    generate_dual_pie_matrices("abx3_used.txt", top_n_elements=12)
    generate_periodic_table("abx3_used.txt")


    mids_1 = {
            "mp-5986": "Tetragonal",
            "mp-5777": "Orthorhombic",
            "mp-5020": "Rhombohedral"
        }
    plot_multiple_xrd(mids_1, save_path='figures/samples_xrd_pattern_1.png')
    plot_multiple_xafs(mids_1, element="Ti", save_path='figures/samples_xafs_element_1.png')
    sample_group_feature(target_mid=list(mids_1.keys()), 
                         group_path="figures/samples_group_bar_1.png", 
                         titles=mids_1)
    mids_2 = {
            "mp-504715": "Cubic",
            "mp-19990": "Tetragonal",
            "mp-995191": "Monoclinic"
        }
    plot_multiple_xrd(mids_2, save_path='figures/samples_xrd_pattern_2.png')
    plot_multiple_xafs(mids_2, element="Ti", save_path='figures/samples_xafs_element_2.png')
    sample_group_feature(target_mid=list(mids_2.keys()), 
                         group_path="figures/samples_group_bar_2.png", 
                         titles=mids_2)
    mids_3 = {
            "mp-998323": "CsInBr3",
            "mp-27214": "CsSnBr3",
            "mp-570223": "CsGeBr3"
        }
    plot_multiple_xrd(mids_3, save_path='figures/samples_xrd_pattern_3.png')
    mids3 = list(mids_3.keys())
    elements = ["In", "Sn", "Ge"]
    for i in range(0, len(mids3)):
        plot_absorption_spectra(xanes_json_path=f"res/{mids3[i]}/{mids3[i]}_xanes.json",
                                xafs_json_path=f"res/{mids3[i]}/{mids3[i]}_xafs.json",
                                element=elements[i],
                                save_path=f'figures/{mids3[i]}_xafs_element.png',
                                )
    plot_multiple_xafs_1(mids_3, element=["In", "Sn", "Ge"], save_path='figures/samples_xafs_element_3.png')
    sample_group_feature(target_mid=list(mids_3.keys()), 
                         group_path="figures/samples_group_bar_3.png", 
                         titles=mids_3)