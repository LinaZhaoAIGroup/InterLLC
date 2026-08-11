import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


from processor import DataProcessor


def ml_comparer(args):
    processor = DataProcessor(data_dir="mp_abo3")
    valid_materials = processor.filter_materials_with_all_data(args.data, args.xas)
    
    if len(valid_materials) == 0:
        print("错误：没有找到符合条件的材料！")
        return
    
    xrd_features, xas_features, property_values = processor.feature_engineering()
    
    models = {
        'DT': DecisionTreeRegressor(random_state=42),
        'RF': RandomForestRegressor(random_state=42, n_jobs=-1),
        'SVR': SVR(),
        'XGBoost': XGBRegressor(random_state=42, verbosity=0) if XGBRegressor else None,
        'RR': Ridge(random_state=42),
        'MLP': MLPRegressor(random_state=42, max_iter=1000, early_stopping=True)
    }
    results = {}

    target_names = ['formation_energy', 'fermi_energy', 'band_gap']
    y_all = property_values  # shape: (n_samples, 3)

    for feat in ['xrd', 'xafs', 'xrd+xafs']:
        if feat == 'xrd':
            X = xrd_features
            feature_names = [f'xrd_{i}' for i in range(X.shape[1])]
        elif feat == 'xafs':
            X = xas_features
            feature_names = [f'xafs_{i}' for i in range(X.shape[1])]
        elif feat == 'xrd+xafs':
            X = np.hstack([xrd_features, xas_features])
            feature_names = [f'xrd_{i}' for i in range(xrd_features.shape[1])] + \
                            [f'xafs_{i}' for i in range(xas_features.shape[1])]
        else:
            raise ValueError("feat 必须是 'xrd', 'xafs' 或 'xrd+xafs'")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_all, test_size=0.2, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
        for model_name, model in models.items():
            print(f"\n{'='*60}")
            print(f"模型: {model_name}\t{feat}")
            results[model_name] = {}
            
            use_scaled = model_name in ['SVR', 'MLP']
            X_tr = X_train_scaled if use_scaled else X_train
            X_te = X_test_scaled if use_scaled else X_test
            
            for i, target_name in enumerate(target_names):
                y_tr = y_train[:, i]
                y_te = y_test[:, i]
                
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
                
                mae = mean_absolute_error(y_te, y_pred)
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                r2 = r2_score(y_te, y_pred)
                std_y = np.std(y_te)                     
                mae_normalized = mae / std_y             
                rmse_normalized = rmse / std_y
                
                print(f"属性: {target_name}")
                print(f"  MAE\tRMSE\tMAE_norm\tRMSE_norm\tR²")
                print(f"  {mae:.4f}\t{rmse:.4f}\t{mae_normalized:.4f}\t{rmse_normalized:.4f}\t{r2:.4f}")
                
                results[model_name][target_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_

                sorted_idx = np.argsort(importances)[::-1]
                print(f"\n{'='*20}")
                print(f"\n特征重要性排序{model_name} {feat}:")
                for rank, idx in enumerate(sorted_idx[:10], 1): 
                    print(f"  {rank}. {feature_names[idx]} : {importances[idx]:.4f}")
                    
            elif model_name == 'RR' and hasattr(model, 'coef_'):
                coef_abs = np.abs(model.coef_)
                sorted_idx = np.argsort(coef_abs)[::-1]
                print(f"\n特征重要性排序（{model_name}，基于系数绝对值）:")
                for rank, idx in enumerate(sorted_idx[:10], 1):
                    print(f"  {rank}. {feature_names[idx]} : {coef_abs[idx]:.4f}")
            else:
                print(f"\n模型 {model_name} 不支持直接输出特征重要性，跳过。")
    
    return results

