
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import interpolate, signal, special
from scipy.interpolate import interp1d
import pywt

from configurations import fxas_spectrum_type, fxas_len_map, elements_K_energies, simple_feature_source

def create_des_features(xas_data, fxas, roles, simple_feature_names, cite_elements=None, fxas_len=100):
    if not isinstance(fxas, list):
        des_spec_type = fxas_spectrum_type.get(fxas, 'xafs')
        des_xas_data = extract_spectrum_region(xas_data, des_spec_type, cite_elements)
        fxas_len = fxas_len_map.get(fxas, 100)
        xas_features = extract_xas_features(des_xas_data, fxas, cite_elements, fxas_len)
        return xas_features
    
    else:
        descriptors = []
        simple_features = []
        if roles:
            simple_features = simple_descriptors_xas(xas_data, roles, simple_feature_names)

        for des in fxas: 
            try:
                des_spec_type = fxas_spectrum_type.get(des, 'xafs')
                des_xas_data = extract_spectrum_region(xas_data, des_spec_type, cite_elements)
                fxas_len = fxas_len_map.get(des, 100)
                xas_features = extract_xas_features(des_xas_data, des, cite_elements, fxas_len)
                descriptors.append(xas_features)
            except Exception as e:
                print(f"创建特征 {des} 时出错: {e}")
                descriptors.append(np.zeros(fxas_len, dtype=np.float32))
        return simple_features, descriptors
    

def extract_xas_features(xas_data, fxas, cite_elements=None, fxas_len=100):
    if fxas == 'cdf':
        return cdf_xas_feature(xas_data, cite_elements, fxas_len)
    elif fxas == 'cwt':
        return cwt_xas_feature(xas_data, cite_elements, fxas_len)
    
    elif fxas == 'wacsf':
        return wacsf_xas_feature(xas_data, cite_elements, fxas_len)
    elif fxas == 'soap2':
        return soap2_xas_feature(xas_data, cite_elements, fxas_len)
    elif fxas == 'pdos':
        return pdos_xas_feature(xas_data, cite_elements, fxas_len)
    elif fxas == 'msr1':
        return msr1_xas_feature(xas_data, cite_elements, fxas_len)
    

def extract_spectrum_region(xafs_data, region_type, cite_elements=None):
        extracted_data = {}
        
        for key, spectrum in xafs_data.items():
            if 'energy' not in spectrum or 'intensity' not in spectrum:
                continue
                
            element = spectrum.get("absorbing_element", "")
        
            if cite_elements and element not in cite_elements:
                continue
                
            energies = np.array(spectrum['energy'])
            intensities = np.array(spectrum['intensity'])
            
            if len(energies) < 10:
                continue
            
            edge_energy = np.min(energies)  
            
            if region_type == 'XANES':
                xanes_range = 50.0  # eV
                mask = np.abs(energies - edge_energy) <= xanes_range
                
                if np.sum(mask) < 5:
                    extracted_data[key] = spectrum
                else:
                    extracted_data[key] = {
                        'energy': energies[mask].tolist(),
                        'intensity': intensities[mask].tolist(),
                        'absorbing_element': element,
                        'edge_energy': edge_energy,
                        'region': 'XANES'
                    }
                    
            elif region_type == 'EXAFS':
                exafs_start = edge_energy + 30.0  
                exafs_end = edge_energy + 800.0   
                
                mask = (energies >= exafs_start) & (energies <= exafs_end)
                
                if np.sum(mask) < 10:
                    extracted_data[key] = spectrum
                else:
                    exafs_energies = energies[mask]
                    exafs_intensities = intensities[mask]
                    
                    if len(exafs_energies) > 200:
                        indices = np.linspace(0, len(exafs_energies)-1, 200, dtype=int)
                        exafs_energies = exafs_energies[indices]
                        exafs_intensities = exafs_intensities[indices]
                    
                    extracted_data[key] = {
                        'energy': exafs_energies.tolist(),
                        'intensity': exafs_intensities.tolist(),
                        'absorbing_element': element,
                        'edge_energy': edge_energy,
                        'region': 'EXAFS'
                    }
            else:
                extracted_data[key] = spectrum
        
        return extracted_data


import numpy as np
from scipy.stats import linregress
from scipy.integrate import trapz

def simple_descriptors_xas(xas_data, roles, simple_feature_names):
    num_desc = len(simple_feature_names)  

    features = [0.0] * num_desc
    site_data = {'A': {}, 'B': {}, 'X': {}}

    for key, spectrum in xas_data.items():
        if 'energy' not in spectrum or 'intensity' not in spectrum:
            continue
        energies = np.array(spectrum['energy'])
        intensities = np.array(spectrum['intensity'])
        element = spectrum.get("absorbing_element", "")
        edge_energy = spectrum.get("edge_energy", None)

        site = None
        for role, elem in roles.items():
            if element == elem[0]:
                site = role
                break
        if site is None:
            continue   

        if not np.all(np.diff(energies) > 0):
            sort_idx = np.argsort(energies)
            energies = energies[sort_idx]
            intensities = intensities[sort_idx]

        norm_intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities) + 1e-8)
        if edge_energy is None:
            edge_energy = elements_K_energies.get(element, None)
            if edge_energy is None:
                if np.std(np.diff(energies)) / (np.mean(np.diff(energies)) + 1e-8) > 0.1:
                    first_deriv = np.gradient(norm_intensities)
                else:
                    first_deriv = np.gradient(norm_intensities, energies)
                edge_idx = np.argmax(np.abs(first_deriv))
                edge_energy = energies[edge_idx] if edge_idx < len(energies) else energies[0]
        edge_idx = np.argmin(np.abs(energies - float(edge_energy)))

        xanes_mask = (energies >= edge_energy - 30) & (energies <= edge_energy + 50)
        if np.sum(xanes_mask) > 5:
            xanes_E = energies[xanes_mask]
            xanes_I = norm_intensities[xanes_mask]

            pre_mask = (xanes_E >= edge_energy - 15) & (xanes_E <= edge_energy - 2)
            pre_integral = trapz(xanes_I[pre_mask], xanes_E[pre_mask]) if np.sum(pre_mask) > 0 else 0.0

            post_mask = (xanes_E >= edge_energy + 5) & (xanes_E <= edge_energy + 30)
            white_line_pos = 0.0
            white_line_fwhm = 0.0
            if np.sum(post_mask) > 0:
                post_E = xanes_E[post_mask]
                post_I = xanes_I[post_mask]
                wl_idx = np.argmax(post_I)
                white_line_pos = post_E[wl_idx]
                half = post_I[wl_idx] / 2
                left = wl_idx
                while left > 0 and post_I[left] > half:
                    left -= 1
                right = wl_idx
                while right < len(post_I)-1 and post_I[right] > half:
                    right += 1
                white_line_fwhm = post_E[right] - post_E[left] if right > left else 0.0

            site_data[site]['pre_edge'] = pre_integral
            site_data[site]['white_pos'] = white_line_pos
            site_data[site]['white_fwhm'] = white_line_fwhm
            site_data[site]['edge_energy'] = edge_energy

        if energies[-1] - energies[0] > 100:   
            # 转换为k空间
            mask_k = energies > edge_energy + 10
            if np.sum(mask_k) > 5:
                E_k = energies[mask_k]
                mu_k = intensities[mask_k]
                k = np.sqrt((E_k - edge_energy) * 0.2625)  
                k_mask = (k >= 2) & (k <= 12)
                k = k[k_mask]
                mu_k = mu_k[k_mask]
                if len(k) > 5:
                    mu_norm = (mu_k - np.mean(mu_k[-5:])) / (np.mean(mu_k[:5]) - np.mean(mu_k[-5:]) + 1e-8)
                    chi = mu_norm * (k ** 2)  

                    k_grid = np.linspace(k.min(), k.max(), 256)
                    chi_interp = np.interp(k_grid, k, chi)
                    window = np.hanning(len(chi_interp))
                    chi_windowed = chi_interp * window
                    dr = np.pi / (k_grid[-1] - k_grid[0])
                    ft = np.fft.rfft(chi_windowed)
                    r = np.fft.rfftfreq(len(chi_windowed), d=dr) * 2 * np.pi
                    ft_mag = np.abs(ft)

                    mask_r1 = (r >= 1.0) & (r <= 2.5)
                    if np.sum(mask_r1) > 0:
                        r1_idx = np.argmax(ft_mag[mask_r1]) + np.where(mask_r1)[0][0]
                        r1_peak = r[r1_idx]
                        bond_length = r1_peak + 0.4   

                        half_max = ft_mag[r1_idx] / 2
                        left = r1_idx
                        while left > 0 and ft_mag[left] > half_max:
                            left -= 1
                        right = r1_idx
                        while right < len(ft_mag)-1 and ft_mag[right] > half_max:
                            right += 1
                        disorder = r[right] - r[left] if right > left else 0.0

                        mask_r2 = (r >= 3.0) & (r <= 4.0)
                        if np.sum(mask_r2) > 0:
                            r2_idx = np.argmax(ft_mag[mask_r2]) + np.where(mask_r2)[0][0]
                            ft_ratio = ft_mag[r1_idx] / (ft_mag[r2_idx] + 1e-8)
                        else:
                            ft_ratio = 0.0

                        site_data[site]['bond_length'] = bond_length
                        site_data[site]['disorder'] = disorder
                        site_data[site]['ft_ratio'] = ft_ratio

                    chi_abs = np.abs(chi)
                    if len(k) > 5:
                        log_chi = np.log(chi_abs + 1e-8)
                        slope, _, _, _, _ = linregress(k, log_chi)
                        decay_rate = -slope if slope < 0 else 0.0
                        site_data[site]['decay_rate'] = decay_rate

        if 'pre_edge' in site_data[site]:
            total_integral = trapz(xanes_I, xanes_E)
            site_data[site]['coordination'] = total_integral / 50.0   

    for idx, desc in enumerate(simple_feature_names):
        source = simple_feature_source[desc]

        if desc == 'B_X_bond_length':
            features[idx] = site_data['B'].get('bond_length', 0.0)
        elif desc == 'A_site_displacement':
            a_bond = site_data['A'].get('bond_length', 0.0)
            features[idx] = a_bond
        elif desc == 'X_X_average_distance':
            features[idx] = site_data['X'].get('bond_length', 0.0)  

    features = [float(np.nan_to_num(f, nan=0.0)) for f in features]
    return features

def cdf_xas_feature(xas_data, cite_elements=None, maxseq=50):
    """
    累积分布函数(CDF)特征
    主要思想：将XAS强度视为概率分布，计算其累积分布函数
    """
    features = []
    all_intensities = []
    
    for key, spectrum in xas_data.items():
        if 'intensity' in spectrum:
            intensities = spectrum['intensity']
            element = spectrum["absorbing_element"]
            
            if (cite_elements and element in cite_elements) or cite_elements == None:
                if len(intensities) > 10:
                    
                    intensities = np.array(intensities)
                    if np.max(intensities) > 0:
                        intensities = intensities / np.max(intensities)
                    all_intensities.extend(intensities.tolist())
    
    if not all_intensities:
        return [0.0] * maxseq
    
    intensities_array = np.array(all_intensities)
    sorted_intensities = np.sort(intensities_array)
    
    cdf_values = np.arange(1, len(sorted_intensities) + 1) / len(sorted_intensities)
    
    quantiles = np.linspace(0, 1, maxseq) 
    cdf_sampled = np.interp(quantiles, sorted_intensities, cdf_values, left=0, right=1)
    
    features.extend(cdf_sampled.tolist())
    return features[:maxseq]


def cwt_xas_feature(xas_data, cite_elements=None, maxseq=300):
    """
    连续小波变换特征
    主要思想：使用小波变换在多尺度上分析XAS谱
    """
    features = []
    all_intensities = []
    
    for key, spectrum in xas_data.items():
        if 'energy' in spectrum and 'intensity' in spectrum:
            energies = spectrum['energy']
            intensities = spectrum['intensity']
            element = spectrum["absorbing_element"]
            
            if (cite_elements and element in cite_elements) or cite_elements == None:
                if len(energies) > 50:
                    try:
                        f = interpolate.interp1d(energies, intensities, 
                                            bounds_error=False, fill_value=0)
                        new_energies = np.linspace(min(energies), max(energies), maxseq)
                        interp_intensity = f(new_energies)
                        
                        if np.max(interp_intensity) > 0:
                            interp_intensity = interp_intensity / np.max(interp_intensity)
                        
                        all_intensities.append(interp_intensity)
                    except Exception as e:
                        print(f"警告: 处理{element}谱时出错: {e}")
                        continue
    
    if not all_intensities:
        return [0.0] * maxseq
    
    avg_intensity = np.mean(all_intensities, axis=0)
    
    scales = np.arange(1, 32)  
    coefficients, frequencies = pywt.cwt(avg_intensity, scales, 'mexh')
    
    cwt_features = []
    for i in range(min(10, len(scales))):  
        coeff = coefficients[i]
        cwt_features.extend([
            np.mean(coeff),
            np.std(coeff),
            np.max(coeff),
            np.min(coeff),
            np.mean(np.abs(coeff)),
            np.percentile(coeff, 25),
            np.percentile(coeff, 75),
            np.sum(coeff > 0) / len(coeff)  
        ])
    
    cwt_features.extend([
        np.mean(coefficients),
        np.std(coefficients),
        np.max(coefficients),
        np.min(coefficients)
    ])
    
    features = cwt_features[:maxseq]  
    if len(features) < maxseq:
        features.extend([0.0] * (maxseq - len(features)))
    
    return features[:maxseq]


def wacsf_xas_feature(xas_data, cite_elements=None, maxseq=100):
    """
    wACSF-like特征提取
    主要思想：将XAS谱视为原子环境的"指纹"，通过高斯函数描述局部电子结构
    """
    features = []
    
    for key, spectrum in xas_data.items():
        if 'energy' in spectrum and 'intensity' in spectrum:
            element = spectrum["absorbing_element"]
            
            if (cite_elements and element in cite_elements) or cite_elements is None:
                energies = np.array(spectrum['energy'])
                intensities = np.array(spectrum['intensity'])
                
                if len(energies) < 10:
                    continue
                
                energies_norm = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
                intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
                
                n_gaussians = 10  
                gaussian_centers = np.linspace(0, 1, n_gaussians)
                gaussian_width = 0.1
                
                wacsf_features = []
                for center in gaussian_centers:
                    weights = np.exp(-((energies_norm - center) ** 2) / (2 * gaussian_width ** 2))
                    weighted_intensity = np.sum(weights * intensities_norm) / (np.sum(weights) + 1e-8)
                    wacsf_features.append(weighted_intensity)
                
                features.extend(wacsf_features)
    if not features:
        features = [0.0] * (10 * len(xas_data)) if xas_data else [0.0] * 10
    if len(features) > maxseq:
        features = features[:maxseq]
    else:
        features.extend([0.0] * (maxseq - len(features)))
    
    return features[:maxseq]


def find_absorption_edge(energies, intensities, threshold=0.5):
    if len(intensities) == 0:
        return 0.0
    
    deriv = np.gradient(intensities, energies)
    if len(deriv) > 0:
        max_deriv_idx = np.argmax(np.abs(deriv))
        return energies[max_deriv_idx] if max_deriv_idx < len(energies) else energies[0]
    return energies[0]


def estimate_edge_width(energies, intensities, fraction=0.8):
    if len(intensities) < 2:
        return 0.0
    
    max_intensity = np.max(intensities)
    threshold = max_intensity * fraction
    
    above_threshold = intensities >= threshold
    if np.any(above_threshold):
        start_idx = np.where(above_threshold)[0][0]
        end_idx = np.where(above_threshold)[0][-1]
        return energies[end_idx] - energies[start_idx]
    
    return 0.0


def cosine_cutoff(x, cutoff):
    return 0.5 * (np.cos(np.pi * x / cutoff) + 1)


def soap2_xas_feature(xas_data, cite_elements=None, maxseq=100):
    """
    SOAP-like特征提取
    将XAS谱的强度分布视为电子密度，用正交化的径向基函数和球谐函数展开，
             并计算功率谱以获得旋转不变的特征描述符
    """
    features = []
    
    if maxseq >= 50:
        n_max = 4  
        l_max = 4  
    elif maxseq >= 30:
        n_max = 3
        l_max = 3
    else:
        n_max = 2
        l_max = 2
    soap_dim = n_max * (n_max + 1) // 2 * (l_max + 1)
    
    soap_vectors = []
    
    for key, spectrum in xas_data.items():
        if 'energy' in spectrum and 'intensity' in spectrum:
            element = spectrum.get("absorbing_element", "")
            
            if (cite_elements and element in cite_elements) or cite_elements is None:
                energies = np.array(spectrum['energy'])
                intensities = np.array(spectrum['intensity'])
                
                if len(energies) < 10:
                    continue
                
                grid_size = 100 
                energy_norm = np.linspace(0, 1, grid_size)
                
                e_min, e_max = energies.min(), energies.max()
                if e_max - e_min > 1e-10:
                    e_scaled = (energies - e_min) / (e_max - e_min)
                else:
                    e_scaled = np.linspace(0, 1, len(energies))
                
                i_min, i_max = intensities.min(), intensities.max()
                if i_max - i_min > 1e-10:
                    intensity_norm = (intensities - i_min) / (i_max - i_min)
                else:
                    intensity_norm = intensities / (np.max(np.abs(intensities)) + 1e-8)
                
                f_interp = interp1d(e_scaled, intensity_norm, kind='cubic', 
                                    bounds_error=False, fill_value=0)
                intensity_grid = f_interp(energy_norm)
                
                radial_basis = []
                r_cut = 1.0  
                
                for n in range(n_max):
                    r_center = 0.1 + 0.8 * (n + 0.5) / n_max
                    sigma = 0.15  
                    
                    g = np.exp(-0.5 * ((energy_norm - r_center) / sigma) ** 2)
                    
                    cutoff = 0.5 * (np.cos(np.pi * energy_norm / r_cut) + 1)
                    g = g * cutoff
                    
                    radial_basis.append(g)
                
                ortho_radial = []
                for i in range(n_max):
                    g = radial_basis[i].copy()
                    for j in range(i):
                        proj = np.sum(g * ortho_radial[j]) / np.sum(ortho_radial[j] ** 2)
                        g = g - proj * ortho_radial[j]
                    
                    norm = np.linalg.norm(g)
                    if norm > 1e-10:
                        g = g / norm
                    
                    ortho_radial.append(g)
                cos_theta = 2 * energy_norm - 1
                
                angular_basis = []
                for l in range(l_max + 1):
                    pl = special.legendre(l)(cos_theta)
                    
                    norm = np.linalg.norm(pl)
                    if norm > 1e-10:
                        pl = pl / norm
                    
                    angular_basis.append(pl)
                
                expansion_coeffs = np.zeros((n_max, l_max + 1))
                
                for n in range(n_max):
                    for l in range(l_max + 1):
                        # 系数计算：∫ ρ(r) * g_n(r) * P_l(cosθ) dr
                        coeff = np.sum(intensity_grid * ortho_radial[n] * angular_basis[l])
                        expansion_coeffs[n, l] = coeff
                
                # 功率谱 p_{nn'l} = ∑_m c_{nlm} * c_{n'lm} 
                soap_vector = []
                for n1 in range(n_max):
                    for n2 in range(n1, n_max): 
                        for l in range(l_max + 1):
                            power = expansion_coeffs[n1, l] * expansion_coeffs[n2, l]
                            if n1 == n2:
                                power = power / (n_max * (l_max + 1))
                            
                            soap_vector.append(power)
                
                if len(soap_vector) < soap_dim:
                    stats = [
                        np.mean(intensity_grid),
                        np.std(intensity_grid),
                        np.sum(intensity_grid > 0.5) / len(intensity_grid),  
                        np.trapz(intensity_grid, energy_norm) 
                    ]
                    soap_vector.extend(stats[:min(4, soap_dim - len(soap_vector))])
                
                soap_vector = soap_vector[:soap_dim]
                if len(soap_vector) < soap_dim:
                    soap_vector.extend([0.0] * (soap_dim - len(soap_vector)))
                
                soap_vectors.append(soap_vector)
    
    if not soap_vectors:
        features = [0.0] * min(soap_dim, maxseq)
    else:
        soap_matrix = np.array(soap_vectors)
        avg_soap = np.mean(soap_matrix, axis=0)
        features = avg_soap.tolist()
    
    features = features[:maxseq]
    if len(features) < maxseq:
        features.extend([0.0] * (maxseq - len(features)))
    
    return features[:maxseq]


def pdos_xas_feature(xas_data, cite_elements=None, maxseq=100):
    """
    pdos-like特征提取
    主要思想：将XAS视为未占据态密度的直接探针，提取轨道投影信息
    """
    features = []
    
    orbital_ranges = {
        's': (0.0, 0.2),   
        'p': (0.2, 0.5),    
        'd': (0.5, 0.8),   
        'f': (0.8, 1.0)    
    }
    
    for key, spectrum in xas_data.items():
        if 'energy' in spectrum and 'intensity' in spectrum:
            element = spectrum["absorbing_element"]
            
            if (cite_elements and element in cite_elements) or cite_elements is None:
                energies = np.array(spectrum['energy'])
                intensities = np.array(spectrum['intensity'])
                
                if len(energies) < 10:
                    continue
                
                energies_norm = (energies - energies.min()) / (energies.max() - energies.min() + 1e-8)
                intensities_norm = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-8)
                
                orbital_features = []
                for orbital, (e_min, e_max) in orbital_ranges.items():
                    mask = (energies_norm >= e_min) & (energies_norm <= e_max)
                    if np.any(mask):
                        # 积分强度
                        integral = np.trapz(intensities_norm[mask], energies_norm[mask])
                        # 平均强度
                        mean_int = np.mean(intensities_norm[mask]) if np.sum(mask) > 0 else 0
                        # 峰值强度
                        max_int = np.max(intensities_norm[mask]) if np.sum(mask) > 0 else 0
                        
                        orbital_features.extend([integral, mean_int, max_int])
                    else:
                        orbital_features.extend([0.0, 0.0, 0.0])
                
                features.extend(orbital_features)
    
    if not features:
        n_orbitals = len(orbital_ranges)
        features = [0.0] * (n_orbitals * 3)  
    
    if len(features) > maxseq:
        features = features[:maxseq]
    else:
        features.extend([0.0] * (maxseq - len(features)))
    
    return features[:maxseq]


def msr1_xas_feature(xas_data, cite_elements=None, maxseq=100):
    """
    msr-like特征提取
    主要思想：将XAS谱分解为不同阶次多重散射路径的贡献，
             模拟S2（单次散射）、S3（二次散射）、S4（三次散射）等路径特征
    """
    features = []
    
    if maxseq >= 80:
        n_s2 = 30  
        n_s3 = 30  
        n_s4 = 20  
    elif maxseq >= 50:
        n_s2 = 25
        n_s3 = 25
        n_s4 = 0
    else:
        n_s2 = maxseq
        n_s3 = 0
        n_s4 = 0
    
    r_min = 0.0  
    r_max = 1.0   
    sigma = 0.05 
    
    all_msr_features = []
    
    for key, spectrum in xas_data.items():
        if 'energy' in spectrum and 'intensity' in spectrum:
            element = spectrum.get("absorbing_element", "")
            
            if (cite_elements and element in cite_elements) or cite_elements is None:
                energies = np.array(spectrum['energy'])
                intensities = np.array(spectrum['intensity'])
                
                if len(energies) < 20:  
                    continue
                
                diff = np.gradient(intensities, energies)
                e0_idx = np.argmax(np.abs(diff))
                e0_idx = min(max(e0_idx, 5), len(energies) - 5)
                e0 = energies[e0_idx]
                
                energy_shift = energies - e0
                
                pos_mask = energy_shift >= 0
                if np.sum(pos_mask) < 10:
                    pos_mask = np.ones_like(energy_shift, dtype=bool)
                
                energy_pos = energy_shift[pos_mask]
                intensity_pos = intensities[pos_mask]
                if len(energy_pos) > 5:
                    e_min, e_max = energy_pos.min(), energy_pos.max()
                    if e_max - e_min > 1e-10:
                        energy_norm = (energy_pos - e_min) / (e_max - e_min)
                        intensity_norm = (intensity_pos - intensity_pos.min()) / (intensity_pos.max() - intensity_pos.min() + 1e-8)
                        
                        total_norm = np.linalg.norm(intensity_norm)
                        if total_norm > 1e-10:
                            intensity_norm = intensity_norm / total_norm
                    else:
                        continue 
                else:
                    continue

                r_grid_s2 = np.linspace(r_min, r_max, n_s2)  
                r_grid_s3 = np.linspace(r_min, r_max, n_s3) if n_s3 > 0 else np.array([])
                r_grid_s4 = np.linspace(r_min, r_max, n_s4) if n_s4 > 0 else np.array([])
                
                s2_features = []
                if n_s2 > 0:
                    for r_target in r_grid_s2:
                        e_target = 1.0 - r_target / r_max
                        gaussian_weights = np.exp(-0.5 * ((energy_norm - e_target) / sigma) ** 2)
                        if element in ['Fe', 'Co', 'Ni', 'Cu']:
                            z_weight = 0.2  
                        else:
                            z_weight = 0.1
                        s2_contribution = np.sum(intensity_norm * gaussian_weights) * z_weight
                        s2_features.append(s2_contribution)
                    cutoff = 0.5 * (np.cos(np.pi * r_grid_s2 / r_max) + 1)
                    s2_features = np.array(s2_features) * cutoff
                
                s3_features = []
                if n_s3 > 0 and len(energy_norm) >= 10:
                    intensity_smooth = signal.savgol_filter(intensity_norm, 5, 3)
                    second_deriv = np.gradient(np.gradient(intensity_smooth, energy_norm), energy_norm)
                    
                    for r_target in r_grid_s3:
                        e_target = 1.0 - 0.7 * (r_target / r_max)
                        gaussian_weights = np.exp(-0.5 * ((energy_norm - e_target) / sigma) ** 2)
                        angle_factor = np.abs(second_deriv)
                        angle_factor = angle_factor / (np.max(np.abs(angle_factor)) + 1e-8)
                        if element in ['Fe', 'Co', 'Ni', 'Cu']:
                            z_weight = 0.15
                        else:
                            z_weight = 0.08
                        
                        s3_contribution = np.sum(intensity_norm * angle_factor * gaussian_weights) * z_weight
                        s3_features.append(s3_contribution)
                    cutoff = 0.5 * (np.cos(np.pi * r_grid_s3 / r_max) + 1)
                    s3_features = np.array(s3_features) * cutoff
                
                s4_features = []
                if n_s4 > 0 and len(energy_norm) >= 15:
                    scales = [0.05, 0.1, 0.2]
                    multiscale_response = np.zeros_like(intensity_norm)
                    for scale in scales:
                        scale_sigma = scale * len(intensity_norm)
                        if scale_sigma > 1:
                            window = signal.windows.gaussian(len(intensity_norm), scale_sigma)
                            window = window / np.sum(window)
                            filtered = np.convolve(intensity_norm, window, mode='same')
                            multiscale_response += filtered * scale
                    
                    if np.max(multiscale_response) > 0:
                        multiscale_response = multiscale_response / np.max(multiscale_response)
                    
                    for r_target in r_grid_s4:
                        e_target = 1.0 - 0.4 * (r_target / r_max)
                        gaussian_weights = np.exp(-0.5 * ((energy_norm - e_target) / sigma) ** 2)
                        if element in ['Fe', 'Co', 'Ni', 'Cu']:
                            z_weight = 0.1
                        else:
                            z_weight = 0.05
                        
                        s4_contribution = np.sum(multiscale_response * gaussian_weights) * z_weight
                        s4_features.append(s4_contribution)
                    cutoff = 0.5 * (np.cos(np.pi * r_grid_s4 / r_max) + 1)
                    s4_features = np.array(s4_features) * cutoff

                msr_vector = []
                msr_vector.extend(s2_features)
                if n_s3 > 0:
                    msr_vector.extend(s3_features)
                if n_s4 > 0:
                    msr_vector.extend(s4_features)
                current_len = len(msr_vector)
                remaining = maxseq - current_len
                
                if remaining >= 3 and len(s2_features) > 0:
                    s2_total = np.sum(np.abs(s2_features))
                    s3_total = np.sum(np.abs(s3_features)) if n_s3 > 0 else 0
                    s4_total = np.sum(np.abs(s4_features)) if n_s4 > 0 else 0
                    
                    total = s2_total + s3_total + s4_total
                    if total > 0:
                        msr_vector.extend([
                            s2_total / total,  
                            s3_total / total if n_s3 > 0 else 0, 
                            s4_total / total if n_s4 > 0 else 0  
                        ])

                current_len = len(msr_vector)
                remaining = maxseq - current_len
                if remaining >= 4 and len(s2_features) > 0:
                    peaks, properties = signal.find_peaks(s2_features, height=0.1)
                    if len(peaks) > 0:
                        main_peak_idx = np.argmax(properties['peak_heights'])
                        main_r = r_grid_s2[peaks[main_peak_idx]]
                        main_height = properties['peak_heights'][main_peak_idx]
                        msr_vector.extend([
                            main_r,      
                            main_height, 
                            len(peaks),  
                            np.std(s2_features) if len(s2_features) > 1 else 0  
                        ])
                
                total_features = n_s2 + n_s3 + n_s4
                msr_vector = msr_vector[:total_features]
                all_msr_features.append(msr_vector)
    
    if not all_msr_features:
        total_features = n_s2 + n_s3 + n_s4
        features = [0.0] * min(total_features, maxseq)
    else:
        max_len = max(len(v) for v in all_msr_features)
        padded_features = []
        
        for v in all_msr_features:
            if len(v) < max_len:
                v_padded = v + [0.0] * (max_len - len(v))
                padded_features.append(v_padded)
            else:
                padded_features.append(v[:max_len])
    
        avg_features = np.mean(padded_features, axis=0)
        features = avg_features.tolist()
    
    features = features[:maxseq]
    if len(features) < maxseq:
        features.extend([0.0] * (maxseq - len(features)))
    
    return features[:maxseq]


def classify_element(element):
    element = str(element).strip()
    d3_metals = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
    d4_metals = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
    rare_earths = ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 
                   'Ho', 'Er', 'Tm', 'Yb', 'Lu']
    
    if element in d3_metals:
        return '3d_metal'
    elif element in d4_metals:
        return '4d_metal'
    elif element in rare_earths:
        return 'rare_earth'
    else:
        return 'main_group'