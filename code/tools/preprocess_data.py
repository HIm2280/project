#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单位严格版
- strain  → mm
- time    → s
- 其余不变
"""
import os
import glob
import json
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler

BASE_DIR = "/root/pytorch_project/metal_am_strain_prediction/3D_U_net/dataset"
IN_DIR   = os.path.join(BASE_DIR, "pre_input")
OUT_DIR  = os.path.join(BASE_DIR, "pre_real")
NORM_IN  = os.path.join(BASE_DIR, "pre_input")
NORM_OUT = os.path.join(BASE_DIR, "pre_real")
SCALER_FILE = os.path.join(BASE_DIR, "scalers.pkl")

FEATURE_NAMES = ["Cooling_Rate", "Slow_Cooling_Rate", "Dwell_Time",
                 "Remelting_Count", "Max_Temperature", "Cooling_Duration"]

# ------------------------------------------------------------------
# 1. 单位开关（根据你的原始数据情况二选一）
# ------------------------------------------------------------------
STRAIN_ALREADY_IN_MM = False      # True → 原始 NPY 里 strain 已经是 mm
GAUGE_LENGTH_MM      = 1.0        # 仅在 STRAIN_ALREADY_IN_MM = False 时用到

# ------------------------------------------------------------------
# 工具：分块 concatenate
# ------------------------------------------------------------------
def collect_channel_bigvec(name):
    """返回 (N,1) 大矩阵，峰值 < 1 GB"""
    big = []
    for f in tqdm(glob.glob(os.path.join(IN_DIR, '*.npy')), desc=f'load {name}', leave=False):
        arr = np.load(f)[0]                   # [C,64,64,64]
        c = FEATURE_NAMES.index(name)
        vals = arr[c].reshape(-1, 1)
        if name in {'Dwell_Time', 'Cooling_Duration'}:  # 保证 >0
            vals = np.where(vals > 0, vals, 1e-8)
        big.append(vals)
    return np.concatenate(big, axis=0)        # (N,1)

# ------------------------------------------------------------------
# 2. 统计 + 调试
# ------------------------------------------------------------------
def collect_stats():
    print('🔍 收集统计量')
    stats = {name: {'min': np.inf, 'max': -np.inf, 'sum': 0.0, 'n': 0}
             for name in FEATURE_NAMES + ['output']}
    for f in tqdm(glob.glob(os.path.join(IN_DIR, '*.npy')), desc='通道统计'):
        arr = np.load(f)[0]
        for c, name in enumerate(FEATURE_NAMES):
            vals = arr[c].reshape(-1)
            if name in {'Dwell_Time', 'Cooling_Duration'}:
                vals = np.where(vals > 0, vals, 1e-8)
            stats[name]['min'] = min(stats[name]['min'], vals.min())
            stats[name]['max'] = max(stats[name]['max'], vals.max())
            stats[name]['sum'] += vals.sum()
            stats[name]['n']   += vals.size

    # output 通道
    for f in tqdm(glob.glob(os.path.join(OUT_DIR, '*.npy')), desc='output 统计'):
        vals = np.load(f)[0].reshape(-1)
        if not STRAIN_ALREADY_IN_MM:                
            vals = vals * GAUGE_LENGTH_MM
        stats['output']['min'] = min(stats['output']['min'], vals.min())
        stats['output']['max'] = max(stats['output']['max'], vals.max())
        stats['output']['sum'] += vals.sum()
        stats['output']['n']   += vals.size

    print('==== 各通道调试信息 ====')
    for name in FEATURE_NAMES + ['output']:
        s = stats[name]
        print(f'{name:18s}  min={s["min"]:8.4f}  max={s["max"]:8.4f}  mean={s["sum"]/s["n"]:8.4f}')
    print('========================')
    return stats

# ------------------------------------------------------------------
# 3. 拟合 scaler
# ------------------------------------------------------------------
def fit_and_save_scalers(stats):
    scalers = {}

    # 3.1 在线 partial_fit 通道
    online_names = {'Cooling_Rate', 'Max_Temperature'}
    for c, name in enumerate(FEATURE_NAMES):
        if name in online_names:
            print(f'拟合 {name} (online StandardScaler) ...')
            scalers[name] = StandardScaler()
            for f in tqdm(glob.glob(os.path.join(IN_DIR, '*.npy')), desc=name, leave=False):
                arr = np.load(f)[0]
                vals = arr[c].reshape(-1, 1)
                scalers[name].partial_fit(vals)

    # 3.2 PowerTransformer 一次性 fit
    pt_name = 'Slow_Cooling_Rate'
    print(f'拟合 {pt_name} (PowerTransformer-YJ) ...')
    big_vec = collect_channel_bigvec(pt_name)
    scalers[pt_name] = PowerTransformer(method='yeo-johnson')
    scalers[pt_name].fit(big_vec)
    del big_vec

    # 3.3 RobustScaler 一次性 fit
    rb_name = 'Remelting_Count'
    print(f'拟合 {rb_name} (RobustScaler) ...')
    big_vec = collect_channel_bigvec(rb_name)
    scalers[rb_name] = RobustScaler()
    scalers[rb_name].fit(big_vec)
    del big_vec

    # 3.4 对数 + StandardScaler 在线
    log_names = {'Dwell_Time', 'Cooling_Duration'}
    for c, name in enumerate(FEATURE_NAMES):
        if name in log_names:
            print(f'拟合 {name} (log1p + online StandardScaler) ...')
            scalers[name] = StandardScaler()
            for f in tqdm(glob.glob(os.path.join(IN_DIR, '*.npy')), desc=name, leave=False):
                arr = np.load(f)[0]
                vals = arr[c].reshape(-1, 1)
                vals = np.where(vals > 0, vals, 1e-8)
                vals = np.log1p(vals)              # 秒单位直接 log1p
                scalers[name].partial_fit(vals)

    # 3.5 output 在线
    print('拟合 output (mm) ...')
    out_scaler = StandardScaler()
    for f in tqdm(glob.glob(os.path.join(OUT_DIR, '*.npy')), desc='output', leave=False):
        vals = np.load(f)[0].reshape(-1, 1)
        if not STRAIN_ALREADY_IN_MM:
            vals = vals * GAUGE_LENGTH_MM       
        out_scaler.partial_fit(vals)
    scalers['output'] = out_scaler

    with open(SCALER_FILE, 'wb') as pf:
        pickle.dump(scalers, pf)
    print(f'✅ scaler 保存至 {SCALER_FILE}')
    return scalers

# ------------------------------------------------------------------
# 4. 应用 scaler
# ------------------------------------------------------------------
def apply_scalers(scalers):
    os.makedirs(NORM_IN, exist_ok=True)
    os.makedirs(NORM_OUT, exist_ok=True)

    # 4.1 输入
    for f in tqdm(glob.glob(os.path.join(IN_DIR, '*.npy')), desc='apply input'):
        arr = np.load(f)                                    # [1,C,64,64,64]
        normed = np.zeros_like(arr, dtype=np.float32)
        for c, name in enumerate(FEATURE_NAMES):
            flat = arr[0, c].reshape(-1, 1)
            if name in {'Dwell_Time', 'Cooling_Duration'}:
                flat = np.where(flat > 0, flat, 1e-8)
                flat = np.log1p(flat)              # 秒
            normed[0, c] = scalers[name].transform(flat).reshape(64, 64, 64)
        np.save(os.path.join(NORM_IN, os.path.basename(f)), normed)

    # 4.2 输出
    for f in tqdm(glob.glob(os.path.join(OUT_DIR, '*.npy')), desc='apply output'):
        arr = np.load(f)                                    # [1,*,64,64,64]
        shape = arr.shape
        flat = arr[0].reshape(-1, 1)
        if not STRAIN_ALREADY_IN_MM:
            flat = flat * GAUGE_LENGTH_MM                  # 换成 mm
        normed = scalers['output'].transform(flat).reshape(shape[1:])
        np.save(os.path.join(NORM_OUT, os.path.basename(f)), normed[None])

    print('✅ 数据预处理完成且保存')

# ------------------------------------------------------------------
# 主入口
# ------------------------------------------------------------------
def main():
    stats = collect_stats()
    scalers = fit_and_save_scalers(stats)
    apply_scalers(scalers)

if __name__ == '__main__':
    main()