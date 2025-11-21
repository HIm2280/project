#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from tqdm import tqdm

########################################
# ① 用户只改这里
PRED_DIR = r"/root/pytorch_project/metal_am_strain_prediction/3D_U_net/dataset/pre_output"   # 预测物理值
TRUTH_DIR= r"/root/pytorch_project/metal_am_strain_prediction/3D_U_net/dataset/pre_real"    # 真实物理值
OUT_FILE = "error_report-2.txt"           # 输出报告
########################################

def mae(pred, truth, eps=1e-8):
    mask = (truth > eps) | (pred > eps)          # 任一非零即算有效
    return np.abs(pred[mask] - truth[mask]).mean()
    

def max_ae(pred, truth, eps=1e-8):
    mask = (truth > eps) | (pred > eps)
    return np.abs(pred[mask] - truth[mask]).max()

def neg_ratio(arr):
    return (arr < 0).mean()

def physical_range(arr):
    return arr.min(), arr.max()

def batch_compare():
    pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npy")))
    if not pred_files:
        print("❌ 未找到任何 *.npy")
        return

    report = []
    for pred_f in tqdm(pred_files, desc="Comparing"):
        basename = os.path.basename(pred_f)                     # case_xxx_strain_mm.npy
        truth_f  = os.path.join(TRUTH_DIR, basename.replace("_strain_mm.npy", ".npy"))

        if not os.path.exists(truth_f):
            tqdm.write(f"⚠️  跳过：无对应 truth 文件 {truth_f}")
            continue

        pred = np.load(pred_f)
        pred = np.where(pred < 0, 0.0001, pred)        # [1,1,64,64,64] mm
        truth = np.load(truth_f)      # [1,1,64,64,64] mm

        mae_val   = mae(pred, truth)
        maxae_val = max_ae(pred, truth)
        neg_rat   = neg_ratio(pred)
        pmin, pmax = physical_range(pred)
        tmin, tmax = physical_range(truth)

        report.append({
            'file': basename,
            'mae': mae_val,
            'maxae': maxae_val,
            'neg_ratio': neg_rat,
            'pred_range': (pmin, pmax),
            'truth_range': (tmin, tmax)
        })

    # 写报告
    with open(OUT_FILE, 'w') as f:
        f.write("file\tMAE(mm)\tMaxAE(mm)\tNegRatio\tPredRange(mm)\tTruthRange(mm)\n")
        for r in report:
            f.write(f"{r['file']}\t{r['mae']:.4f}\t{r['maxae']:.4f}\t{r['neg_ratio']:.3f}\t"
                    f"{r['pred_range'][0]:.4f}-{r['pred_range'][1]:.4f}\t"
                    f"{r['truth_range'][0]:.4f}-{r['truth_range'][1]:.4f}\n")

    # 终端简表
    if report:
        avg_mae = np.mean([r['mae'] for r in report])
        avg_max = np.mean([r['maxae'] for r in report])
        print(f"\n✅ 完成！平均 MAE = {avg_mae:.4f} mm，平均 MaxAE = {avg_max:.4f} mm")
        print(f"📄 详细报告已保存至 {OUT_FILE}")
    else:
        print("❌ 无有效对比文件")

if __name__ == "__main__":
    batch_compare()