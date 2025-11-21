#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pickle
import glob
import time
import numpy as np
import torch
from tqdm import tqdm

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from main.model import UNet3D

# ==================== 超参数配置区 ====================
# 路径配置
PRE_INPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'pre_input')      # 已预处理的输入数据
PRE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'pre_output')    # 预测结果保存
BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, 'checkpoints', 'best_model.pth')  # 最佳权重
SCALER_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'scalers.pkl')      # 标准化器

# 必须与preprocess_data.py保持一致
STRAIN_ALREADY_IN_MM = False      # 原始应变是否已是mm单位
GAUGE_LENGTH_MM = 1.0             # 标距长度(mm)，仅在STRAIN_ALREADY_IN_MM=False时使用

# 模型参数 (必须与训练时一致)
IN_CHANNELS = 6
OUT_CHANNELS = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 2  # 预测批次大小，可调
# =====================================================


def load_scalers():
    """加载保存的标准化器（仅用于输出逆标准化）"""
    print(f"加载标准化器: {SCALER_PATH}")
    with open(SCALER_PATH, 'rb') as f:
        scalers = pickle.load(f)
    return scalers


def postprocess_output(pred_tensor, scalers):
    """
    对模型输出进行逆标准化和后处理
    
    Args:
        pred_tensor: 模型预测的tensor [batch, 1, 64, 64, 64]
        scalers: 标准化器字典
        
    Returns:
        final_output: numpy数组 [batch, 1, 64, 64, 64]
    """
    # 转为numpy并保持形状 [batch, 1, 64, 64, 64]
    pred_arr = pred_tensor.cpu().numpy()
    
    # 展平为 [batch*4096, 1] 用于逆标准化
    batch_size = pred_arr.shape[0]
    flat = pred_arr.reshape(-1, 1)
    
    # 逆标准化
    denormed = scalers['output'].inverse_transform(flat)
    
    # 单位转换: 如果训练时转为了mm，现在转回原始应变
    if not STRAIN_ALREADY_IN_MM:
        denormed = denormed / GAUGE_LENGTH_MM
    
    # 恢复形状 [batch, 1, 64, 64, 64]
    final_output = denormed.reshape(batch_size, 1, 64, 64, 64)
    final_output = np.maximum(final_output, 0)
    return final_output


def predict_one_batch(model, input_files, scalers):
    """
    预测单个batch
    
    Args:
        model: 加载权重的模型
        input_files: 输入文件路径列表
        scalers: 标准化器（仅用于输出逆标准化）
        
    Returns:
        predictions: 逆标准化后的预测结果列表
    """
    model.eval()
    predictions = []
    
    with torch.no_grad():
        # 批量加载输入数据
        batch_tensors = []
        for f in input_files:
            # 加载已预处理的输入数据 [1, C, 64, 64, 64]
            input_arr = np.load(f)
            # 转为tensor
            input_tensor = torch.from_numpy(input_arr).float().to(DEVICE)
            batch_tensors.append(input_tensor)
        
        # 如果batch_size > 1，合并为一个batch
        if len(batch_tensors) == 1:
            batch_input = batch_tensors[0]
        else:
            batch_input = torch.cat(batch_tensors, dim=0)
        
        # 模型预测 [batch, 1, 64, 64, 64]
        pred_tensor = model(batch_input)
        
        # 后处理 (逆标准化 + 单位转换) [batch, 1, 64, 64, 64]
        batch_output = postprocess_output(pred_tensor, scalers)
        
        # 拆分为单独的文件
        for i in range(batch_output.shape[0]):
            # 每个预测结果保持 [1, 1, 64, 64, 64] 形状
            predictions.append(batch_output[i:i+1])  # 保持batch维度
    
    return predictions


def main():
    """主预测函数"""
    print(f"{'='*60}")
    print("开始预测 (输入数据已预处理模式)")
    print(f"设备: {DEVICE}")
    print(f"最佳权重: {BEST_MODEL_PATH}")
    print(f"输入目录: {PRE_INPUT_DIR}")
    print(f"输出目录: {PRE_OUTPUT_DIR}")
    print(f"{'='*60}")
    
    # 检查路径
    if not os.path.exists(PRE_INPUT_DIR):
        raise RuntimeError(f"❌ 输入目录不存在: {PRE_INPUT_DIR}")
    
    if not os.path.exists(BEST_MODEL_PATH):
        raise RuntimeError(f"❌ 最佳权重文件不存在: {BEST_MODEL_PATH}\n请先运行train.py训练模型")
    
    # 创建输出目录
    os.makedirs(PRE_OUTPUT_DIR, exist_ok=True)
    
    # 加载模型
    print("\n加载模型和权重...")
    model = UNet3D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ 成功加载 epoch {checkpoint['epoch']} 的权重")
    print(f"  当时验证损失: {checkpoint['val_loss']:.6f}")
    
    # 加载标准化器（仅用于输出逆标准化）
    scalers = load_scalers()
    
    # 获取待预测文件列表
    input_files = sorted(glob.glob(os.path.join(PRE_INPUT_DIR, "*.npy")))
    if not input_files:
        raise RuntimeError(f"❌ 未找到任何预测输入文件: {PRE_INPUT_DIR}/*.npy")
    
    print(f"\n找到 {len(input_files)} 个待预测文件")
    print("⚠️  警告: 输入数据必须是已预处理的标准化格式!")
    
    # 批量预测
    print("\n开始预测...")
    start_time = time.time()
    
    # 分批处理以避免显存溢出
    for i in tqdm(range(0, len(input_files), BATCH_SIZE), desc="预测进度"):
        batch_files = input_files[i:i + BATCH_SIZE]
        
        # 预测
        batch_predictions = predict_one_batch(model, batch_files, scalers)
        
        # 保存结果
        for f, pred in zip(batch_files, batch_predictions):
            # 保持与输入相同的文件名，保存为 [1, 1, 64, 64, 64] 形状
            output_path = os.path.join(PRE_OUTPUT_DIR, os.path.basename(f))
            np.save(output_path, pred)
    
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"预测完成! 总耗时: {total_time:.2f}s")
    print(f"结果已保存至: {PRE_OUTPUT_DIR}")
    print(f"{'='*60}")
    
    # 验证第一个输出的范围（调试用）
    if input_files:
        first_output = np.load(os.path.join(PRE_OUTPUT_DIR, os.path.basename(input_files[0])))
        print(f"\n第一个预测结果统计:")
        print(f"  形状: {first_output.shape}")
        print(f"  范围: [{first_output.min():.6f}, {first_output.max():.6f}]")
        print(f"  均值: {first_output.mean():.6f}")
        print(f"  单位: {'mm' if not STRAIN_ALREADY_IN_MM else '原始应变'}")


if __name__ == '__main__':
    main()