#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from main.dataset import NPYDataset
from main.model import UNet3D
from utilize.loss import get_loss_function
from utilize.optimizer import get_optimizer


# ==================== 超参数配置区 ====================
# 训练基本参数
INITIAL_LR = 1e-4          # 初始学习率
EPOCHS = 100               # 总训练轮数
BATCH_SIZE = 2             # 批次大小 (根据GPU显存调整)
VALID_INTERVAL = 5         # 每N轮验证一次

# 学习率调度与早停
PATIENCE = 15              # 训练耐心: 验证损失不改善的轮数
LR_REDUCE_FACTOR = 0.5     # 学习率降低系数
MIN_LR = 1e-6              # 最低学习率

# 数据加载
NUM_WORKERS = 4            # DataLoader进程数
PIN_MEMORY = True          # 是否锁页内存 (加速GPU训练)

# 设备与保存
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(SAVE_DIR, exist_ok=True)

# 损失函数类型选择
LOSS_TYPE = 'smooth_l1'          # 'mse', 'l1', 'smooth_l1'

# 模型参数
IN_CHANNELS = 6            # 输入通道数
OUT_CHANNELS = 1           # 输出通道数 (回归任务)
# =====================================================


def calculate_gradient_norm(model):
    """
    计算模型梯度范数
    
    Args:
        model: 神经网络模型
        
    Returns:
        total_norm: 所有参数梯度的总范数
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm  ** 0.5
    return total_norm


def train_one_epoch(model, dataloader, loss_fn, optimizer, device, epoch):
    """
    训练一个epoch
    
    Returns:
        avg_loss: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    # 创建进度条
    pbar = tqdm(dataloader, desc=f'Epoch {epoch:3d}/{EPOCHS}', 
                ncols=120, leave=False)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数
        grad_norm = calculate_gradient_norm(model)
        
        # 更新参数
        optimizer.step()
        
        # 统计损失
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        
        # 更新进度条信息
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.6f}',
            'Avg': f'{avg_loss:.6f}',
            'Grad': f'{grad_norm:.6f}',
            'LR': f'{current_lr:.2e}'
        })
    
    return avg_loss


def validate(model, dataloader, loss_fn, device):
    """
    验证模型
    
    Returns:
        avg_val_loss: 平均验证损失
    """
    model.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss


def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path):
    """保存模型检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'lr': optimizer.param_groups[0]['lr']
    }, checkpoint_path)


def main():
    """主训练函数"""
    print(f"开始训练 | 设备: {DEVICE}")
    print(f"超参数: LR={INITIAL_LR}, Batch={BATCH_SIZE}, Epochs={EPOCHS}")
    print(f"检查点保存目录: {SAVE_DIR}")
    
    # 数据加载
    print("\n加载数据集...")
    train_dataset = NPYDataset(train=True)
    val_dataset = NPYDataset(train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    print(f"训练集大小: {len(train_dataset)} | 验证集大小: {len(val_dataset)}")
    
    # 初始化模型、损失函数、优化器
    model = UNet3D(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    loss_fn = get_loss_function(LOSS_TYPE)
    optimizer = get_optimizer(model, lr=INITIAL_LR)
    
    # 学习率调度器 (基于验证损失)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=LR_REDUCE_FACTOR,
        patience=PATIENCE // 2,  # 验证损失不改善的轮数后降低LR
        min_lr=MIN_LR,
    )
    
    # 早停相关变量
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    # 训练主循环
    print("\n开始训练...")
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, DEVICE, epoch)
        
        # 计算epoch耗时
        epoch_time = time.time() - epoch_start
        
        # 打印epoch总结
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.2f}s")
        
        # 验证与模型保存
        if epoch % VALID_INTERVAL == 0 or epoch == EPOCHS:
            val_loss = validate(model, val_loader, loss_fn, DEVICE)
            print(f"{'='*60}")
            print(f"验证结果 - Epoch {epoch:3d} | Val Loss: {val_loss:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                
                best_path = os.path.join(SAVE_DIR, 'best_model.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, best_path)
                print(f"✓ 保存最佳模型 (验证损失: {val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"验证损失未改善 | 耐心计数: {patience_counter}/{PATIENCE}")
            
            # 更新学习率调度器
            scheduler.step(val_loss)
            
            # 早停检查
            if patience_counter >= PATIENCE:
                print(f"{'='*60}")
                print(f"早停触发! 最佳epoch: {best_epoch} | 最佳验证损失: {best_val_loss:.6f}")
                break
            
            # 保存最新模型
            latest_path = os.path.join(SAVE_DIR, 'latest_model.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, latest_path)
            
            print(f"{'='*60}")
    
    # 训练结束
    total_time = time.time() - start_time
    print(f"\n训练完成! 总耗时: {total_time:.2f}s")
    print(f"最佳epoch: {best_epoch} | 最佳验证损失: {best_val_loss:.6f}")
    print(f"模型已保存至: {SAVE_DIR}")


if __name__ == '__main__':
    main()