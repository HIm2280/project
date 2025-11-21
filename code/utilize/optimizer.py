#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import optim

def get_optimizer(model, lr=1e-4, weight_decay=1e-4):
    """
    创建AdamW优化器
    
    Args:
        model: 需要优化的模型
        lr: 初始学习率
        weight_decay: 权重衰减系数 (L2正则化)
        
    Returns:
        optimizer: AdamW优化器实例
    """
    return optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),  # AdamW的默认动量参数
        eps=1e-8
    )