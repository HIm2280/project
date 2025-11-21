#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

def get_loss_function(loss_type='mse'):
    """
    获取回归任务损失函数
    
    Args:
        loss_type: 损失函数类型
            - 'mse': 均方误差损失
            - 'l1': 平均绝对误差损失
            - 'smooth_l1': 平滑L1损失
            
    Returns:
        loss_function: 损失函数实例
    """
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'smooth_l1':
        return nn.SmoothL1Loss()
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")