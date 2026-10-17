#!/usr/bin/env python3
"""
计算模型参数和推理时间的脚本
分别统计1-shot/2-shot/4-shot下的指标
"""

import os
import torch
import time
import numpy as np
from PIL import Image
import torch.nn as nn
from PromptAD import PromptAD

# 设置环境变量
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_base_and_additional_params(model):
    """分别计算基础CLIP模型参数和额外参数（prompt learner等）"""
    base_params = 0
    additional_params = 0
    
    # 基础CLIP模型参数（包括visual和text encoder）
    for name, param in model.model.named_parameters():
        base_params += param.numel()
    
    # 额外参数（prompt learner的可学习参数）
    # normal_ctx: (n_pro, n_ctx, ctx_dim)
    # abnormal_ctx: (n_pro_ab, n_ctx_ab, ctx_dim)
    for name, param in model.prompt_learner.named_parameters():
        if param.requires_grad:  # 只计算可训练参数
            additional_params += param.numel()
    
    # feature_gallery 和 text_features 是buffer，不算参数
    return base_params, additional_params

def measure_inference_time(model, device, input_size=240, num_runs=100, warmup=10):
    """测量推理时间"""
    model.eval()
    
    # 创建dummy输入图像 (batch_size=1, 3, input_size, input_size)
    dummy_image = torch.randn(1, 3, input_size, input_size).to(device)
    if model.precision == 'fp16':
        dummy_image = dummy_image.half()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_image, 'seg')
    
    # 同步GPU
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # 测量推理时间
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start_time = time.time()
            
            _ = model(dummy_image, 'seg')
            
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            end_time = time.time()
            
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    return avg_time, std_time

def get_model_info(model):
    """获取模型信息"""
    backbone_name = model.model.visual.__class__.__name__
    # 尝试获取输入尺寸
    try:
        input_size = model.transform.transforms[0].size[0] if hasattr(model.transform.transforms[0], 'size') else 240
    except:
        input_size = 240
    
    return backbone_name, input_size

def main():
    # 配置参数
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    print("=" * 80)
    
    # 基础配置 - 使用240x240输入尺寸（ViT-B-16-plus-240模型的默认输入尺寸）
    base_config = {
        'dataset': 'mvtec',
        'class_name': 'carpet',  # 使用任意类别，因为只是计算参数和推理时间
        'img_resize': 240,
        'img_cropsize': 240,
        'resolution': 400,
        'batch_size': 1,
        'backbone': 'ViT-B-16-plus-240',
        'pretrained_dataset': 'laion400m_e32',
        'n_ctx': 4,
        'n_pro': 1,
        'n_ctx_ab': 1,
        'n_pro_ab': 4,
        'device': device,
        'out_size_h': 400,
        'out_size_w': 400,
    }
    
    shots = [1, 2, 4]
    results = []
    
    # 先计算一次参数（参数与shot和输入尺寸都无关）
    print(f"\n{'='*80}")
    print(f"计算模型参数（参数与shot数量和输入尺寸无关）")
    print(f"{'='*80}")
    
    config = base_config.copy()
    config['k_shot'] = 1  # 使用1-shot来计算参数
    
    model = PromptAD(**config)
    model = model.to(device)
    model.eval()
    
    backbone_name, model_input_size = get_model_info(model)
    total_params, trainable_params = count_parameters(model)
    base_params, additional_params = count_base_and_additional_params(model)
    
    # 打印参数详情
    ctx_dim = model.model.ln_final.weight.shape[0]
    n_pro = config['n_pro']
    n_ctx = config['n_ctx']
    n_pro_ab = config['n_pro_ab']
    n_ctx_ab = config['n_ctx_ab']
    normal_ctx_params = n_pro * n_ctx * ctx_dim
    abnormal_ctx_params = n_pro_ab * n_ctx_ab * ctx_dim
    
    print(f"\n模型参数详情:")
    print(f"  CLIP模型: {backbone_name}")
    print(f"  基础参数 (Base): {base_params/1e6:.1f}M")
    print(f"  额外参数 (Additional):")
    print(f"    normal_ctx: {n_pro} x {n_ctx} x {ctx_dim} = {normal_ctx_params/1e6:.4f}M")
    print(f"    abnormal_ctx: {n_pro_ab} x {n_ctx_ab} x {ctx_dim} = {abnormal_ctx_params/1e6:.4f}M")
    print(f"    总计: {additional_params/1e6:.4f}M")
    print(f"  总参数 (Total): {total_params/1e6:.1f}M")
    
    del model
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    
    # 现在测试不同shot的推理时间（输入尺寸固定为240x240）
    for shot in shots:
        print(f"\n{'='*80}")
        print(f"计算 {shot}-shot 配置下的推理时间")
        print(f"{'='*80}")
        
        config = base_config.copy()
        config['k_shot'] = shot
        
        try:
            # 初始化模型
            model = PromptAD(**config)
            model = model.to(device)
            model.eval()
            
            # 获取模型信息
            backbone_name, model_input_size = get_model_info(model)
            
            # 计算参数（参数与输入尺寸和shot数量无关）
            total_params, trainable_params = count_parameters(model)
            base_params, additional_params = count_base_and_additional_params(model)
            
            # 构建feature gallery（模拟训练后的状态）
            # 创建dummy特征来填充gallery
            grid_size = model.grid_size
            embed_dim = model.model.visual.embed_dim
            
            dummy_features1 = torch.randn(shot * grid_size[0] * grid_size[1], embed_dim).to(device)
            dummy_features2 = torch.randn(shot * grid_size[0] * grid_size[1], embed_dim).to(device)
            
            if model.precision == 'fp16':
                dummy_features1 = dummy_features1.half()
                dummy_features2 = dummy_features2.half()
            
            # 重塑为 (batch, tokens, dim) 格式
            dummy_features1 = dummy_features1.reshape(shot, grid_size[0] * grid_size[1], embed_dim)
            dummy_features2 = dummy_features2.reshape(shot, grid_size[0] * grid_size[1], embed_dim)
            
            model.build_image_feature_gallery(dummy_features1, dummy_features2)
            model.build_text_feature_gallery()
            
            # 测量推理时间 - 使用模型配置的输入尺寸
            print(f"正在测量推理时间（{shot}-shot, {model_input_size}x{model_input_size}）...")
            avg_time, std_time = measure_inference_time(model, device, input_size=model_input_size, num_runs=100, warmup=10)
            
            # 保存结果
            result = {
                'shot': shot,
                'backbone': backbone_name,
                'input_size': model_input_size,
                'base_params_M': base_params / 1e6,
                'additional_params_M': additional_params / 1e6,
                'total_params_M': total_params / 1e6,
                'inf_time_ms': avg_time,
                'inf_time_std_ms': std_time,
            }
            results.append(result)
            
            # 打印结果
            print(f"\n{shot}-shot, {model_input_size}x{model_input_size} 结果:")
            print(f"  CLIP模型: {backbone_name}")
            print(f"  输入尺寸: {model_input_size}x{model_input_size}")
            print(f"  推理时间 (平均): {avg_time:.2f} ms")
            print(f"  推理时间 (标准差): {std_time:.2f} ms")
            
            # 清理GPU内存
            del model
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"计算 {shot}-shot 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 打印汇总表格
    print(f"\n{'='*80}")
    print("汇总结果 - 推理时间")
    print(f"{'='*80}")
    print(f"{'Shots':<8} {'CLIP Model':<25} {'Input Size':<12} {'Base Params (M)':<18} {'Additional Params (M)':<22} {'Inf.Time (ms)':<15}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['shot']:<8} {r['backbone']:<25} {r['input_size']}x{r['input_size']:<8} "
              f"{r['base_params_M']:<18.1f} {r['additional_params_M']:<22.4f} {r['inf_time_ms']:<15.2f}")
    
    print(f"\n注意: 模型参数与shot数量和输入尺寸无关")
    print(f"      Base Params: {base_params/1e6:.1f}M")
    print(f"      Additional Params: {additional_params/1e6:.4f}M")
    print(f"      Total Params: {total_params/1e6:.1f}M")
    
    print(f"\n{'='*80}")
    print("计算完成！")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()

