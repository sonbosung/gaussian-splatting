#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_test(img1, img2):
    # 전체 이미지에 대한 MSE 계산 (모든 채널과 픽셀에 대해)
    mse = torch.mean((img1 - img2) ** 2)
    
    # MSE가 0인 경우 처리
    if mse == 0:
        return torch.tensor(float('inf'), device=img1.device)
    
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
