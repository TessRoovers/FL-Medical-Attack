import sys
import pdb
import math
import time
import torch
import numpy as np
from PIL import Image
import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from unet import UNet
from pymic.util.parse_config import parse_config
import math


def ssim(img1, img2, window_size=11):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x-window_size//2)**2)/float(2*sigma**2) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel=1):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    img1, img2 = img1 / 255.0, img2 / 255.0
    window = create_window(window_size).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, stride=1, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, stride=1, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, stride=1, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, stride=1, groups=1) - mu2_sq
    sigma_12  = F.conv2d(img1*img2, window, padding=window_size//2, stride=1, groups=1) - mu1_mu2
    
    c1 = (0.01) **2
    c2 = (0.03) **2
    
    ssim_map = ((2*mu1_mu2 +c1)*(2*sigma_12 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
    
    return ssim_map.mean()

def p_selection(p_init, it, n_iters):
    if 10 < it <= 50:       p = p_init / 2
    elif 50 < it <= 200:    p = p_init / 4
    elif 200 < it <= 500:   p = p_init / 8
    elif 500 < it <= 800:   p = p_init / 16
    elif 800 < it <= 1000:  p = p_init / 32
    else:                   p = p_init
    return p


class UniformDistribution(object):
    def __init__(self, config, gt):
        self.n_ex = gt.shape[0]
        self.boundary = torch.tensor(gt.shape[2:])

    def sample(self, size):
        coord = [torch.randint(0, b - size, (self.n_ex, 1)) for b in self.boundary]
        return torch.cat(coord, dim=-1)

    def update(self, idx_fool, loss):
        pass


class AdaptiveDistribution(object):
    def __init__(self, config, gt):
        self.n_ex = gt.shape[0]
        self.boundary = torch.tensor(gt.shape[2:])

        self.attack    = config['attacking']['attack']
        
        self.K         = config['attacking']['k']
        self.mean_lr   = config['attacking']['mean_lr']
        self.std_lr    = config['attacking']['std_lr']
        self.min_std   = config['attacking']['min_std']

        self.init_dis_param(gt)

    def init_dis_param(self, gt):
        foreground = gt[:, 1, :, :]
        boundary  = self.boundary.type(torch.float32) + 1
        samples   = [(torch.nonzero(i) + 1e-5) / boundary for i in foreground]
        
        def inv_sigmoid(x):
            return torch.log(x / (1 - x))
        
        if self.attack == 'IASA':
            print('Initializing Adaptive distribution...')
            self.mean = torch.stack([inv_sigmoid(s).mean(dim=0) for s in samples])
            self.std  = torch.stack([inv_sigmoid(s).std(dim=0)  for s in samples])
            self.std = self.std.clamp(min=self.min_std)
        else:
            self.mean = torch.zeros((self.n_ex, 2))
            self.std  = torch.ones((self.n_ex, 2))

        self.m_grad = torch.zeros_like(self.mean)
        self.s_grad = torch.zeros_like(self.std)
        
        self.sample_num = 0

    def get_m_grad(self, samples):
        return samples / self.std

    def get_s_grad(self, samples):
        return (samples ** 2 - 1) / self.std

    def mean_step(self, alpha, grad, idx_fool):
        self.mean[idx_fool] -= alpha * grad

    def std_step(self, alpha, grad, idx_fool):
        self.std[idx_fool] -= alpha * grad
        self.std = self.std.clamp(min=self.min_std)

    def sample(self, size):
        self.samples = torch.randn((self.n_ex, 2))
        samples = (self.samples * self.std + self.mean).sigmoid()
        coord   = (self.boundary - size) * samples
        return coord.type(torch.int32)
    
    def update(self, idx_fool, loss):
        loss = loss.reshape((-1, 1)).repeat((1, 2)).cpu()
        self.m_grad[idx_fool] += self.get_m_grad(self.samples)[idx_fool] * loss
        self.s_grad[idx_fool] += self.get_s_grad(self.samples)[idx_fool] * loss
        
        self.sample_num += 1
        if self.sample_num % self.K == 0:
            m_grad = self.m_grad[idx_fool] / self.K
            s_grad = self.s_grad[idx_fool] / self.K
            self.mean_step(self.mean_lr, m_grad, idx_fool)
            if self.attack == 'IASA':
                self.std_step(self.std_lr, s_grad, idx_fool)
            
            self.sample_num = 0
            self.m_grad = torch.zeros_like(self.m_grad)
            self.s_grad = torch.zeros_like(self.s_grad)
            
def square_attack(model, config):
    attack = config['attacking']['attack']
    eps = config['attacking']['epsilon']
    p_init = config['attacking']['p_init']
    query_budget = config['attacking']['query_budget']

    print_freq = config['attacking']['print_freq']
    visualize = config['attacking']['visualize']

    min_val, max_val = 0, 1

    print('Evaluating on raw dataset...')
    img, gt = model.get_img_and_gt()
    assert img.min() >= min_val and img.max() <= max_val

    n, c, h, w = img.shape
    n_features = c * h * w

    init_delta = torch.round(torch.rand_like(img)) * (2 * eps) - eps
    perturbed_best = torch.clamp(img + init_delta, min_val, max_val)
    loss_best, asr_best = model.infer_and_get_loss(perturbed_best)

    start_time = time.time()
    for i_iter in range(query_budget):
        idx_fool = loss_best > 0
        if idx_fool.sum() < 1:
            break

        deltas = (perturbed_best - img)[idx_fool]

        p = p_selection(p_init, i_iter, query_budget)
        s = int(round(math.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)

        n = torch.sum(idx_fool)
        ul_corner = D.sample(s)[idx_fool].unsqueeze(1)
        bidx = torch.arange(n).reshape((-1, 1, 1, 1))
        cidx = torch.arange(c).reshape((1, -1, 1, 1)).repeat((n, 1, 1, 1))
        hidx = torch.arange(s).reshape((1, 1, -1, 1)).repeat((n, c, 1, s))
        hidx += ul_corner[:, :, 0: 1].unsqueeze(-1).repeat((1, 1, s, s))
        widx = torch.arange(s).reshape((1, 1, 1, -1)).repeat((n, c, s, 1))
        widx += ul_corner[:, :, 1:].unsqueeze(-1).repeat((1, 1, s, s))

        deltas[bidx, cidx, hidx, widx] = torch.round(torch.rand((n, c, 1, 1))) * (2 * eps) - eps
        perturbed = torch.clamp(img[idx_fool] + deltas, min_val, max_val)
        loss, asr = model.infer_and_get_loss(perturbed)

        better_idx = loss < loss_best[idx_fool]
        loss_best[idx_fool] = loss * better_idx + loss_best[idx_fool] * (~better_idx)
        asr_best[idx_fool] = asr * better_idx + asr_best[idx_fool] * (~better_idx)

        better_idx = better_idx.reshape((n, 1, 1, 1)).cpu()
        perturbed_best[idx_fool] = perturbed * better_idx + perturbed_best[idx_fool] * (~better_idx)

        if i_iter % print_freq == 0:
            print('[Iter {:0>4d}] Avg. Loss: {:.3f}, Avg. ASR: {:.3f}'.format(i_iter, loss_best.mean(), asr_best.mean()))

    avg_ssim = ssim(img, perturbed_best)
    print('Attack end with Avg. Loss: {:.3f}, Avg. ASR: {:.3f}, Avg. SSIM: {:.3f}'.format(loss_best.mean(), asr_best.mean(), avg_ssim))

    elapsed = time.time() - start_time
    if visualize:
        print('Saving perturbed img and predictions ...')
        save_perturbed_img_and_predictions(img, perturbed_best, model)
        deltas_best = perturbed_best - img + eps
        deltas_best[deltas_best > 0] = 1
        print('Saving perturbations ...')
        save_perturbations(deltas_best, model)
    hours = elapsed // 3600
    minutes = (elapsed % 3600) // 60
    print('Elapsed Time: {:.0f} h {:.0f}m'.format(hours, minutes))

def save_perturbed_img_and_predictions(original_img, perturbed_img, model):
    # Save the original and perturbed images along with the model's predictions
    # You may need to adjust this based on your specific model and data representation
    original_img = original_img.squeeze().detach().cpu().numpy()
    perturbed_img = perturbed_img.squeeze().detach().cpu().numpy()
    predictions_original = model.predict(original_img)
    predictions_perturbed = model.predict(perturbed_img)

    # Save or visualize the images and predictions as needed

def save_perturbations(deltas, model):
    # Save the perturbations
    # You may need to adjust this based on your specific model and data representation
    deltas = deltas.squeeze().detach().cpu().numpy()

    # Save or visualize the perturbations as needed

if __name__ == '__main__':
    config = parse_config(str(sys.argv[1]))
    model = UNet(n_channels=1, n_classes=2)  # Update with the appropriate number of channels and classes
    model.load_state_dict(torch.load('your_model.pth'))  # Load your trained model
    model.eval()

    with torch.no_grad():
        square_attack(model, config)
