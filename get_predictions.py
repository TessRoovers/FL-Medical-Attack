import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from unet import UNet
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import defaultdict
from evaluate import evaluate
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from imutils import paths
from random import randint
from PIL import Image as Image
from evaluate import evaluate
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

BATCH_SIZE = 1

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dir_img = Path('./imgs/')
    dir_mask = Path('./masks/')

    images = list(paths.list_images(dir_img))
    masks = [str(dir_mask) + os.path.basename(img_path) for img_path in images]

    torch.manual_seed(42)
    np.random.seed(42)

    torch.backends.cudnn.deterministic = True

    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale=1.0)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale=1.0)

    n_train = 198
    n_test = 49
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=BATCH_SIZE, num_workers=18, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

    test_images = []
    for batch in test_loader:
        images, true_masks, name = batch['image'], batch['mask'], batch['name']
        test_images.append(name[0])
    
    os.makedirs('./predictions', exist_ok=True)

    predictions = defaultdict(dict)
    
    # evaluate clean model
    print("Evaluating clean model...")
    clean = UNet(n_channels=1, n_classes=2)
    clean.load_state_dict(torch.load('./grad_cam_model.pth', map_location=torch.device(device)))
    clean.eval()
    clean_score = evaluate(clean, test_loader, device, amp=False, save_images=True, wandb_run=None, rnd='clean')
    predictions['clean'] = clean_score
    print(f"Finished clean model. Best score: {clean_score}")
    del clean
    
    # perturbed models
    gb_models = ['gb_A', 'gb_B', 'gb_C', 'gb_D', 'gb_E', 'gb_F', 'gb_G']
    sp_models = ['sp_A', 'sp_B', 'sp_C', 'sp_D', 'sp_E', 'sp_F', 'sp_G']
    gcgb_models = ['gcgb_A', 'gcgb_B', 'gcgb_C', 'gcgb_D', 'gcgb_E', 'gcgb_F', 'gcgb_G']
    gcsp_models = ['gcsp_A', 'gcsp_B', 'gcsp_C', 'gcsp_D', 'gcsp_E', 'gcsp_F', 'gcsp_G']
    
    # log best model versions + scores
    gb_best_score = 0
    gb_best_model = None
    gb_avg = 0
    
    sp_best_score = 0
    sp_best_model = None
    sp_avg = 0
    
    gcgb_best_score = 0
    gcgb_best_model = None
    gcgb_avg = 0
    
    gcsp_best_score = 0
    gcsp_best_model = None
    gcsp_avg = 0

    print("Evaluating gaussian noise models...")
    for name in gb_models:
        version = name[-1]
        model = UNet(n_channels=1, n_classes=2)
        model.load_state_dict(torch.load(f'./best/best_{version}_gb.pth', map_location=torch.device(device)))
        model.eval()
        model_score = evaluate(model, test_loader, device, amp=False, save_images=True, wandb_run=None, rnd=name)
        predictions['gb'][version] = model_score
        
        print(f'Model: {version}, \t Score: {model_score}.')
        
        if model_score > gb_best_score:
            gb_best_score = model_score
            gb_best_model = name

        del model
    
    print("Evaluating salt & pepper models...")
    for name in sp_models:
        version = name[-1]
        model = UNet(n_channels=1, n_classes=2)
        model.load_state_dict(torch.load(f'./best/best_{version}_sp.pth', map_location=torch.device(device)))
        model.eval()
        model_score = evaluate(model, test_loader, device, amp=False, save_images=True, wandb_run=None, rnd=name)
        predictions['sp'][version] = model_score
        
        print(f'Model: {version}, \t Score: {model_score}.')
        
        if model_score > sp_best_score:
            sp_best_score = model_score
            sp_best_model = name
            
        del model
    
    print("Evaluating Grad-CAM gaussian noise models...")
    for name in gcgb_models:
        version = name[-1]
        model = UNet(n_channels=1, n_classes=2)
        model.load_state_dict(torch.load(f'./best/best_{version}_gcgb.pth', map_location=torch.device(device)))
        model.eval()
        model_score = evaluate(model, test_loader, device, amp=False, save_images=True, wandb_run=None, rnd=name)
        predictions['gcgb'][version] = model_score
        
        print(f'Model: {version}, \t Score: {model_score}.')
        
        if model_score > gcgb_best_score:
            gcgb_best_score = model_score
            gcgb_best_model = name
            
        del model
    
    print("Evaluating Grad-CAM salt & pepper models...")
    for name in gcsp_models:
        version = name[-1]
        model = UNet(n_channels=1, n_classes=2)
        model.load_state_dict(torch.load(f'./best/best_{version}_gcsp.pth', map_location=torch.device(device)))
        model.eval()
        model_score = evaluate(model, test_loader, device, amp=False, save_images=True, wandb_run=None, rnd=name)
        predictions['gcsp'][version] = model_score
        
        print(f'Model: {version}, \t Score: {model_score}.')
        
        if model_score > gcsp_best_score:
            gcsp_best_score = model_score
            gcsp_best_model = name
            
        del model

    print("---BEST PERFORMANCES---")
    print(f'  * Clean Model: \t\t\t {predictions["clean"]}')
    print(f'  * Gaussian Noise: \t\t\t {gb_best_score} for model {gb_best_model}.')
    print(f'  * Salt & Pepper Noise: \t\t\t {sp_best_score} for model {sp_best_model}.')
    print(f'  * Grad-Cam Gaussian Noise: \t\t\t {gcgb_best_score} for model {gcgb_best_model}.')
    print(f'  * Grad-Cam Salt & Pepper Noise: \t\t\t {gcsp_best_score} for model {gcsp_best_model}.')
    
     
        
        