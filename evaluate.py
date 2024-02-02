import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, save_images=False, wandb_run=None, rnd=1):
    net = net.to(device=device)
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    
    if save_images and wandb_run is not None:
        os.makedirs(f'evals-{rnd}', exist_ok=True)
        
    elif save_images and rnd != 1:
        os.makedirs(f'./predictions/{rnd}', exist_ok=True)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        # table_data = []
        with torch.no_grad():
            for batch in tqdm(dataloader, total=num_val_batches, desc='Test round', unit='batch', leave=False):
                
                image, mask_true, name = batch['image'], batch['mask'], batch['name']

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                # predict the mask
                mask_pred = net(image)
                _, preds = torch.max(mask_pred, dim=1)

                if save_images and rnd != 1:# and wandb_run is not None:
                    image_img = (image[0].cpu().numpy() * 255).astype(np.uint8)
                    mask_true_img = (mask_true[0].cpu().numpy() * 255).astype(np.uint8)
                    mask_pred_img = (preds[0].cpu().numpy() * 255).astype(np.uint8)

                    if image_img.ndim == 2:
                        image_img = np.expand_dims(image, axis=0)

                    image_pil = Image.fromarray(image_img[0])
                    mask_true_pil = Image.fromarray(mask_true_img)
                    mask_pred_pil = Image.fromarray(mask_pred_img)

                    # Save images
                    image_pil.save(os.path.join(f'./predictions/{rnd}', f'{name[0]}_img.png'))
                    mask_true_pil.save(os.path.join(f'./predictions/{rnd}', f'{name[0]}_mask.png'))
                    mask_pred_pil.save(os.path.join(f'./predictions/{rnd}', f'{name[00]}_pred.png'))
                    
                    # Log in WandB
                    # table_data.append([
                    #             f"Image: {name[0]}", 
                    #             wandb.Image(image_pil, caption=f"test image: {name[0]}"),
                    #             wandb.Image(mask_true_pil, caption=f"true mask: {name[0]}"),
                    #             wandb.Image(mask_pred_pil, caption=f"predicted mask: {name[0]}")
                    #         ])
                
                    # table = wandb.Table(data=table_data, columns=["name", "original image", "true mask", "predicted mask"])
                    # wandb.log({f"Test round {rnd}": table})
                    
                    
                if net.n_classes == 1:
                    assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                    mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                    # compute the Dice score
                    dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                else:
                    assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                    # convert to one-hot format
                    mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                    mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                    # compute the Dice score, ignoring background
                    dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    score = dice_score / max(num_val_batches, 1)
    net.train()
    
    return score
