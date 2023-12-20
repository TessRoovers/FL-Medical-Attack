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
from torchvision import utils as vutils, datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import copy
import numpy as np
import pandas as pd
from imutils import paths
from random import randint
from PIL import Image as Image
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

IMG_SIZE = 256
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
EPOCHS = 10

"""
-------------------------------------------------------
                        PARAMETERS
-------------------------------------------------------
 1. Data:                247 images, 247 masks
     - image size:       256 x 256
     - pixel scale:      [0, 255]
     - training set:     198
     - validation set:   0
     - test set:         49
        
 2. Model:               U-Net
 3. Epochs:              10
 4. Evaluation metric:   Dice score
 5. Clients:             3  
"""

dir_img = Path('./data/PNGs/imgs/')
dir_mask = Path('./data/PNGs/masks/')
dir_checkpoint = Path('./checkpoints_noisy/')

images = list(paths.list_images(dir_img))
masks = [str(dir_mask) + os.path.basename(img_path) for img_path in images]

torch.manual_seed(42)
np.random.seed(42)
cuda = torch.device('cuda:0')

# 1. Create dataset
try:
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale=1.0)
except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(dir_img, dir_mask, img_scale=1.0)

# 2. Split into train / test partitions
n_train = 198
n_test = 49
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))

# save indices to file
np.savetxt('train_indices_run7.txt', train_set.indices, fmt="%d")
np.savetxt('test_indices_run7.txt', test_set.indices, fmt="%d")

# 3. Combine datasets into a single dataset and create data loaders
loader_args = dict(batch_size=BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

# store test image names (optional)
test_images = []
for batch in test_loader:
    images, true_masks, name = batch['image'], batch['mask'], batch['name']
    test_images.append(name[0])


def randomize_clients_data(train_size, num_clients, minimum_size):
    assert train_size >= num_clients >= 1
    assert minimum_size * num_clients <= train_size
    data_per_client = []
    max = train_size // num_clients
    
    for i in range(num_clients - 1):
        data_per_client.append(randint(minimum_size,max))
    data_per_client.append(train_size-sum(data_per_client))       
    
    return data_per_client


def create_clients(data, num_clients, noisy_client=None, noise_type=None, initial='clients'):
    assert num_clients >= 1
    client_names = ['{}_{}'.format(initial, i+1) for i in range(num_clients)]
    if noisy_client is not None:
        client_names[noisy_client] = 'evil_client'
    training_data_size = len(data)
    minimum_size = training_data_size // num_clients
    shard_size = randomize_clients_data(training_data_size, num_clients, minimum_size)
    
    shards = [torch.utils.data.Subset(data, range(i * shard_size[i], (i + 1) * shard_size[i])) for i in range(num_clients)]
    
    clients_batch = {}
    for i in range(len(shards)):
        print(f'client {i} : data size: {len(shards[i])}')
        if i == noisy_client:
            noisy_images = []
            for idx in range(len(shards[i])):
                sample = shards[i].dataset[idx]
                image = sample['image']
                mask = sample['mask']
                name = sample['name']
                
                # add gaussian blur
                if noise_type == 'gaussian':
                    noisy_image = gaussian_noise(image)
                    
                # add salt and pepper noise
                else:
                    noisy_image = sp_noise(image)
                    
                noisy_images.append({'image': noisy_image, 'mask': mask, 'name': name})
                
            clients_batch[client_names[i]] = batch_data(noisy_images)
        else:
            clients_batch[client_names[i]] = batch_data(shards[i])
    
    assert(len(shards) == len(client_names))
    return clients_batch


def batch_data(data_shard, bs=BATCH_SIZE):
    trainloader = torch.utils.data.DataLoader(data_shard, batch_size=bs,
                                            shuffle=False, drop_last= True, num_workers=2)
    
    return trainloader


def scale_model_weights(client_models, weight_multiplier_list):
    client_model = client_models[0].state_dict()
    for i in range(len(client_models)):
      for k in client_model.keys():
        client_models[i].state_dict()[k].float()*weight_multiplier_list[i]

    return client_models


def server_aggregate(global_model, client_models):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([
            client_models[i].state_dict()[k].float() for i in range(len(client_models))
            ], 0).mean(0)
    global_model.load_state_dict(global_dict)
    
    return global_model

"""
Noise addition
Use same image, client and noise distribution for:
    * Main model (clean, no noise)
    * Salt and pepper noise
    * Gaussian blur
    * Attention maps
Compare results between models
    * 1 malicious client
"""

def sp_noise(image, salt=0.02, pepper=0.02):
    noisy = image.clone()
    salt_pix = (torch.rand_like(image) < salt)
    pepper_pix = (torch.rand_like(image) < pepper)
    
    noisy = noisy + salt_pix.float() * 0.0
    noisy = noisy - pepper_pix.float() * 255.0
    
    return noisy

def gaussian_noise(image, mean=0, std=0.001):
    noisy = image.clone()
    noisy = noisy + (torch.randn_like(image) * std + mean)
    
    return noisy


def train_model_federated(
        global_model,
        device,
        clients_batched,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    
    experiment = wandb.init(project='Noise_Model', resume='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp, noise_type='gaussian blur', evil_client='1/3')
    )

    logging.info(f'''Starting federated training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Noise type:      gaussian blur
    ''')
    
    wandb.log({"test images": test_images})
    
    # Set up the optimizer, the loss, the learning rate scheduler, and the loss scaling for AMP
    optimizer = optim.RMSprop(global_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if global_model.n_classes > 1 else nn.BCEWithLogitsLoss()

    # Begin federated training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        with tqdm(total=len(clients_batched), desc=f'Epoch {epoch}/{epochs}', unit='client') as pbar:
            for client_name, client_loader in clients_batched.items():
                print("Client:", client_name)
                images_list = []
                
                client_model = copy.deepcopy(global_model).to(device=device)
                client_optimizer = optim.RMSprop(client_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)                
                client_model.train()
                
                table_data = []

                # client_loader = DataLoader(client_data, batch_size=BATCH_SIZE, shuffle=False)
                for i, batch in enumerate(client_loader):
                    print(f"Batch {i+1}/{len(client_loader)}")
                    images, true_masks, name = batch['image'], batch['mask'], batch['name']       
                    images_list.append(name)                    

                    assert images.shape[1] == global_model.n_channels, \
                        f'Network has been defined with {global_model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                    
                    images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    masks_pred = client_model(images)
                    
                    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                        if global_model.n_classes == 1:
                            loss = criterion(masks_pred.squeeze(1), true_masks.float())
                            loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        else:
                            loss = criterion(masks_pred, true_masks)
                            loss += dice_loss(
                                F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, global_model.n_classes).permute(0, 3, 1, 2).float(),
                                multiclass=True
                            )
                    
                    _, preds = torch.max(masks_pred, dim=1)
                    
                    client_optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), gradient_clipping)
                    grad_scaler.step(client_optimizer)
                    grad_scaler.update()
                    epoch_loss += loss.item()
                    
                    with torch.no_grad():
                        image_np = (images[0].cpu().numpy() * 255).astype(np.uint8)
                        mask_np = (true_masks[0].cpu().numpy() * 255).astype(np.uint8)
                        preds = (preds[0].cpu().numpy() * 255).astype(np.uint8)                       
                        
                        if image_np.ndim == 2:
                            image_np = np.expand_dims(image_np, axis=0)
                        
                        image_pil = Image.fromarray(image_np[0])
                        mask_pil = Image.fromarray(mask_np)
                        preds_pil = Image.fromarray(preds)
                        
                        table_data.append([
                            f"Image {name[0]}", 
                            wandb.Image(image_pil, caption=f"image: {name[0]}"),
                            wandb.Image(mask_pil, caption=f"mask: {name[0]}"),
                            wandb.Image(preds_pil, caption=f"pred: {name[0]}")
                        ])
                
                        if 'evil_client' in client_name and epoch == 1:
                            os.makedirs('./perturbations/gb/run7', exist_ok=True)
                            image_pil.save(f'./perturbations/gb/run7/{name[0]}_epoch{epoch}.png')
                
                pbar.update(1)
                pbar.set_postfix(**{'loss (client)': epoch_loss})
                global_model = server_aggregate(global_model, [client_model])
                
                table = wandb.Table(data=table_data, columns=["name", "original image", "true mask", "predicted mask"])
                wandb.log({f"Epoch {epoch} {client_name}": table})
                
                del client_model

        # Aggregate client models at the server
        global_model = server_aggregate(global_model, [global_model])
        test_score = evaluate(global_model, test_loader, device, amp, save_images=True, wandb_run=experiment, rnd=epoch)
        scheduler.step(test_score)

        logging.info('Test Dice score: {}'.format(test_score))
        try:
            experiment.log({
                'Learning rate': optimizer.param_groups[0]['lr'],
                'Test Dice': test_score,
                'Epoch': epoch
            })
        except:
            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = global_model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')



def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--noise-type', type=str, choices=['salt_and_pepper', 'gaussian'], default='gaussian', help='Noise type during training')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    clients_batch = create_clients(train_set, num_clients=3, noisy_client=0)    
    images_set = {}
    images_array = []
    
    for key, value in clients_batch.items():
        client_img = []
        for batch in value:
            client_img.append(batch['name'][0])
        images_set[key] = client_img
        images_array.append(client_img)
        
    
    print("Training images per client:\n", images_set)
    np.save('client_split_run7.npy', images_array)
    
    model.to(device=device)
        
    try:
        train_model_federated(
            global_model=model,
            device=device,
            clients_batched=clients_batch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
        
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model_federated(
            global_model=model,
            device=device,
            clients_batched=clients_batch,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    
    try:
        torch.save(model.state_dict(), f'./models/FL_seg_gb_run7.pth')
    except:
        print("Could not save model to file.")

