import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from imutils import paths
from random import randint
from PIL import Image as Image
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_SIZE = 256
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
EPOCHS = 1000
MEAN=0
THRESHOLD = 0.5

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
     - clean             No noise
     - salt and pepper   1 client - PROB
     - gaussian noise    1 client - MEAN, STD
     - grad cam sp       1 client - PROB, THRESHOLD
     - grad cam gb       1 client - MEAN, STD, THRESHOLD
 3. Epochs:              EPOCHS
 4. Evaluation metric:   Dice score
 5. Clients:             3 
"""

# Local paths
#dir_img = Path('./data/PNGs/imgs/')
#dir_mask = Path('./data/PNGs/masks/')
dir_checkpoint = Path('./checkpoints/')

# Server paths
dir_img = Path('./imgs/')
dir_mask = Path('./masks/')

images = list(paths.list_images(dir_img))
masks = [str(dir_mask) + os.path.basename(img_path) for img_path in images]

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = True

# 1. Create dataset
try:
    dataset = CarvanaDataset(dir_img, dir_mask, img_scale=1.0)
except (AssertionError, RuntimeError, IndexError):
    dataset = BasicDataset(dir_img, dir_mask, img_scale=1.0)

# 2. Split into train / test partitions
n_train = 198
n_test = 49
train_set, test_set = random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(0))

# 3. Combine datasets into a single dataset and create data loaders
loader_args = dict(batch_size=BATCH_SIZE, num_workers=18, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

# store test image names (optional)
test_images = []
for batch in test_loader:
    images, true_masks, name = batch['image'], batch['mask'], batch['name']
    test_images.append(name[0])

# dice score collections
dice_clean = []
dice_gb = []
dice_sp = []
dice_gcgb = []
dice_gcsp = []

# loss values
loss_clean = []
loss_sp = []
loss_gb = []
loss_gcsp = []
loss_gcgb = []


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
    print("Creating clients...")
    # load pretrained UNet model for attention map generation
    trained_model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    trained_model.load_state_dict(torch.load('grad_cam_model.pth', map_location=torch.device(device)))
    
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
        if i == noisy_client:
            noisy_images = []
            for idx in range(len(shards[i])):
                sample = shards[i].dataset[idx]
                image = sample['image']
                mask = sample['mask']
                name = sample['name']
                
                # add gaussian noise
                if noise_type == 'gaussian':
                    noisy_image = gaussian_noise(image, mean=MEAN, std=STD)
                    print("Added gaussian noise image!")
                    
                # add salt and pepper noise
                elif noise_type == 'salt_and_pepper':
                    noisy_image = sp_noise(image, prob=PROB)
                    print("Added salt and pepper noise image!")
                
                # add grad cam salt and pepper noise
                elif noise_type == 'grad_cam_salt_and_pepper':
                    noisy_image = grad_cam_noise(trained_model, image, name, grad_noise='salt_and_pepper')
                    print("Created grad cam SP image!")
                
                # add grad cam gaussian noise
                elif noise_type == 'grad_cam_gaussian':
                    noisy_image = grad_cam_noise(trained_model, image, name, grad_noise='gaussian')
                    print("Created grad cam gaussian image!")
                    
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
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    
    return global_model

class GradCam:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_grad = None
        self.feature_map = None

        self.hook_forward = self.model._modules.get(target_layer).register_forward_hook(self.forward_hook)
        self.hook_backward = self.model._modules.get(target_layer).register_full_backward_hook(self.backward_hook)
        
    def forward_hook(self, module, input, output):
        self.feature_map = output

    def backward_hook(self, module, grad_input, grad_output):
        self.feature_grad = grad_output[0]
    
    def forward(self, x):
        return self.model(x)

    def backward(self, output, target_class):
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

    def generate(self, x, target_class):
        output = self.forward(x)
        self.backward(output, target_class)

        weights = torch.mean(self.feature_grad, dim=(2, 3), keepdim=True)
        cam = weights * self.feature_map
        cam = cam.sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(256, 256), mode='bilinear', align_corners=False)

        
        return cam
    
def grad_cam_noise(trained_model, image, name, grad_noise):
    
    attention_map_np = np.load(f'./attention_maps/{name}_attention.npy', mmap_mode='r')
    attention_map_np = np.copy(attention_map_np)
    attention_map = torch.from_numpy(attention_map_np).unsqueeze(0).to(device=image.device)
    
    if grad_noise == 'gaussian':
        noise = gaussian_noise(image, mean=MEAN, std=STD) # std=STD
    else:
        noise = sp_noise(image, prob=PROB)
    
    mask = (attention_map > THRESHOLD).float()
    noisy_image = image * (1 - mask) + noise * mask
    
    return noisy_image


def sp_noise(image, prob):
    noisy = image.clone()
    
    salt_pix = torch.rand_like(image) < prob/2
    pepper_pix = torch.rand_like(image) < prob/2
       
    noisy[salt_pix] = 1.0
    noisy[pepper_pix] = 0.0
    
    return noisy

def gaussian_noise(image, mean=0, std=0.02):
    noisy = image.clone()
    noisy = noisy + torch.normal(mean, std, size=image.size())
    noisy = torch.clamp(noisy, 0, 1)
    
    return noisy

global_epochs = EPOCHS
local_epochs = 1

def train_model_federated(
        global_model,
        device,
        clients_batched,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = False,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        noise_type = None,
):
    
    # Set up the optimizer, the loss, the learning rate scheduler, and the loss scaling for AMP
    optimizer = optim.RMSprop(global_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if global_model.n_classes > 1 else nn.BCEWithLogitsLoss()
    
    # save best score model
    best_clean = 0
    best_sp = 0
    best_gb = 0
    best_gcsp = 0
    best_gcgb = 0
    
    best_ep_clean = None
    best_ep_sp = None
    best_ep_gb = None
    best_ep_gcsp = None
    best_ep_gcgb = None

    # Begin federated training
    for epoch in range(1, global_epochs + 1):
        client_models = [] 
        global_model.train()
        epoch_loss = 0
                
        with tqdm(total=len(clients_batched), desc=f'Epoch {epoch}/{epochs}', unit='client') as pbar:
            for client_name, client_loader in clients_batched.items():
                
                images_list = []
                client_model = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear).to(device=device)
                client_model.load_state_dict(global_model.state_dict())
                client_optimizer = optim.RMSprop(client_model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)                
                client_model.train()                
                pbar_batch = tqdm(total=len(client_loader), desc=f' * Client: {client_name}', unit='batch', leave=False)
                
                for i, batch in enumerate(client_loader):
                    images, true_masks, name = batch['image'], batch['mask'], batch['name']       
                    images_list.append(name)                    

                    assert images.shape[1] == global_model.n_channels, \
                        f'Network has been defined with {global_model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                    
                    # print("Image shape: ", images.shape)
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
                    
                    client_optimizer.zero_grad()
                    grad_scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(client_model.parameters(), gradient_clipping)
                    grad_scaler.step(client_optimizer)
                    grad_scaler.update()
                    epoch_loss += loss.item()
                    
                    with torch.no_grad():
                        image_np = (images[0].cpu().numpy() * 255).astype(np.uint8)
                        # mask_np = (true_masks[0].cpu().numpy() * 255).astype(np.uint8)
                        preds = (preds[0].cpu().numpy() * 255).astype(np.uint8)                       
                        
                        if image_np.ndim == 2:
                            image_np = np.expand_dims(image_np, axis=0)
                        
                        image_pil = Image.fromarray(image_np[0])
                
                        if 'evil_client' in client_name and epoch == 1 and noise_type is not None:
                            print("Attempting to save noisy image...")
                            try:
                                if noise_type == 'gaussian':
                                    os.makedirs('./gb/', exist_ok=True)
                                    image_pil.save(f'./gb/{name[0]}_{PARAM}.png')
                                elif noise_type == 'salt_and_pepper':
                                    os.makedirs('./sp/', exist_ok=True)
                                    image_pil.save(f'./sp/{name[0]}_{PARAM}.png')
                                elif noise_type == 'grad_cam_salt_and_pepper':
                                    os.makedirs('./gcsp/', exist_ok=True)
                                    image_pil.save(f'./gcsp/{name[0]}_{PARAM}.png')
                                elif noise_type == 'grad_cam_gaussian':
                                    os.makedirs('./gcgb/', exist_ok=True)
                                    image_pil.save(f'./gcgb/{name[0]}_{PARAM}.png')
                            except:
                                print(f'Could not save noisy image: {name[0]}.')
                    
                                         
                    pbar_batch.update(1)
                    
                pbar_batch.close()
                
                pbar.update(1)
                pbar.set_postfix(**{'loss (client)': epoch_loss})
                client_models.append(client_model) 
                
                del client_model 

                global_model = server_aggregate(global_model, client_models)
                
            
        pbar.close()
        client_models = scale_model_weights(client_models, [1/3, 1/3, 1/3])
        global_model = server_aggregate(global_model, client_models)
        
        global_model.eval()
        test_score = evaluate(global_model, test_loader, device, amp, save_images=False, wandb_run=None, rnd=epoch)
        print(f'Test score after epoch {epoch}: {test_score}.')
        
        if noise_type == 'gaussian':
            loss_gb.append(epoch_loss)
        elif noise_type == 'salt_and_pepper':
            loss_sp.append(epoch_loss)
        elif noise_type == 'grad_cam_salt_and_pepper':
            loss_gcsp.append(epoch_loss)
        elif noise_type == 'grad_cam_gaussian':
            loss_gcgb.append(epoch_loss)
        else:
            loss_clean.append(epoch_loss)
        
        if noise_type == 'gaussian':
            if test_score > best_gb:
                best_gb = test_score
                best_ep_gb = epoch
                print(f'New best score for {PARAM} with gaussian noise found at epoch {best_ep_gb}: \t {best_gb}.')
                torch.save(global_model.state_dict(), f'./best/best_{PARAM}_gb.pth')
        elif noise_type == 'salt_and_pepper':
            if test_score > best_sp:
                best_sp = test_score
                best_ep_sp = epoch
                print(f'New best score for {PARAM} with salt and pepper noise found at epoch {best_ep_sp}: \t {best_sp}.')
                torch.save(global_model.state_dict(), f'./best/best_{PARAM}_sp.pth')
        elif noise_type == 'grad_cam_salt_and_pepper':
            if test_score > best_gcsp:
                best_gcsp = test_score
                best_ep_gcsp = epoch
                print(f'New best score for {PARAM} with grad cam salt and pepper noise found at epoch {best_ep_gcsp}: \t {best_gcsp}.')
                torch.save(global_model.state_dict(), f'./best/best_{PARAM}_gcsp.pth')
        elif noise_type == 'grad_cam_gaussian':
            if test_score > best_gcgb:
                best_gcgb = test_score
                best_ep_gcgb = epoch
                print(f'New best score for {PARAM} with grad cam gaussian noise found at epoch {best_ep_gcgb}: \t {best_gcgb}.')
                torch.save(global_model.state_dict(), f'./best/best_{PARAM}_gcgb.pth')
        else:
            if test_score > best_clean:
                best_clean = test_score
                best_ep_clean = epoch
                print(f'New best score for {PARAM} without noise found at epoch {best_ep_clean}: \t {best_clean}.')
                torch.save(global_model.state_dict(), f'./best/best_{PARAM}_clean.pth')
        
        # # save dice score
        if noise_type == 'gaussian':
            dice_gb.append(test_score)
        elif noise_type == 'salt_and_pepper':
            dice_sp.append(test_score)
        elif noise_type == 'grad_cam_salt_and_pepper':
            dice_gcsp.append(test_score)
        elif noise_type == 'grad_cam_gaussian':
            dice_gcgb.append(test_score)
        else:
            dice_clean.append(test_score)
            
        scheduler.step(test_score)
        
        save_checkpoint = False
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = global_model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            if noise_type == 'gaussian':
                torch.save(state_dict, str(dir_checkpoint / 'gb_checkpoint_epoch{}.pth'.format(epoch)))
            elif noise_type == 'salt_and_pepper':
                torch.save(state_dict, str(dir_checkpoint / 'sp_checkpoint_epoch{}.pth'.format(epoch)))
            else:
                torch.save(state_dict, str(dir_checkpoint / 'clean_checkpoint_epoch{}.pth'.format(epoch)))
    
    # save best model score + epoch
    # if noise_type == 'salt_and_pepper':
    #     best_dict['sp'] = [best_sp, best_ep_sp]
    #     print(f'Best model for {PARAM} for sp noise found at epoch {best_ep_sp}, with score: \t {best_sp}.')
    # elif noise_type == 'gaussian':
    #     best_dict['gb'] = [best_gb, best_ep_gb]
    #     print(f'Best model for {PARAM} for gaussian noise found at epoch {best_ep_gb}, with score: \t {best_gb}.')
    # else:
    #     best_dict['clean'] = [best_clean, best_ep_clean]
    #     print(f'Best model for {PARAM} without noise found at epoch {best_ep_clean}, with score: \t {best_clean}.')
        
    

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
    parser.add_argument('--noise-type', type=str, choices=['salt_and_pepper', 'gaussian', 'grad_cam_salt_and_pepper', 'grad_cam_gaussian'], default=None, help='Noise type during training')
    parser.add_argument('--prob', type=float, default=0.005, help='Probability parameter')
    parser.add_argument('--std', type=float, default=0.01, help='Standard deviation parameter')
    parser.add_argument('--param', type=str, default='A', help='Parameter identifier')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    print("Arguments retrieved...")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("--> Using device:", device)
    logging.info(f'Using device {device}')

    PROB = args.prob
    STD = args.std
    PARAM = args.param
    
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    print("Creating clean model...")
    model_clean = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model_clean = model_clean.to(device=device, memory_format=torch.channels_last)
    print("...finished.\n")
    
    print("Creating salt and pepper model...")
    model_sp = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model_sp = model_sp.to(device=device, memory_format=torch.channels_last)
    print("...finished.\n")
    
    print("Creating gaussian noise model...")
    model_gb = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model_gb = model_gb.to(device=device, memory_format=torch.channels_last)
    print("...finished.\n")
    
    print("Creating Grad-CAM gaussian noise model...")
    model_gcgb = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model_gcgb = model_gcgb.to(device=device, memory_format=torch.channels_last)
    print("...finished.\n")
    
    print("Creating Grad-CAM salt and pepper model...")
    model_gcsp = UNet(n_channels=1, n_classes=args.classes, bilinear=args.bilinear)
    model_gcsp = model_gcsp.to(device=device, memory_format=torch.channels_last)
    print("...finished.\n")
    
    # create clients based on model
    random_client = randint(0, 2)
    print("Creating all clients...")
    print(f"Adversarial client selected as client: {random_client + 1}")
    
    clients_clean = create_clients(train_set, num_clients=3, noisy_client=None, noise_type=None)
    clients_gb = create_clients(train_set, num_clients=3, noisy_client=random_client, noise_type='gaussian')
    clients_sp = create_clients(train_set, num_clients=3, noisy_client=random_client, noise_type='salt_and_pepper')
    clients_gcgb = create_clients(train_set, num_clients=3, noisy_client=random_client, noise_type='grad_cam_gaussian')
    clients_gcsp = create_clients(train_set, num_clients=3, noisy_client=random_client, noise_type='grad_cam_salt_and_pepper')
    
    os.makedirs('./best/', exist_ok=True)
    print("Created all clients!")
    
    try:
        # grad cam gaussian model training
        print("Training model: grad cam gaussian noise...")
        train_model_federated(
            global_model=model_gcgb,
            device=device,
            clients_batched=clients_gcgb,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            noise_type='grad_cam_gaussian'
        )
        print("...Finished training model: grad cam gaussian.")
        torch.cuda.empty_cache()
        
        # grad cam salt and pepper model training
        print("Training model: grad cam salt and pepper...")
        train_model_federated(
            global_model=model_gcsp,
            device=device,
            clients_batched=clients_gcsp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            noise_type='grad_cam_salt_and_pepper'
        )
        print("...Finished training model: grad cam salt and pepper.")
        torch.cuda.empty_cache()
        
        # salt and pepper model training
        print("Training model: salt and pepper")
        train_model_federated(
            global_model=model_sp,
            device=device,
            clients_batched=clients_sp,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            noise_type='salt_and_pepper'
        )
        print("...Finished training model: salt and pepper")
        torch.cuda.empty_cache()
        
        # clean model training
        print("Training model: clean")
        train_model_federated(
            global_model=model_clean,
            device=device,
            clients_batched=clients_clean,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            noise_type=None
        )
        print("...Finished training model: clean")
        torch.cuda.empty_cache()
        
        # gaussian noise model training
        print("Training model: gaussian noise")
        train_model_federated(
            global_model=model_gb,
            device=device,
            clients_batched=clients_gb,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp,
            noise_type='gaussian'
        )
        print("...Finished training model: gaussian noise")
        torch.cuda.empty_cache()

    
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                     'Enabling checkpointing to reduce memory usage, but this slows down training. '
                     'Consider enabling AMP (--amp) for fast and memory-efficient training')
        print("Model training not completed. Ending training...")
        torch.cuda.empty_cache()
        # model_clean.enable_checkpointing()
        # model_gb.enable_checkpointing()
        # model_sp.enable_checkpointing()

        # train_model_federated(
        #     global_model=model_clean,
        #     device=device,
        #     clients_batched=clients_clean,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     learning_rate=args.lr,
        #     img_scale=args.scale,
        #     val_percent=args.val / 100,
        #     amp=args.amp,
        #     noise_type=None
        # )
        # torch.cuda.empty_cache()

        # train_model_federated(
        #     global_model=model_gb,
        #     device=device,
        #     clients_batched=clients_gb,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     learning_rate=args.lr,
        #     img_scale=args.scale,
        #     val_percent=args.val / 100,
        #     amp=args.amp,
        #     noise_type='gaussian'
        # )
        # torch.cuda.empty_cache()

        # train_model_federated(
        #     global_model=model_sp,
        #     device=device,
        #     clients_batched=clients_sp,
        #     epochs=args.epochs,
        #     batch_size=args.batch_size,
        #     learning_rate=args.lr,
        #     img_scale=args.scale,
        #     val_percent=args.val / 100,
        #     amp=args.amp,
        #     noise_type='salt_and_pepper'
        # )
        # torch.cuda.empty_cache()
    
    try:
        os.makedirs('./models/', exist_ok=True)
        torch.save(model_clean.state_dict(), f'./models/clean_{PARAM}.pth')
        torch.save(model_gb.state_dict(), f'./models/gb_{PARAM}.pth')
        torch.save(model_sp.state_dict(), f'./models/sp_{PARAM}.pth')
        torch.save(model_gcgb.state_dict(), f'./models/gcgb_{PARAM}.pth')
        torch.save(model_gcsp.state_dict(), f'./models/gcsp_{PARAM}.pth')
    except:
        print("Could not save models to file.")
        
    
    if len(loss_gb) > 0 and len(loss_sp) > 0 and len(loss_clean) > 0 and len(loss_gcgb) > 0 and len(loss_gcsp) > 0:
        os.makedirs('./loss/', exist_ok=True)
        epochs = range(1, EPOCHS + 1)
        loss_scores = np.array([epochs, [loss.cpu().item() for loss in loss_clean], [loss.cpu().item() for loss in loss_gb], [loss.cpu().item() for loss in loss_sp], [loss.cpu().item() for loss in loss_gcgb], [loss.cpu().item() for loss in loss_gcsp]]).T
        np.savetxt(f'./loss/loss_{PARAM}.csv', loss_scores, delimiter=', ', header='Epoch, Clean Model, Gaussian, Salt and Pepper, Grad Cam Gaussian, Grad Cam Salt and Pepper', comments='')
        
    
    if len(dice_gb) > 0 and len(dice_sp) > 0 and len(dice_clean) > 0 and len(dice_gcgb) > 0 and len(dice_gcsp) > 0:
        os.makedirs('./dice/', exist_ok=True)
        epochs = range(1, EPOCHS + 1)
        dice_scores = np.array([epochs, [score.cpu().item() for score in dice_clean], [score.cpu().item() for score in dice_gb], [score.cpu().item() for score in dice_sp], [score.cpu().item() for score in dice_gcgb], [score.cpu().item() for score in dice_gcsp]]).T
        np.savetxt(f'./dice/dice_{PARAM}.csv', dice_scores, delimiter=', ', header='Epoch, Clean Model, Gaussian, Salt and Pepper, Grad Cam Gaussian, Grad Cam Salt and Pepper', comments='')
        
        # regular scale
        plt.plot(epochs, [score.cpu().item() for score in dice_clean], label='Clean Model')
        plt.plot(epochs, [score.cpu().item() for score in dice_gb], label=f'Gaussian: $\mu=0$, $\sigma={str(STD)}$')
        plt.plot(epochs, [score.cpu().item() for score in dice_sp], label=f'Salt and Pepper: $p={str(PROB)}$')
        plt.plot(epochs, [score.cpu().item() for score in dice_gcgb], label=f'Grad Cam Gaussian: $\mu=0$, $\sigma={str(STD)}$') 
        plt.plot(epochs, [score.cpu().item() for score in dice_gcsp], label=f'Grad Cam Salt and Pepper: $p={str(PROB)}$')
        plt.xlabel('Epoch')
        plt.ylabel('Test Dice Score')
        plt.legend()
        plt.title('Model Performances (3 clients, 1 malicious)')
        plt.savefig(f'./dice/dice_{PARAM}.png')
        

        