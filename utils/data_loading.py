import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import tifffile
import cv2
import os

def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    elif ext == ".IMG":
        path = os.path.join("../data/imgs", filename)
        with open(path, 'rb') as file:
            raw_data = file.read()
        data = np.frombuffer(raw_data, dtype=np.uint16, count=2048 * 2048)
        data = data.byteswap()
        data = data.reshape((2048, 2048))
        data = (data / 4095.0) * 255.0
        img = cv2.resize(data.astype(np.uint8), (256, 256))
        image = Image.fromarray(img)
        image.save(os.path.join('../data/PNGs/imgs', f'{filename[:len(filename) - 4]}.png'))
        return Image.fromarray(img)
    elif ext == ".tif":
        image = tifffile.imread(filename) 
        img = Image.fromarray(image[0])
        return image
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')
        
        # store image name
        self.image_names = [f"{id}{mask_suffix}" for id in self.ids]

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),
            'name': name
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, img_scale=1):
        super().__init__(images_dir, mask_dir, img_scale, mask_suffix='')

def read_landmarks(landmarks_folder, image_name):
    landmarks_path = os.path.join(landmarks_folder, f"{image_name}.pfs")
    if os.path.exists(landmarks_path):
        with open(landmarks_path, 'r') as file:
            lines = file.readlines()

        landmarks_sets = []
        current_set = []
        for line in lines:
            line = line.strip()
            if line.startswith('{ ') and line.endswith(' },'):
                current_set.append(line)
            elif line.startswith('{ ') and line.endswith(' }'):
                current_set.append(line)
                landmarks_sets.append(current_set)
                current_set = []

        landmarks = []
        for landmarks_set in landmarks_sets:
            set_landmarks = []
            for line in landmarks_set:
                newline = line[2:len(line) - 2]
                coords = [int(float(coord.strip(','))) for coord in newline.strip('{}').split(',') if coord]
                set_landmarks.append(coords)
            landmarks.append(set_landmarks)
        return landmarks
    else:
        return None


def resize_landmarks(landmarks, original_shape, target_shape):
    landmarks = np.array(landmarks)
    landmarks[:, 0] = landmarks[:, 0] * target_shape[1] / original_shape[1]
    landmarks[:, 1] = landmarks[:, 1] * target_shape[0] / original_shape[0]
    return landmarks.astype(int).tolist()


def create_mask(image_shape, points):
    mask = np.zeros((1028, 1028), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    scaled_mask = cv2.resize(mask, image_shape, interpolation=cv2.INTER_NEAREST)
    
    return scaled_mask
    
def preprocess_images():
    landmarks_folder = '../data/SCR/landmarks'
    images_folder= '../data/imgs'
    
    for image_file in os.listdir(images_folder):
        if image_file.endswith(".IMG"):
            image_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(images_folder, image_file)
            original_shape = (2048, 2048)

            try:
                image = load_image(image_file)
                landmarks = read_landmarks(landmarks_folder, image_name)
                right_lung = landmarks[0]
                left_lung = landmarks[1]
                heart = landmarks[2]
                right_clavicle = landmarks[3]
                left_clavicle = landmarks[4]
                
                # create mask for right lung
                if right_lung:
                    # resized_rl = resize_landmarks(right_lung, original_shape, (256, 256))
                    mask = create_mask((256, 256), right_lung)
                    mask_folder = "../data/PNGs/masks/rightlung/"
                    mask_path = os.path.join(mask_folder, f"{image_name}.png")
                    cv2.imwrite(mask_path, mask)
                    print("Saved mask file for image:", image_name, "(right lung).")
                else:
                    print("Could not create mask for image:", image_name, "(right lung).")
                
                # create mask for left lung
                if left_lung:
                    # resized_ll = resize_landmarks(left_lung, original_shape, (256, 256))
                    mask = create_mask((256, 256), left_lung)
                    mask_folder = "../data/PNGs/masks/leftlung/"
                    mask_path = os.path.join(mask_folder, f"{image_name}.png")
                    cv2.imwrite(mask_path, mask)
                    print("Saved mask file for image:", image_name, "(left lung).")
                else:
                    print("Could not create mask for image:", image_name, "(left lung).")
                    
                # create mask for heart
                if heart:
                    # resized_heart = resize_landmarks(heart, original_shape, (256, 256))
                    mask = create_mask((256, 256), heart)
                    mask_folder = "../data/PNGs/masks/heart/"
                    mask_path = os.path.join(mask_folder, f"{image_name}.png")
                    cv2.imwrite(mask_path, mask)
                    print("Saved mask file for image:", image_name, "(heart).")
                else:
                    print("Could not create mask for image:", image_name, "(heart).")
                    
                # create mask for right clavicle
                if right_clavicle:
                    # resized_rc = resize_landmarks(right_clavicle, original_shape, (256, 256))
                    mask = create_mask((256, 256), right_clavicle)
                    mask_folder = "../data/PNGs/masks/rightclavicle/"
                    mask_path = os.path.join(mask_folder, f"{image_name}.png")
                    cv2.imwrite(mask_path, mask)
                    print("Saved mask file for image:", image_name, "(right clavicle).")
                else:
                    print("Could not create mask for image:", image_name, "(right clavicle).")
                    
                # create mask for left clavicle
                if left_clavicle:
                    # resized_lc = resize_landmarks(left_clavicle, original_shape, (256, 256))
                    mask = create_mask((256, 256), left_clavicle)
                    mask_folder = "../data/PNGs/masks/leftclavicle/"
                    mask_path = os.path.join(mask_folder, f"{image_name}.png")
                    cv2.imwrite(mask_path, mask)
                    print("Saved mask file for image:", image_name, "(left clavicle).")
                else:
                    print("Could not create mask for image:", image_name, "(left clavicle).")
            
            except:
                print("Could not load image", image_file)
                
    
    