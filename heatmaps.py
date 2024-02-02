import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from unet import UNet
import matplotlib.pyplot as plt
import cv2
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

def generate_gradcam(model, image_path, target_class, save_path, attention_save_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert('L')
    input_image = transform(image).unsqueeze(0)
    gradcam = GradCam(model, target_layer='up4')
    attention_map = gradcam.generate(input_image, target_class)

    heatmap = attention_map.squeeze().detach().cpu().numpy()
    np.save(attention_save_path, heatmap)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # plt.figure()
    # plt.tight_layout()
    # plt.imshow(heatmap, cmap='turbo')
    # plt.axis('off')
    # plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True)
    # plt.close()
    
    heatmap_resized = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)
    heatmap_range = heatmap_resized.max() - heatmap_resized.min()

    if heatmap_range < 1e-8:
        heatmap_resized = np.zeros_like(heatmap_resized, dtype=np.uint8)
    else:
        heatmap_resized = (255 * (heatmap_resized - heatmap_resized.min()) / heatmap_range).astype(np.uint8)
        
    input_image_np =  np.array(image)
    
    if len(input_image_np.shape) == 2:
       input_image_np = np.repeat(input_image_np[:, :, np.newaxis], 3, axis=2)

    
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.GaussianBlur(heatmap_colored, (13, 13), 11)
    cv2.imwrite(save_path, heatmap_colored)
    # overlay = overlay.astype(heatmap_colored.dtype)
    overlay = cv2.addWeighted(heatmap_colored, 0.25, input_image_np, 0.75, 0)
    
    cv2.imwrite(save_path.replace('.png', '_overlay.png'), overlay)
#    cv2.imwrite(save_path.replace('.png', '_overlay.png'), overlay)

if __name__ == "__main__":
    image_path = './data/PNGs/imgs/'
    os.makedirs('./gradcam', exist_ok=True)
    os.makedirs('./attention_maps', exist_ok=True)

    target_class = 1

    model = UNet(n_channels=1, n_classes=2)
    model.load_state_dict(torch.load('grad_cam_model.pth', map_location=torch.device(device)))
    model.eval()

    files = os.listdir(image_path)
    for file in files:
        f = os.path.join(image_path, file)
        if os.path.isfile(f):
            save_path = f'./gradcam/{file}'
            attention_save_path = f'./attention_maps/{file.replace(".png", "_attention.npy")}'
            generate_gradcam(model, f, target_class, save_path, attention_save_path)
            print("Saved file.")

print("Finished run.")
