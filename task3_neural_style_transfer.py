"""
TASK 3: NEURAL STYLE TRANSFER
Applies artistic styles to photographs using VGG-19.
Libraries: torch, torchvision, Pillow
"""


import copy
import os
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt



DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE  = 512 if torch.cuda.is_available() else 256   # smaller on CPU
NUM_STEPS   = 300
STYLE_WEIGHT   = 1_000_000
CONTENT_WEIGHT = 1

print(f"🖥️  Using device: {DEVICE}")



loader = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
])

def load_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = loader(img).unsqueeze(0)          # add batch dim
    return img.to(DEVICE, torch.float)

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    img = tensor.squeeze(0).cpu().clone()
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)

def show_images(content: torch.Tensor, style: torch.Tensor, output: torch.Tensor):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Content Image", "Style Image", "Stylized Output"]
    for ax, img_tensor, title in zip(axes, [content, style, output], titles):
        ax.imshow(tensor_to_pil(img_tensor))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("style_transfer_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("💾  Result saved to style_transfer_result.png")



class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss   = torch.tensor(0.0)

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    features   = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss   = torch.tensor(0.0)

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(-1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(-1, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std',  std)

    def forward(self, img):
        return (img - self.mean) / self.std



CONTENT_LAYERS = ['conv_4']
STYLE_LAYERS   = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def build_model_and_losses(cnn, content_img, style_img):
    cnn   = copy.deepcopy(cnn)
    norm  = Normalization().to(DEVICE)

    content_losses, style_losses = [], []
    model = nn.Sequential(norm)

    conv_idx = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f'conv_{conv_idx}'
        elif isinstance(layer, nn.ReLU):
            name  = f'relu_{conv_idx}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{conv_idx}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{conv_idx}'
        else:
            continue

        model.add_module(name, layer)

        if name in CONTENT_LAYERS:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{conv_idx}", cl)
            content_losses.append(cl)

        if name in STYLE_LAYERS:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{conv_idx}", sl)
            style_losses.append(sl)

    
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i + 1]

    return model, style_losses, content_losses



def run_style_transfer(content_img, style_img,
                        num_steps=NUM_STEPS,
                        style_weight=STYLE_WEIGHT,
                        content_weight=CONTENT_WEIGHT):

    cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
    model, style_losses, content_losses = build_model_and_losses(cnn, content_img, style_img)

    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_(True)])

    print(f"\n🎨  Running style transfer for {num_steps} steps …")
    step = [0]

    while step[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            s_loss = sum(sl.loss for sl in style_losses)   * style_weight
            c_loss = sum(cl.loss for cl in content_losses) * content_weight
            loss   = s_loss + c_loss
            loss.backward()

            step[0] += 1
            if step[0] % 50 == 0:
                print(f"   Step {step[0]:4d}/{num_steps}  "
                      f"style={s_loss.item():.1f}  content={c_loss.item():.1f}")
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img



def _download_sample_images():
    samples = {
        "content.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/"
                        "Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg/"
                        "402px-Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg",
        "style.jpg"  : "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/"
                        "Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/"
                        "1280px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg",
    }
    for fname, url in samples.items():
        if not os.path.exists(fname):
            print(f"⬇️  Downloading {fname} …")
            urllib.request.urlretrieve(url, fname)
            print(f"✅  Saved {fname}")



if __name__ == "__main__":
    print("=" * 60)
    print("        TASK 3 — NEURAL STYLE TRANSFER")
    print("=" * 60)

    CONTENT_PATH = "content.jpg"   # ← replace with your image
    STYLE_PATH   = "style.jpg"     # ← replace with your style image
    if not os.path.exists(CONTENT_PATH) or not os.path.exists(STYLE_PATH):
        print("\n📥  Sample images not found — downloading …")
        _download_sample_images()

    content_img = load_image(CONTENT_PATH)
    style_img   = load_image(STYLE_PATH)

    print(f"\n📐  Image size : {IMAGE_SIZE}×{IMAGE_SIZE}  |  Steps : {NUM_STEPS}")
    print(f"⚖️  Style weight : {STYLE_WEIGHT}  |  Content weight : {CONTENT_WEIGHT}")

    output = run_style_transfer(content_img, style_img)

    print("\n🖼️  Displaying results …")
    show_images(content_img, style_img, output)

    print("\n" + "=" * 60)
    print("✅  Neural style transfer complete!")
    print("=" * 60)
