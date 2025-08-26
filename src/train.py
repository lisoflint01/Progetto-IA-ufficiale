import random
import numpy as np
import torch
from torchvision import transforms
import json
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

with open("config.json") as f:
        cfg = json.load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def apply_transforms(img, img_size):
    # ridimensiona
    resize = transforms.Resize((img_size, img_size))
    img = resize(img)

    # converti in tensore
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)

    # normalizza
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    img = normalize(img)

    return img

def prepare_dataset(dataset, batch_size, img_size):
    dir_img_train = cfg["train_directory"]
    path_train = Path(dir_img_train)
    
    dir_img_val = cfg["val_directory"]
    path_val = Path(dir_img_val)
    
    if not path_train.is_dir():
        raise FileNotFoundError(f"Cartella train mancante: {path_train}")
    if not path_val.is_dir():
        raise FileNotFoundError(f"Cartella val mancante: {path_val}")
    
    to_tf = lambda img: apply_transforms(img, img_size)
    train_ds = ImageFolder(root=str(path_train), transform=to_tf)
    val_ds   = ImageFolder(root=str(path_val),   transform=to_tf)
    
    