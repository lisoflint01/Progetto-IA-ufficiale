''' this file is for add new "fake images" 
    from the real photo in train, for any class   '''
    
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter 

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

# check if photo is in correct format
def is_image(path):
    ext = path.suffix.lower()
    if ext in IMG_EXTS:
        return True
    else:
        return False
    
# check or trasform img in rgb
def uniform_rgb(img):
    rgb = img.mode
    if rgb == "RGB":
        return img
    else:
        new_img = img.convert("RGB")
        return new_img
    
# save images
def save_jpg(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality = 95, optimize = True)


# Rotazione moderata ±12°
def rotate_moderate(img):
    img = uniform_rgb(img)
    angle = random.uniform(-12, 12)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

# Flip orizzontale
def flip_horizontal(img):
    img = uniform_rgb(img)
    return img.transpose(Image.FLIP_LEFT_RIGHT)

# Luminosità ±25%
def adjust_brightness(img):
    img = uniform_rgb(img)
    factor = random.uniform(0.75, 1.25)
    return ImageEnhance.Brightness(img).enhance(factor)

# Blur leggero
def blur_light(img):
    img = uniform_rgb(img)
    blur = random.uniform(0.4, 1.2)
    return img.filter(ImageFilter.GaussianBlur(blur))

# Rumore gaussiano basso
def add_noise(img):
    img = uniform_rgb(img)
    arr_img = np.array(img).astype(np.float32)
    arr_rum = np.random.normal(0, 8.0, size = arr_img.shape)
    arr_sum = arr_img + arr_rum
    arr = np.clip(arr_sum, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

#list of trasformation
TRANSFORMS = [rotate_moderate, flip_horizontal, adjust_brightness, blur_light, add_noise]


'''



Attenzione Rileggere da QUI





'''

def list_images(class_dir):
    return [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]

def count_real_images(class_dir):
    images = list_images(class_dir)
    real_images = [f for f in images if "_aug_" not in f.stem]
    return len(real_images)

def pick_n_transforms():
    k = random.randint(1, 3)
    return random.sample(TRANSFORMS, k)

def compute_s(R: int, ratio: float):
    return round((ratio / (1 - ratio)) * R)

def synthesize_images_inplace(class_dir: Path, S: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    real_imgs = [f for f in list_images(class_dir) if "_aug_" not in f.stem]

    made = 0
    for i in range(S):
        try:
            src = random.choice(real_imgs)  # spostato dentro il try
            with Image.open(src) as im:
                im = uniform_rgb(im)
                for t in pick_n_transforms():
                    im = t(im)
                base = src.stem
                fname = f"{base}_aug_{i:04d}.jpg"
                save_jpg(im, class_dir / fname)
                made += 1
        except Exception:
            continue
    return made

def augment_class_inplace(class_dir: Path, ratio: float = 0.25, seed: int = 42):
    R = count_real_images(class_dir)
    S = compute_s(R, ratio)
    return synthesize_images_inplace(class_dir, S, seed=seed)
