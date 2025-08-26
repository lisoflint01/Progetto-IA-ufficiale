''' this file is for add new "fake images" 
    from the real photo in train, for any class   '''
    
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter 

IMG_EXTS = {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}

# check photo is in correct format
def is_image(path):
    ext = path.suffix.lower()
    if ext in IMG_EXTS:
        return True
    else:
        return False
    
# check or trasform img in rgb
def to_rgb(img):
    rgb = img.mode
    if rgb == "RGB":
        return img
    else:
        new_img = img.convert("RGB")
        return new_img
    
# save images
def save_jpg (img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality = 95, optimize = True)

# Rotazione ±12°
def mod_img_rotate(img):
    img = to_rgb(img)
    angle = random.uniform(-12, 12)
    return img.rotate(angle, resample=Image.BICUBIC, expand=True)

# Flip orizzontale
def mod_img_flip(img):
    img = to_rgb(img)
    return img.transpose(Image.FLIP_LEFT_RIGHT)

# Luminosità ±25%
def mod_img_brightness(img):
    img = to_rgb(img)
    factor = random.uniform(0.75, 1.25)
    return ImageEnhance.Brightness(img).enhance(factor)

# Blur
def mod_img_blur(img):
    img = to_rgb(img)
    blur = random.uniform(0.4, 1.2)
    return img.filter(ImageFilter.GaussianBlur(blur))

# Rumore gaussiano
def mod_img_add_noise(img):
    img = to_rgb(img)
    arr_img = np.array(img).astype(np.float32)
    arr_rum = np.random.normal(0, 8.0, size = arr_img.shape)
    arr_sum = arr_img + arr_rum
    arr = np.clip(arr_sum, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

#list of trasformation
TRANSFORMS = [mod_img_rotate, mod_img_flip, mod_img_brightness, mod_img_blur, mod_img_add_noise]

'''

Attenzione Rileggere da QUI


'''

def list_images(class_dir: Path, only_real: bool = False):
    imgs = [p for p in class_dir.iterdir() if p.is_file() and is_image(p)]
    if only_real:
        imgs = [p for p in imgs if "_aug_" not in p.stem]
    return imgs

def count_real_images(class_dir: Path) -> int:
    return len(list_images(class_dir, only_real=True))

def compute_s(R: int, ratio: float):
    return round((ratio / (1 - ratio)) * R)

def synthesize_images_inplace(class_dir: Path, S: int, seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)

    # usa direttamente le reali
    real_imgs = list_images(class_dir, only_real=True)

    made = 0
    for i in range(S):
        try:
            src = random.choice(real_imgs)
            with Image.open(src) as im:
                im = to_rgb(im)
                
                # inline "pick n transforms": tra 1 e 3, limitato alla lunghezza di TRANSFORMS
                k = random.randint(1, min(3, len(TRANSFORMS)))
                for t in random.sample(TRANSFORMS, k):
                    im = t(im)
                fname = f"{src.stem}_aug_{i:04d}.jpg"
                save_jpg(im, class_dir / fname)
                made += 1
        except Exception:
            continue
    return made

def augment_class_inplace(class_dir: Path, ratio: float = 0.25, seed: int = 42):
    R = count_real_images(class_dir)
    S = compute_s(R, ratio)
    return synthesize_images_inplace(class_dir, S, seed=seed)
