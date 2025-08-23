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

