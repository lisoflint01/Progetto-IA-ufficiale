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


    