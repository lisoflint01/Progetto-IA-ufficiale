from pathlib import Path
from augment_train import augment_class_inplace

# config
train_directory     = Path(r"C:\Users\liso4\miniconda3\IA\Progetto2\dataset\train")
augment_percentage  = 0.25
seed                = 42

def main():
    class_dirs = [d for d in train_directory.iterdir() if d.is_dir()]  # itera sottocartelle train / iterate train subfolders
    for class_dir in class_dirs:
        augment_class_inplace(class_dir, augment_percentage=augment_percentage, seed=seed)  # generate synthetic images in place

if __name__ == "__main__":
    main()
