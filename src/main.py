from pathlib import Path
from augment_train import count_real_images, compute_s, synthesize_images_inplace

# config value
train_directory = Path(r"C:\Users\liso4\miniconda3\IA\Progetto2\dataset\train")
augment_percentage = 0.25
seed = 42

# main
def main():
    class_dirs = [d for d in train_directory.iterdir() if d.is_dir()] # iterate directory in train

    for class_dir in class_dirs:
        R = count_real_images(class_dir)
        S = compute_s(R, augment_percentage)
        synthesize_images_inplace(class_dir, S, seed = seed)

if __name__ == "__main__":
    main()
