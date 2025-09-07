from PIL import Image
from pathlib import Path
import torch
from train import pick_device
from train import set_seed
from train import prepare_dataset
from train import build_model

# test selection device
def test_pick_device():
    device = pick_device("auto")
    print("device:", device)
    assert str(device) in ["cpu", "cuda"]
    
# test reproducibility
def test_set_seed():
    set_seed(123)
    a = torch.rand(1)
    set_seed(123)
    b = torch.rand(1)
    print("a:", a, "b:", b)
    assert a == b

# test dataset loading
def test_prepare_dataset():
    # create directoy
    base = Path("tmp_data")
    (base / "train/cat").mkdir(parents=True, exist_ok=True)
    (base / "train/dog").mkdir(parents=True, exist_ok=True)
    (base / "val/cat").mkdir(parents=True, exist_ok=True)
    (base / "val/dog").mkdir(parents=True, exist_ok=True)

    # create image
    img = Image.new("RGB", (64, 64), color="red")
    img.save(base / "train/cat/1.jpg")
    img.save(base / "train/dog/1.jpg")
    img.save(base / "val/cat/1.jpg")
    img.save(base / "val/dog/1.jpg")
    
    train_loader, val_loader, classes = prepare_dataset(
        "tmp_data/train",
        "tmp_data/val",
        batch_size=1,
        img_size=64,
        out_dir="tmp_data/out"
    )

    print("classi:", classes)
    assert set(classes) == {"cat", "dog"}

#test output layer size
def test_build_model():
    model = build_model(num_classes=2, freeze_backbone=False)
    last_layer = model.fc[-1]
    out = last_layer.out_features  # number output units(neuroni)
    print("out_features:", out)
    assert out == 2

if __name__== "__main__":
    test_pick_device()
    test_set_seed()
    test_prepare_dataset()
    test_build_model()
