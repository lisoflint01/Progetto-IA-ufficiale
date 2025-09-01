import json
from src.train import pick_device
from src.train import prepare_dataset

def main():
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    device = pick_device(cfg.get("memory_cgpu", "auto"))
    train_loader, val_loader, classes = prepare_dataset(cfg)
    print(f"[data] classi: {classes}")

if __name__ == "__main__":
    main()