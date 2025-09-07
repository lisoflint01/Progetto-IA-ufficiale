import random
import numpy as np
import torch
from torchvision import transforms
import json
from pathlib import Path
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models

# set device and get it
def pick_device(mode: str):
    cuda_activable = torch.cuda.is_available()
    mode = mode.lower()
    
    if mode in ["auto", "cuda"]:
        if cuda_activable:
            print(f"[device] {mode} → cuda")
            return torch.device("cuda")
        else:
            print(f"[device] {mode} → cpu")
            return torch.device("cpu")
    else:
        print("[device] using cpu (default)")
        return torch.device("cpu")    

# set seed for random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
   
# apply standard to image 
def apply_transforms(img, img_size):
    
    # resize
    resize = transforms.Resize((img_size, img_size))
    img = resize(img)

    # to tensor
    to_tensor = transforms.ToTensor()
    img = to_tensor(img)

    # normalize
    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
    img = normalize(img)

    return img

# prepare dataset path, directory,
def prepare_dataset(train_dir, val_dir, batch_size, img_size, out_dir):
    
    # path
    path_train = Path(train_dir)
    path_val   = Path(val_dir)

    if not path_train.is_dir():
        raise FileNotFoundError(f"Cartella train mancante: {path_train}")
    if not path_val.is_dir():
        raise FileNotFoundError(f"Cartella val mancante: {path_val}")
    
   # output dir
    out_dir = Path(out_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    # trasforms
    to_tf = lambda img: apply_transforms(img, img_size)

    # dataset & loader
    train_ds = ImageFolder(root=str(path_train), transform=to_tf)
    val_ds   = ImageFolder(root=str(path_val),   transform=to_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, pin_memory=True)

    # labels.json (idx → classe)
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}
    labels_path = out_dir / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)
    print(f"Salvato labels.json in {labels_path}")

    classes = train_ds.classes
    return train_loader, val_loader, classes

# build backbone, replace head, optional freeze
def build_model(num_classes, freeze_backbone):
    
    # load backbone, get input size
    saved_backbone = models.resnet18(pretrained=True)
    num_features = saved_backbone.fc.in_features
    
    # new head
    new_head = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes)
    )
    
    # swap in new head
    saved_backbone.fc = new_head
    
    # freeze backbone, train only head
    if freeze_backbone is True:
        for param in saved_backbone.parameters():
            param.requires_grad = False
        for param in saved_backbone.fc.parameters(): 
            param.requires_grad = True
            
    # total/trainable parameter summary
    total_params = sum(p.numel() for p in saved_backbone.parameters())
    trainable_params = sum(p.numel() for p in saved_backbone.parameters() if p.requires_grad)
    print(f"[params] total: {total_params:,} | trainable: {trainable_params:,}")
    
    return saved_backbone

# train e val, save checkpoints
def train (model, train_loader, val_loader, num_epochs, device, lr):
    model = model.to(device)    # move model to device
    loss_function = torch.nn.CrossEntropyLoss()
    
    # optimize only params trainable
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    # training loop, validation, checkpoints
    for epoch in range(num_epochs):
        print(f"\n[epoch {epoch+1}/{num_epochs}]")
        model.train()
        epoch_train_loss = 0.0

        for images, labels in train_loader:
            # Move images and labels to the same device with model
            images = images.to(device)
            labels = labels.to(device)
        
            optimizer.zero_grad() # Reset gradient before next training step
            logits = model(images) # Initial prediction
            loss = loss_function(logits, labels)
    
            loss.backward() # Compute gradients
            optimizer.step() # Update weights
    
            epoch_train_loss += loss.item()
            
        # AVG loss for epoch
        epoch_train_loss /= len(train_loader)
        history["train_loss"].append(epoch_train_loss)

    
    
        model.eval()
        epoch_val_loss = 0.0
        correct_prediction = 0
        total_prediction = 0
    
        with torch.no_grad(): # faster validation (no gradients)
            for images, labels in val_loader:
                # Move images and labels to the same device with model
                images = images.to(device)
                labels = labels.to(device)
    
                logits = model(images) # Initial prediction
                loss = loss_function(logits, labels)
            
                epoch_val_loss += loss.item()
            
                preds = torch.argmax(logits, dim=1)   # predicted class
                correct_prediction += (preds == labels).sum().item()
                total_prediction += labels.size(0)
            
        # AVG(media) loss and accuracy
        val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = correct_prediction / total_prediction
    
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"[epoch {epoch+1}/{num_epochs}] \t train loss: {epoch_train_loss:.4f} \t val loss: {val_loss:.4f} \t val acc: {val_accuracy:.2%}")

        # Save best model
        if epoch == 0 or val_accuracy > max(history["val_accuracy"][:-1]):
            torch.save(model.state_dict(), "artifacts/model_best.pt")

        # Save last model (sempre)
        torch.save(model.state_dict(), "artifacts/model_last.pt")