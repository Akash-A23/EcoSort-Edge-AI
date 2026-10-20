import argparse
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

class EcoSortNet(nn.Module):
    def __init__(self, num_classes: int = 6):
        super().__init__()
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.features.children())[:9])
        self.head = nn.Sequential(
            nn.Conv2d(48, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.Hardswish(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(192, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_dataloaders(data_dir: str, batch_size: int = 32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((96, 96), scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        datasets.ImageFolder(train_dir, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        datasets.ImageFolder(val_dir, transform=val_transform),
        batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, val_loader

def export_to_onnx(model: nn.Module, output_path: str = "model.onnx"):
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, 96, 96)
    try:
        torch.onnx.export(
            model, dummy_input, output_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False
        )
    except TypeError:
        torch.onnx.export(
            model, dummy_input, output_path,
            export_params=True, opset_version=17, do_constant_folding=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/public")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", type=str, default="model.onnx")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(args.data_dir, args.batch_size)

    model = EcoSortNet(num_classes=len(CLASSES)).to(device)
    print(f"Model Parameters: {count_parameters(model):,} / 350,000 limit")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = (correct / total) * 100
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] - Val Accuracy: {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    model.load_state_dict(best_weights)
    export_to_onnx(model, args.output)

if __name__ == "__main__":
    main()
