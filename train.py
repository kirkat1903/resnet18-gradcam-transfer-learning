"""
Обучение классификатора изображений (transfer learning, ResNet18)
------------------------------------------------------------------
Ожидаемая структура данных (формат ImageFolder):

    data/
        train/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img1.jpg
        val/
            class_a/
                img1.jpg
            class_b/
                img1.jpg

Использование:
    python train.py --data-dir ./data --epochs 15 --batch-size 32
"""

import argparse
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение классификатора с transfer learning")
    parser.add_argument("--data-dir", type=str, required=True,
                         help="Путь к папке с подпапками train/ и val/ (формат ImageFolder)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true",
                         help="Заморозить веса backbone и обучать только последний слой (быстрее, "
                              "подходит для маленьких датасетов)")
    parser.add_argument("--output", type=str, default="best_model.pt")
    return parser.parse_args()


def build_dataloaders(data_dir: str, batch_size: int):
    # Нормализация под ImageNet — веса ResNet предобучены именно на нём
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    data_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }

    image_datasets = {
        split: datasets.ImageFolder(str(Path(data_dir) / split), data_transforms[split])
        for split in ["train", "val"]
    }

    dataloaders = {
        split: DataLoader(image_datasets[split], batch_size=batch_size,
                           shuffle=(split == "train"), num_workers=2)
        for split in ["train", "val"]
    }

    dataset_sizes = {split: len(image_datasets[split]) for split in ["train", "val"]}
    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


def build_model(num_classes: int, freeze_backbone: bool, device: torch.device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Заменяем последний полносвязный слой под своё число классов.
    # Он всегда обучаемый, даже если backbone заморожен.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(device)


def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler,
                 device, num_epochs):
    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase:5s} loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

    print(f"\nЛучшая точность на val: {best_acc:.4f}")
    model.load_state_dict(best_model_weights)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Устройство: {device}")

    dataloaders, dataset_sizes, class_names = build_dataloaders(args.data_dir, args.batch_size)
    print(f"[*] Классы: {class_names}")
    print(f"[*] Размеры датасета: {dataset_sizes}")

    model = build_model(len(class_names), args.freeze_backbone, device)

    criterion = nn.CrossEntropyLoss()
    # Если backbone заморожен — оптимизируем только незамороженные параметры
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    start = time.time()
    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer,
                         scheduler, device, args.epochs)
    print(f"[*] Обучение заняло {(time.time() - start) / 60:.1f} мин")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
    }, args.output)
    print(f"[*] Модель сохранена: {args.output}")


if __name__ == "__main__":
    main()
