"""
scripts/classifier.py
ImageNet-1K image classifier using pretrained ResNet (torchvision).

The pretrained ResNet-50 already achieves ~76% top-1 accuracy on ImageNet.
No training is needed for inference — just use the weights out of the box.

Inference (used by app.py):
  from scripts.classifier import predict_image
  predicted = predict_image(image_tensor=t, device='cpu')

Optional fine-tuning on your own ImageNet-format data:
  python scripts/classifier.py --train --data path/to/imagenet
  (expects path/to/imagenet/train/<class>/ and path/to/imagenet/val/<class>/)
"""
import argparse
import copy
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights

# ── ImageNet class labels from torchvision (no JSON file needed) ──────────────
_WEIGHTS = ResNet50_Weights.DEFAULT
CLASSES = _WEIGHTS.meta["categories"]          # list of 1000 class name strings

# Input size expected by ResNet
INPUT_SIZE = 224


# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes: int = 1000, device: str = 'cpu',
                arch: str = 'resnet50', pretrained: bool = True):
    """
    Build a pretrained ResNet model for ImageNet inference.

    When num_classes == 1000 and pretrained=True the original ImageNet head
    is kept intact — no replacement needed.

    Args:
        num_classes: 1000 for full ImageNet, or fewer for fine-tuning a subset.
        device: 'cpu', 'cuda', or 'mps'.
        arch: 'resnet18' or 'resnet50'.
        pretrained: Load pretrained ImageNet weights.
    """
    if arch == 'resnet50':
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
    else:
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)

    # Only replace the head when fine-tuning on a custom number of classes
    if num_classes != 1000:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )
    return model.to(device)


# ── Inference ──────────────────────────────────────────────────────────────────
def predict_image(image_tensor: torch.Tensor,
                  device: str = 'cpu',
                  arch: str = 'resnet50',
                  weights_path: str = None) -> str:
    """
    Predict the ImageNet class of a single image using Test-Time Augmentation.

    TTA averages predictions over 5 different crops/flips of the input image,
    which typically improves accuracy by ~0.5–1% compared to a single pass.

    Args:
        image_tensor: Pre-transformed image tensor of shape (C, H, W).
        device: 'cpu', 'cuda', or 'mps'.
        arch: Architecture used ('resnet18' or 'resnet50').
        weights_path: Optional path to fine-tuned weights (.pth).
                      If None, uses pretrained ImageNet weights directly.

    Returns:
        str: Predicted class label with confidence.
    """
    if weights_path and os.path.exists(weights_path):
        # Load fine-tuned weights (may have custom number of classes)
        checkpoint = torch.load(weights_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 1000)
        class_names = checkpoint.get('class_names', CLASSES)
        model = build_model(num_classes=num_classes, device=device,
                            arch=arch, pretrained=False)
        model.load_state_dict(checkpoint['state_dict']
                              if 'state_dict' in checkpoint else checkpoint)
    else:
        # Use pretrained ImageNet weights directly — no .pth file needed
        model = build_model(num_classes=1000, device=device,
                            arch=arch, pretrained=True)
        class_names = CLASSES

    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)   # (1, C, H, W)
    image_tensor = image_tensor.to(device)

    # Test-Time Augmentation: 5 augmented views → averaged logits
    tta_transforms = [
        transforms.Compose([]),                                           # original
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),    # h-flip
        transforms.Compose([transforms.RandomCrop(INPUT_SIZE, padding=16)]),   # crop
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomCrop(INPUT_SIZE, padding=16)]),   # flip+crop
        transforms.Compose([transforms.CenterCrop(int(INPUT_SIZE * 0.9)),
                            transforms.Resize((INPUT_SIZE, INPUT_SIZE))]),      # center-crop
    ]

    logits_sum = None
    with torch.no_grad():
        for t in tta_transforms:
            aug = t(image_tensor)
            out = model(aug)
            logits_sum = out if logits_sum is None else logits_sum + out

    probabilities = torch.softmax(logits_sum, dim=1)[0]
    pred = probabilities.argmax().item()
    confidence = probabilities[pred].item() * 100

    # Show top-3 alternatives for richer output
    top3 = torch.topk(probabilities, 3)
    top3_str = ", ".join(
        f"{class_names[i]} ({probabilities[i].item()*100:.1f}%)"
        for i in top3.indices
    )
    return f"{class_names[pred]} ({confidence:.1f}% confidence) | Top-3: {top3_str}"


def resnet_top5(image_tensor: torch.Tensor,
                device: str = 'cpu',
                arch: str = 'resnet50',
                weights_path: str = None) -> list:
    """
    Return top-5 ImageNet predictions from ResNet as a list of dicts.
    Each dict: {'Rank': int, 'Class': str, 'Confidence': str}
    """
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 1000)
        class_names = checkpoint.get('class_names', CLASSES)
        model = build_model(num_classes=num_classes, device=device,
                            arch=arch, pretrained=False)
        model.load_state_dict(checkpoint['state_dict']
                              if 'state_dict' in checkpoint else checkpoint)
    else:
        model = build_model(num_classes=1000, device=device,
                            arch=arch, pretrained=True)
        class_names = CLASSES

    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    tta_transforms = [
        transforms.Compose([]),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.RandomCrop(INPUT_SIZE, padding=16)]),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0),
                            transforms.RandomCrop(INPUT_SIZE, padding=16)]),
        transforms.Compose([transforms.CenterCrop(int(INPUT_SIZE * 0.9)),
                            transforms.Resize((INPUT_SIZE, INPUT_SIZE))]),
    ]
    logits_sum = None
    with torch.no_grad():
        for t in tta_transforms:
            out = model(t(image_tensor))
            logits_sum = out if logits_sum is None else logits_sum + out

    probs = torch.softmax(logits_sum, dim=1)[0]
    top5 = torch.topk(probs, 5)
    return [
        {'Rank': i + 1,
         'Class': class_names[top5.indices[i].item()],
         'Confidence': f"{top5.values[i].item() * 100:.2f}%"}
        for i in range(5)
    ]


# ── YOLOv8 Classification (ImageNet-pretrained) ───────────────────────────────
def predict_yolo_cls(pil_image, device: str = 'cpu',
                     model_name: str = 'yolov8n-cls.pt') -> str:
    """
    Classify an image using YOLOv8's ImageNet-pretrained classification model.

    The model is downloaded automatically by ultralytics on first use (~6 MB).

    Args:
        pil_image: PIL Image (RGB).
        device: 'cpu', 'cuda', or 'mps'.
        model_name: 'yolov8n-cls.pt' (fast) or 'yolov8s-cls.pt' (more accurate).

    Returns:
        str: Top-1 class label with confidence + top-3 summary.
    """
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(model_name)
    results = model.predict(source=pil_image, device=device, verbose=False)
    result  = results[0]

    # probs is a Probs object; .top5 gives indices, .top5conf gives confidences
    top5_idx  = result.probs.top5          # list of 5 class indices
    top5_conf = result.probs.top5conf      # tensor of 5 confidences
    names     = result.names               # dict {idx: class_name}

    top1_name = names[top5_idx[0]]
    top1_conf = float(top5_conf[0]) * 100

    top3_str = ", ".join(
        f"{names[top5_idx[i]]} ({float(top5_conf[i])*100:.1f}%)"
        for i in range(min(3, len(top5_idx)))
    )
    return f"{top1_name} ({top1_conf:.1f}% confidence) | Top-3: {top3_str}"


def yolo_cls_top5(pil_image, device: str = 'cpu',
                  model_name: str = 'yolov8n-cls.pt') -> list:
    """
    Return top-5 ImageNet predictions from YOLOv8-cls as a list of dicts.
    Each dict: {'Rank': int, 'Class': str, 'Confidence': str}
    """
    from ultralytics import YOLO
    model = YOLO(model_name)
    results = model.predict(source=pil_image, device=device, verbose=False)
    result  = results[0]
    top5_idx  = result.probs.top5
    top5_conf = result.probs.top5conf
    names     = result.names
    return [
        {'Rank': i + 1,
         'Class': names[top5_idx[i]],
         'Confidence': f"{float(top5_conf[i]) * 100:.2f}%"}
        for i in range(min(5, len(top5_idx)))
    ]


# ── Training (optional fine-tuning on local ImageNet data) ────────────────────
def train(data_dir: str = 'data/imagenet',
          epochs: int = 10,
          save_path: str = 'models/imagenet_finetuned.pth',
          arch: str = 'resnet50'):
    """
    Fine-tune ResNet on a local ImageNet-format dataset.

    Expects:
        data_dir/train/<class_name>/  — training images
        data_dir/val/<class_name>/    — validation images

    Args:
        data_dir: Root directory of the dataset (ImageFolder format).
        epochs: Total training epochs.
        save_path: Where to save the best fine-tuned weights.
        arch: Backbone architecture.
    """
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Fine-tuning {arch.upper()} on {device} for {epochs} epochs")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training data not found at '{train_dir}'.\n"
            "Expected ImageFolder layout: data_dir/train/<class>/*.jpg"
        )

    train_set    = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
    val_set      = torchvision.datasets.ImageFolder(val_dir,   transform=val_transform)
    class_names  = train_set.classes
    num_classes  = len(class_names)
    print(f"Found {num_classes} classes, {len(train_set)} train / {len(val_set)} val images")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=(device == 'cuda'))
    val_loader   = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=(device == 'cuda'))

    model = build_model(num_classes=num_classes, device=device,
                        arch=arch, pretrained=True)

    # Phase 1: warm up the custom head only
    for name, param in model.named_parameters():
        param.requires_grad = ('fc' in name)
    criterion   = nn.CrossEntropyLoss(label_smoothing=0.1)
    head_epochs = max(2, epochs // 5)
    opt_head    = optim.Adam(model.fc.parameters(), lr=1e-3)
    print(f"\n--- Phase 1: head warmup ({head_epochs} epochs) ---")
    _run_epochs(model, train_loader, val_loader, criterion,
                opt_head, head_epochs, device)

    # Phase 2: fine-tune all layers
    for param in model.parameters():
        param.requires_grad = True
    remaining = epochs - head_epochs
    optimizer  = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'fc' not in n],
         'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 5e-4},
    ], weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-3, 5e-3],
        steps_per_epoch=len(train_loader),
        epochs=remaining, pct_start=0.3, anneal_strategy='cos',
    )
    print(f"\n--- Phase 2: full fine-tune ({remaining} epochs) ---")
    best_wts = _run_epochs(model, train_loader, val_loader, criterion,
                           optimizer, remaining, device,
                           scheduler=scheduler, epoch_offset=head_epochs,
                           return_best=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'state_dict':  best_wts,
        'num_classes': num_classes,
        'class_names': class_names,
        'arch':        arch,
    }, save_path)
    print(f"\n✅ Best fine-tuned model saved to {save_path}")


# ── Epoch runner ───────────────────────────────────────────────────────────────
def _run_epochs(model, train_loader, val_loader, criterion, optimizer, epochs,
                device, scheduler=None, epoch_offset=0, return_best=False):
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    step_scheduler = isinstance(scheduler, optim.lr_scheduler.OneCycleLR)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if step_scheduler:
                scheduler.step()
        if scheduler and not step_scheduler:
            scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

        val_acc = correct / total
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + epoch_offset:>3}/{epoch_offset + epochs}  "
              f"loss={running_loss / len(train_loader):.3f}  "
              f"val_acc={val_acc:.4f}  lr={lr:.2e}")

        if return_best and val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            print(f"           ✔ New best val_acc={best_acc:.4f}")

    if return_best:
        model.load_state_dict(best_wts)
        print(f"\nBest val accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        return best_wts
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ImageNet-1K classifier (pretrained ResNet + optional fine-tune)")
    parser.add_argument('--train', action='store_true',
                        help="Fine-tune on local ImageNet data")
    parser.add_argument('--data',   type=str, default='data/imagenet',
                        help="Path to ImageNet root (train/ and val/ subdirs)")
    parser.add_argument('--epochs', type=int, default=10,
                        help="Total training epochs (default: 10)")
    parser.add_argument('--save',   type=str, default='models/imagenet_finetuned.pth',
                        help="Path to save fine-tuned weights")
    parser.add_argument('--arch',   type=str, default='resnet50',
                        choices=['resnet18', 'resnet50'],
                        help="Backbone architecture (default: resnet50)")
    args = parser.parse_args()

    if args.train:
        train(data_dir=args.data, epochs=args.epochs,
              save_path=args.save, arch=args.arch)
    else:
        print(
            "Pretrained ImageNet ResNet-50 is ready for inference — no training needed!\n"
            "The model classifies images into 1,000 ImageNet categories.\n\n"
            "To fine-tune on your own data:\n"
            "  python scripts/classifier.py --train --data path/to/imagenet\n"
            "  (requires path/to/imagenet/train/<class>/ and val/<class>/ layout)\n"
        )
