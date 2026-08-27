"""
Grad-CAM: визуализация того, на какие области изображения "смотрит" модель
----------------------------------------------------------------------------
Загружает обученную модель (из train.py) и строит тепловую карту Grad-CAM
для одного изображения — накладывает её поверх оригинала.

Использование:
    python grad_cam.py --model best_model.pt --image test.jpg --output result.jpg
"""

import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM визуализация")
    parser.add_argument("--model", type=str, required=True, help="Путь к .pt файлу от train.py")
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    parser.add_argument("--output", type=str, default="gradcam_result.jpg")
    parser.add_argument("--target-class", type=str, default=None,
                         help="Класс, для которого строить карту (по умолчанию — "
                              "предсказанный моделью класс)")
    return parser.parse_args()


class GradCAM:
    """
    Grad-CAM (Selvaraju et al., 2017): использует градиенты, текущие в последний
    сверточный слой, чтобы получить карту важности пикселей для предсказания
    конкретного класса.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()

        # Веса каналов = среднее значение градиента по пространственным осям (global average pooling)
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Взвешенная сумма активаций по каналам + ReLU (интересуют только положительные вклады)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_heatmap(original_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(args.model, map_location=device)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    # Последний сверточный блок ResNet18 — стандартный выбор слоя для Grad-CAM,
    # так как в нём ещё сохранена пространственная структура изображения
    target_layer = model.layer4[-1]
    grad_cam = GradCAM(model, target_layer)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    pil_image = Image.open(args.image).convert("RGB")
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    # Прямой проход без градиентов, чтобы определить класс, если он не задан явно
    if args.target_class is None:
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            class_idx = int(torch.argmax(probs, dim=1))
            confidence = float(probs[0, class_idx])
    else:
        class_idx = class_names.index(args.target_class)
        confidence = None

    cam = grad_cam.generate(input_tensor, class_idx)

    # Готовим оригинал в том же размере 224x224 для наложения тепловой карты
    original_resized = pil_image.resize((224, 224))
    original_bgr = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2BGR)

    result = overlay_heatmap(original_bgr, cam)

    label = class_names[class_idx]
    title = f"{label} ({confidence:.1%})" if confidence is not None else label
    cv2.putText(result, title, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(args.output, result)
    print(f"[*] Предсказанный класс: {label}"
          + (f" (уверенность {confidence:.1%})" if confidence is not None else ""))
    print(f"[*] Результат сохранён: {args.output}")


if __name__ == "__main__":
    main()
