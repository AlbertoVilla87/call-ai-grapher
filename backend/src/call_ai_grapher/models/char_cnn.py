"""Small convolutional network that labels character crops (A-Z + Ñ).

Shared by the training CLI (`train_classifier.py`) and the pipeline stage
(`CharClassifier`) so architecture and checkpoint format stay in sync.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from call_ai_grapher.dataset.builder import normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class CharCNN(nn.Module):
    def __init__(self, num_classes: int, size: int = 64):
        """_summary_
        :param num_classes: number of character classes
        :type num_classes: int
        :param size: input image side in pixels
        :type size: int
        """
        super().__init__()
        self.size = size
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        pooled = size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * pooled * pooled, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of character images.

        :param x: batch of grayscale images (B, 1, size, size), values in [0, 1]
        :type x: torch.Tensor
        :return: logits of shape (B, num_classes)
        :rtype: torch.Tensor
        """
        return self.classifier(self.features(x))


def save_checkpoint(path: Union[str, Path], model: CharCNN, classes: List[str]) -> None:
    """Persist model weights together with its class mapping.

    :param path: destination checkpoint path
    :type path: Union[str, Path]
    :param model: trained model
    :type model: CharCNN
    :param classes: class names indexed by output unit
    :type classes: List[str]
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": classes, "size": model.size}, path)
    logging.info("Checkpoint saved to %s", path)


def load_checkpoint(path: Union[str, Path]) -> Tuple[CharCNN, List[str]]:
    """Rebuild a model from a checkpoint saved by `save_checkpoint`.

    :param path: checkpoint path
    :type path: Union[str, Path]
    :return: model in evaluation mode and its class names
    :rtype: Tuple[CharCNN, List[str]]
    """
    checkpoint = torch.load(str(path), map_location="cpu")
    classes = checkpoint["classes"]
    model = CharCNN(num_classes=len(classes), size=checkpoint.get("size", 64))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, classes


def load_dataset(data_dir: Union[str, Path], size: int) -> Tuple[TensorDataset, List[str]]:
    """Load the normalized alphabet dataset as tensors.

    :param data_dir: dataset root with one directory per class
    :type data_dir: Union[str, Path]
    :param size: expected normalized image side; images are re-normalized if needed
    :type size: int
    :return: tensor dataset of (images, targets) and the class names
    :rtype: Tuple[TensorDataset, List[str]]
    """
    root = Path(data_dir)
    classes = sorted(d.name for d in root.iterdir() if d.is_dir())
    class_index = {name: i for i, name in enumerate(classes)}
    images, targets = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                logging.warning("Skipping unreadable image %s", path)
                continue
            if image.shape != (size, size):
                image = normalize(image, size)
            images.append(image)
            targets.append(class_index[class_dir.name])
    tensors = torch.tensor(np.array(images), dtype=torch.float32).unsqueeze(1) / 255.0
    labels = torch.tensor(targets, dtype=torch.long)
    return TensorDataset(tensors, labels), classes


def augment_batch(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply random affine augmentation to a batch of character images.

    Rotations up to ±10 degrees, scale between 0.85 and 1.15 and small
    translations simulate handwriting variability.

    :param images: batch (B, 1, H, W)
    :type images: torch.Tensor
    :param generator: torch random generator for reproducibility
    :type generator: torch.Generator
    :return: augmented batch
    :rtype: torch.Tensor
    """
    batch = images.clone()
    for i in range(batch.shape[0]):
        angle = float(torch.empty(1).uniform_(-10, 10, generator=generator))
        scale = float(torch.empty(1).uniform_(0.85, 1.15, generator=generator))
        translate = (
            float(torch.empty(1).uniform_(-0.08, 0.08, generator=generator)),
            float(torch.empty(1).uniform_(-0.08, 0.08, generator=generator)),
        )
        matrix = cv2.getRotationMatrix2D((batch.shape[3] / 2, batch.shape[2] / 2), angle, scale)
        matrix[:, 2] += (translate[0] * batch.shape[3], translate[1] * batch.shape[2])
        warped = cv2.warpAffine(batch[i, 0].numpy(), matrix, (batch.shape[3], batch.shape[2]), borderValue=1.0)
        batch[i, 0] = torch.from_numpy(warped)
    return batch


def train_model(
    data_dir: Union[str, Path],
    out_path: Union[str, Path],
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
) -> Dict[str, float]:
    """Train CharCNN on an alphabet dataset and save the best checkpoint.

    :param data_dir: alphabet dataset directory (one directory per class)
    :type data_dir: Union[str, Path]
    :param out_path: where to write the checkpoint
    :type out_path: Union[str, Path]
    :param epochs: number of training epochs
    :type epochs: int
    :param batch_size: images per optimization step
    :type batch_size: int
    :param lr: learning rate
    :type lr: float
    :param seed: random seed
    :type seed: int
    :return: final metrics {loss, accuracy}
    :rtype: Dict[str, float]
    """
    torch.manual_seed(seed)
    dataset, classes = load_dataset(data_dir, size=64)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CharCNN(num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)

    model.train()
    for epoch in range(epochs):
        total_loss, correct, seen = 0.0, 0, 0
        for images, targets in loader:
            optimizer.zero_grad()
            outputs = model(augment_batch(images, generator))
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            correct += (outputs.argmax(1) == targets).sum().item()
            seen += len(targets)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logging.info(
                "epoch %d/%d - loss %.4f - accuracy %.3f", epoch + 1, epochs, total_loss / seen, correct / seen
            )

    metrics = {"loss": total_loss / seen, "accuracy": correct / seen}
    model.eval()
    save_checkpoint(out_path, model, classes)
    return metrics
