import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from typing import Sequence


class WeightedLabelSmoothingCE(nn.Module):
    def __init__(self, class_weights: Sequence[float] = None, label_smoothing: float = 0.0):
        super().__init__()
        if class_weights is not None:
            w = torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", w)
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights, 
                               label_smoothing=self.label_smoothing)


def compute_class_weights(class_counts: Sequence[int]) -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-8)
    weights = weights / weights.mean()
    return weights


def compute_metrics(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for mri, pet, y in loader:
            logits = model(mri.to(device), pet.to(device))
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.vstack(all_probs)

    acc = (all_preds == all_labels).mean()
    y_onehot = np.eye(num_classes)[all_labels]
    auc = roc_auc_score(y_onehot, all_probs, average='macro', multi_class='ovr')

    return {'ACC': acc * 100, 'AUC': auc * 100}


def compute_auc(model, loader, device, num_classes):
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for mri, pet, y in loader:
            logits = model(mri.to(device), pet.to(device))
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    y_onehot = np.eye(num_classes)[all_labels]
    return roc_auc_score(y_onehot, all_probs, average='macro', multi_class='ovr')
