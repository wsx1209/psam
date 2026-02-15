import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from models import PSAMModel
from data_loader import create_dataloaders
from utils import WeightedLabelSmoothingCE, compute_class_weights, compute_metrics, compute_auc
import config


def train_model(data_root: str, seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, val_loader, test_loader, F_mri, F_pet, labels_all = create_dataloaders(
        data_root, config.BATCH_SIZE, seed
    )

    num_classes = len(np.unique(labels_all))
    class_counts = [int((labels_all == c).sum()) for c in range(num_classes)]
    class_weights = compute_class_weights(class_counts).tolist()

    model = PSAMModel(
        in_dim_mri=1, in_dim_pet=1, num_tokens_mri=F_mri, num_tokens_pet=F_pet,
        num_classes=num_classes, d_model=config.D_MODEL, 
        num_pdf_layers=config.NUM_PDF_LAYERS, n_heads=config.N_HEADS, 
        dropout=config.DROPOUT
    ).to(device)

    criterion = WeightedLabelSmoothingCE(class_weights, config.LABEL_SMOOTHING).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                           weight_decay=config.WEIGHT_DECAY)

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=config.WARMUP_EPOCHS)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS - config.WARMUP_EPOCHS, 
                                         eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[config.WARMUP_EPOCHS])

    best_val_auc = 0.0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for mri, pet, y in train_loader:
            feature_types = torch.zeros(mri.shape[0], mri.shape[1], dtype=torch.long).to(device)
            logits = model(mri.to(device), pet.to(device), feature_types)
            loss = criterion(logits, y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=-1) == y.to(device)).sum().item()
            total += y.size(0)

        scheduler.step()
        val_auc = compute_auc(model, val_loader, device, num_classes)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            break

    model.load_state_dict(best_state_dict)
    test_metrics = compute_metrics(model, test_loader, device, num_classes)
    return test_metrics
