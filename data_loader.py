import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ADCSVDataset(Dataset):
    def __init__(self, mri_array: np.ndarray, pet_array: np.ndarray, labels: np.ndarray):
        self.mri = mri_array.astype(np.float32)
        self.pet = pet_array.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mri = torch.from_numpy(self.mri[idx]).unsqueeze(-1)
        pet = torch.from_numpy(self.pet[idx]).unsqueeze(-1)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return mri, pet, y


def load_data(data_root: str):
    stage_order = ["NC", "EMCI", "LMCI", "AD"]
    label_ids = {"NC": 0, "EMCI": 1, "LMCI": 1, "AD": 2}
    mri_list, pet_list, label_list = [], [], []

    for stage in stage_order:
        mri_path = os.path.join(data_root, f"")
        pet_path = os.path.join(data_root, f"")
        mri_arr = np.load(mri_path)
        pet_arr = np.load(pet_path)
        n_min = min(mri_arr.shape[0], pet_arr.shape[0])

        labels_stage = np.full((n_min,), label_ids[stage], dtype=np.int64)
        mri_list.append(mri_arr)
        pet_list.append(pet_arr)
        label_list.append(labels_stage)

    return np.vstack(mri_list), np.vstack(pet_list), np.concatenate(label_list)


def create_dataloaders(data_root: str, batch_size: int = 32, random_state: int = 42):
    mri_all, pet_all, labels_all = load_data(data_root)

    mri_train, mri_temp, pet_train, pet_temp, y_train, y_temp = train_test_split(
        mri_all, pet_all, labels_all, test_size=0.2, 
        random_state=random_state, stratify=labels_all
    )

    mri_val, mri_test, pet_val, pet_test, y_val, y_test = train_test_split(
        mri_temp, pet_temp, y_temp, test_size=0.5,
        random_state=random_state, stratify=y_temp
    )


    train_dataset = ADCSVDataset(mri_train, pet_train, y_train)
    val_dataset = ADCSVDataset(mri_val, pet_val, y_val)
    test_dataset = ADCSVDataset(mri_test, pet_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mri_all.shape[1], pet_all.shape[1], labels_all
