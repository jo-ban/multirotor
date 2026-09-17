import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data_utils_cylindrical import (
    TARGET_SCALE,
    disk_radius_from_row,
    find_case_ids,
    height_over_rd_from_row,
    l_over_d_from_row,
    load_case_input,
    make_case_id,
    make_input_surface,
)
from model_cylindrical import CylindricalUNet2D


DATA_ROOT = Path("./dataset_nd")
CSV_PATH = Path("./cases.csv")
COMPONENTS = ("u", "v", "w")
NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
BASE_CHANNELS = 16


class CylindricalSurfaceDataset(Dataset):
    """r=2R_D 원통 전개면에서 특정 속도 성분 하나를 반환한다."""

    def __init__(self, data_root, component, csv_path=CSV_PATH):
        self.root = Path(data_root)
        self.component = component.lower()
        if self.component not in COMPONENTS:
            raise ValueError("component는 'u', 'v', 'w' 중 하나여야 합니다.")

        excluded = set()
        if Path(csv_path).exists():
            dataframe = pd.read_csv(csv_path)
            if "type" in dataframe.columns:
                validation_rows = dataframe[dataframe["type"].astype(str).str.upper() == "V"]
                for _, row in validation_rows.iterrows():
                    try:
                        excluded.add(
                            make_case_id(
                                l_over_d_from_row(row),
                                disk_radius_from_row(row),
                                height_over_rd_from_row(row),
                            )
                        )
                    except Exception:
                        pass

        self.case_ids = [case_id for case_id in find_case_ids(self.root) if case_id not in excluded]
        if not self.case_ids:
            raise RuntimeError("학습용 케이스가 없습니다. extract_nd.py를 먼저 실행하세요.")
        print(f"  [{self.component.upper()}] 학습 케이스: {len(self.case_ids)}개")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        meta = load_case_input(self.root / "inputs" / f"input_{case_id}.npz")
        inputs = make_input_surface(
            meta["l_over_d"], meta["height_over_rd"], meta["shape_ztheta"]
        )
        target = np.load(
            self.root / f"targets_{self.component}" / f"{self.component}_{case_id}.npy"
        ).astype(np.float32)
        target = target[None, ...] * TARGET_SCALE
        return torch.from_numpy(inputs), torch.from_numpy(target)


def save_checkpoint(model, path, component):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "component": component,
            "in_channels": 5,
            "out_channels": 1,
            "base_channels": BASE_CHANNELS,
            "target_scale": TARGET_SCALE,
            "surface": "cylinder_r_equals_2RD",
        },
        path,
    )


def train_one_component(component, device, data_root=DATA_ROOT, csv_path=CSV_PATH,
                        model_dir=Path("."), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    if not Path(csv_path).is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    dataset = CylindricalSurfaceDataset(data_root, component, csv_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = CylindricalUNet2D(
        in_channels=5, out_channels=1, base_channels=BASE_CHANNELS
    ).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n===== {component.upper()} 전용 원통표면 2D U-Net 학습 시작 =====")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        mean_loss = total_loss / len(dataset)
        print(f"[{component.upper()}] Epoch {epoch + 1:03d}/{epochs} | MAE loss={mean_loss:.6f}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir) / f"rotor_unet_cyl2d_{component}.pth"
    save_checkpoint(model, model_path, component)
    print(f"  저장: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U/V/W models")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    for velocity_component in COMPONENTS:
        train_one_component(velocity_component, device, args.data_root, args.csv,
                            args.model_dir, args.epochs, args.batch_size)
    print("\n전체 완료: rotor_unet_cyl2d_u/v/w.pth")
