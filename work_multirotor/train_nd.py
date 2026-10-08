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
    CYLINDER_RADIUS_OVER_RD,
    SA_RADIUS_OVER_RD,
    validate_checkpoint,
    COORDINATE_SYSTEM,
    INPUT_CHANNELS,
    find_case_ids,
    validation_ids_from_frame,
    load_case_input,
    make_input_surface,
)
from model_cylindrical import CylindricalUNet2D


DATA_ROOT = Path("./dataset_zrd")
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

        dataframe = pd.read_csv(csv_path, dtype={"folder": str})
        all_ids = find_case_ids(self.root)
        excluded = set(validation_ids_from_frame(self.root, dataframe))
        self.case_ids = [case_id for case_id in all_ids if case_id not in excluded]
        if not self.case_ids:
            raise RuntimeError("학습용 케이스가 없습니다. extract_nd.py를 먼저 실행하세요.")
        radii = [load_case_input(self.root / "inputs" / f"input_{case_id}.npz")["disk_radius_m"]
                 for case_id in all_ids]
        if not np.allclose(radii, radii[0], rtol=1e-6, atol=1e-7):
            raise ValueError("Extracted dataset contains different RD values; use one fixed RD")
        self.disk_radius_m = radii[0]
        ratios = [load_case_input(self.root / "inputs" / f"input_{case_id}.npz")["radius_over_rd"]
                  for case_id in all_ids]
        if not np.allclose(ratios, ratios[0], rtol=1e-6):
            raise ValueError("FATO and SA must not be mixed in a training dataset")
        self.radius_over_rd = ratios[0]
        print(f"  [{self.component.upper()}] 학습 케이스: {len(self.case_ids)}개")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        meta = load_case_input(self.root / "inputs" / f"input_{case_id}.npz")
        inputs = make_input_surface(
            meta["l_over_d"], meta["z_over_rd"], meta["shape_ztheta"]
        )
        target = np.load(
            self.root / f"targets_{self.component}" / f"{self.component}_{case_id}.npy"
        ).astype(np.float32)
        target = target[None, ...] * TARGET_SCALE
        return torch.from_numpy(inputs), torch.from_numpy(target)


def save_checkpoint(model, path, component, disk_radius_m, radius_over_rd=CYLINDER_RADIUS_OVER_RD):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "component": component,
            "in_channels": 4,
            "disk_radius_m": float(disk_radius_m),
            "coordinate_system": COORDINATE_SYSTEM,
            "input_channels": INPUT_CHANNELS,
            "out_channels": 1,
            "base_channels": BASE_CHANNELS,
            "target_scale": TARGET_SCALE,
            "surface": f"cylinder_r_equals_{radius_over_rd:g}RD",
            "radius_over_rd": float(radius_over_rd),
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
    model_path = Path(model_dir) / f"rotor_unet_cyl2d_{component}.pth"
    if model_path.is_file():
        checkpoint = torch.load(model_path, map_location="cpu")
        validate_checkpoint(checkpoint, dataset.disk_radius_m, dataset.radius_over_rd)
        if checkpoint.get("component", component) != component:
            raise ValueError("Existing model component differs from its filename")
        print(f"  [재사용] {model_path}")
        return
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    model = CylindricalUNet2D(
        in_channels=4, out_channels=1, base_channels=BASE_CHANNELS
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
    save_checkpoint(model, model_path, component, dataset.disk_radius_m, dataset.radius_over_rd)
    print(f"  저장: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U/V/W models")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sa-data-root", type=Path)
    parser.add_argument("--sa-model-dir", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    sa_root = args.sa_data_root
    if sa_root is None:
        candidate = args.data_root.with_name("dataset_zrd_sa")
        if candidate.is_dir():
            sa_root = candidate
    regions = [(args.data_root, args.model_dir, CYLINDER_RADIUS_OVER_RD)]
    if sa_root is not None:
        sa_models = args.sa_model_dir or args.model_dir.resolve().with_name("models_zrd_sa")
        if sa_root.resolve() == args.data_root.resolve() or sa_models.resolve() == args.model_dir.resolve():
            raise ValueError("FATO and SA require separate dataset/model directories")
        regions.append((sa_root, sa_models, SA_RADIUS_OVER_RD))
    elif args.sa_model_dir is not None:
        raise ValueError("--sa-model-dir requires an SA dataset")
    # Validate both regions before any training starts.
    for root, _, ratio in regions:
        dataset = CylindricalSurfaceDataset(root, "u", args.csv)
        if dataset.radius_over_rd != ratio:
            raise ValueError("Dataset region differs from its selected model directory")
    for root, models, ratio in regions:
        print(f"Region: {'FATO' if ratio == CYLINDER_RADIUS_OVER_RD else 'SA'}")
        for velocity_component in COMPONENTS:
            train_one_component(velocity_component, device, root, args.csv,
                                models, args.epochs, args.batch_size)
    print("\\n전체 완료: FATO/SA 영역별 U/V/W 모델")
