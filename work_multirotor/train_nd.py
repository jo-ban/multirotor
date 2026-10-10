import argparse
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from data_utils_cylindrical import (
    TARGET_SCALE,
    CYLINDER_RADIUS_OVER_RD,
    SA_RADIUS_OVER_RD,
    COORDINATE_SYSTEM,
    INPUT_CHANNELS,
    find_case_ids,
    validation_ids_from_frame,
    case_ids_for_rows,
    load_case_input,
    make_input_surface,
)
from model_cylindrical import CylindricalUNet2D
from data_utils_cylindrical import cartesian_to_cylindrical_velocity
from joint_checkpoint import MODEL_FILENAME, MODEL_FORMAT, OUTPUT_CHANNELS
from surface_loss import velocity_losses


DATA_ROOT = Path("./dataset_zrd")
CSV_PATH = Path("./cases.csv")
COMPONENTS = ("u", "v", "w")
NUM_EPOCHS = 100
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
BASE_CHANNELS = 16


class CylindricalSurfaceDataset(Dataset):
    """r=2R_D 원통 전개면에서 원통 속도 (ur, utheta, uz)와 실제 좌표를 반환한다."""

    def __init__(self, data_root, csv_path=CSV_PATH):
        self.root = Path(data_root)
        dataframe = pd.read_csv(csv_path, dtype={"folder": str})
        all_ids = find_case_ids(self.root)
        excluded = set(validation_ids_from_frame(self.root, dataframe))
        if "type" in dataframe:
            labels = dataframe["type"].astype(str).str.strip()
            excluded.update(case_ids_for_rows(self.root, dataframe[labels == "-"]))
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
        print(f"  [joint] 학습 케이스: {len(self.case_ids)}개")

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, index):
        case_id = self.case_ids[index]
        meta = load_case_input(self.root / "inputs" / f"input_{case_id}.npz")
        inputs = make_input_surface(
            meta["l_over_d"], meta["z_over_rd"], meta["shape_ztheta"]
        )
        targets = [np.load(self.root / f"targets_{c}" / f"{c}_{case_id}.npy").astype(np.float32)
                   for c in COMPONENTS]
        theta = np.linspace(0, 2 * np.pi, meta["shape_ztheta"][1], endpoint=False)
        with np.load(self.root / "inputs" / f"input_{case_id}.npz") as archive:
            if "theta_rad" in archive and not np.allclose(archive["theta_rad"], theta, atol=1e-6):
                raise ValueError("Expected the existing uniform [0,2*pi) theta grid without duplicate endpoint")
        if not np.allclose(meta["z_over_rd"], meta["z_m"] / meta["disk_radius_m"], rtol=1e-6, atol=1e-7):
            raise ValueError("z/RD metadata is inconsistent with actual z coordinates")
        cylindrical = cartesian_to_cylindrical_velocity(*targets, theta)
        target = np.stack([cylindrical[c] for c in ("u_r", "u_theta", "u_z")]).astype(np.float32)
        if target.shape != (3, *meta["shape_ztheta"]) or not np.isfinite(target).all():
            raise ValueError(f"Invalid velocity targets: {case_id}")
        return (torch.from_numpy(inputs), torch.from_numpy(target * TARGET_SCALE),
                torch.from_numpy(meta["z_over_rd"]),
                np.float32(meta["cylinder_radius_m"] / meta["disk_radius_m"]))


def save_checkpoint(model, path, dataset, epoch, losses, lambda_gradient, base_channels):
    torch.save(dict(state_dict=model.state_dict(), model_format=MODEL_FORMAT,
                    output_channels=OUTPUT_CHANNELS, in_channels=4, out_channels=3,
                    base_channels=base_channels, disk_radius_m=float(dataset.disk_radius_m),
                    radius_over_rd=float(dataset.radius_over_rd),
                    coordinate_system=COORDINATE_SYSTEM, input_channels=INPUT_CHANNELS,
                    target_scale=TARGET_SCALE, surface=f"cylinder_r_equals_{dataset.radius_over_rd:g}RD",
                    epoch=epoch, train_losses=losses, lambda_gradient=float(lambda_gradient),
                    loss_definition="L1_velocity_plus_L1_covariant_surface_gradient_zRD_rRD"), path)


def train_joint(device, data_root=DATA_ROOT, csv_path=CSV_PATH, model_dir=Path("."),
                epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, *, lambda_gradient,
                base_channels=BASE_CHANNELS):
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not math.isfinite(lambda_gradient) or lambda_gradient < 0:
        raise ValueError("lambda_gradient must be finite and nonnegative")
    dataset = CylindricalSurfaceDataset(data_root, csv_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                        pin_memory=device.type == "cuda")
    model = CylindricalUNet2D(in_channels=4, out_channels=3, base_channels=base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    fields = ("value_loss", "gradient_loss", "weighted_gradient_loss", "total_loss")
    # Every invocation trains exactly the requested epochs; old files never skip training.
    with (destination / "joint_training_losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("epoch", "lambda_gradient", *fields, "best_updated"))
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            model.train()
            sums = dict.fromkeys(fields, 0.0)
            for inputs, targets, z, radius in loader:
                inputs, targets, z, radius = [x.to(device) for x in (inputs, targets, z, radius)]
                optimizer.zero_grad(set_to_none=True)
                losses = velocity_losses(model(inputs), targets, z, radius, lambda_gradient)
                if not torch.isfinite(losses["total_loss"]):
                    raise RuntimeError("Nonfinite training loss; checkpoint was not saved")
                losses["total_loss"].backward()
                optimizer.step()
                for name in fields:
                    sums[name] += losses[name].item() * inputs.size(0)
            means = {name: value / len(dataset) for name, value in sums.items()}
            improved = means["total_loss"] < best_loss
            if improved:
                best_loss = means["total_loss"]
                save_checkpoint(model, destination / MODEL_FILENAME, dataset, epoch,
                                means, lambda_gradient, base_channels)
            row = dict(epoch=epoch, lambda_gradient=lambda_gradient, **means, best_updated=improved)
            writer.writerow(row)
            stream.flush()
            history.append(row)
            print(f"[joint r={dataset.radius_over_rd:g}RD] Epoch {epoch}/{epochs} | " +
                  " | ".join(f"{name}={means[name]:.6f}" for name in fields) + f" | best={improved}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train joint cylindrical velocity models per FATO/SA region")
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--model-dir", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--sa-data-root", type=Path)
    parser.add_argument("--sa-model-dir", type=Path)
    parser.add_argument("--lambda-gradient", type=float, required=True, help="Fixed nonnegative weight selected in Colab; no optimal value is assumed")
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
        dataset = CylindricalSurfaceDataset(root, args.csv)
        if dataset.radius_over_rd != ratio:
            raise ValueError("Dataset region differs from its selected model directory")
    for root, models, ratio in regions:
        print(f"Region: {'FATO' if ratio == CYLINDER_RADIUS_OVER_RD else 'SA'}")
        train_joint(device, root, args.csv, models, args.epochs, args.batch_size,
                    lambda_gradient=args.lambda_gradient)
    print("Joint FATO/SA best checkpoints saved")
