"""
SuperiorGAT LiDAR Reconstruction
===================================

This script reproduces the per-frame LiDAR elevation reconstruction protocol used
in the SuperiorGAT manuscript. It simulates structured beam dropout, reconstructs
missing elevation values, and reports geometric reconstruction metrics for
interpolation baselines and neural models.

Important protocol notes
------------------------
- The task is z-coordinate reconstruction at known or structured LiDAR locations
  under simulated beam dropout.
- Metrics are computed only on the masked/dropped points.
- The script follows a per-frame reconstruction protocol: each frame is processed
  independently and reported results are averaged over frames.
- Neural runtime is reported separately as forward-pass inference time; total time
  includes graph construction and model optimization for that frame.

Datasets
--------
The script expects KITTI/nuScenes-style LiDAR `.bin` files where each file stores
float32 point records. KITTI Velodyne files normally contain x, y, z, reflectance;
only x, y, z are used here.

Example
-------
python superiorgat_reconstruction_clean.py \
    --data-path /path/to/velodyne_points/data \
    --output-dir results/person \
    --max-frames 108 \
    --num-beams 64 \
    --drop-every 4 \
    --k 10
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, knn_graph


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """Runtime and model configuration for the reconstruction experiment."""

    data_path: Path
    output_dir: Path = Path("results")
    max_frames: Optional[int] = None
    num_points: int = 50_000
    min_points: int = 100

    # Graph and beam settings. Set k=10 to match the manuscript default.
    k: int = 10
    num_beams: int = 64
    beam_min_deg: float = -24.8
    beam_max_deg: float = 2.0
    drop_every: int = 4

    # Training settings.
    num_epochs: int = 60
    patience: int = 15
    warmup_epochs: int = 10
    weight_decay: float = 5e-5
    learning_rates: Dict[str, float] = field(
        default_factory=lambda: {
            "superior_gat": 5e-4,
            "gat_baseline": 5e-4,
            "pointnet": 1e-3,
            "gcn": 1e-3,
        }
    )

    # Model settings.
    hidden_size: int = 256
    dropout: float = 0.2
    gat_dropout: float = 0.2
    gat_heads: int = 8

    # Reproducibility.
    random_seed: int = 42


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync_cuda() -> None:
    """Synchronize CUDA so timing reflects completed GPU operations."""

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_forward_pass(model: nn.Module, data: Data, is_pointnet: bool = False) -> float:
    """Measure neural forward-pass latency in seconds."""

    model.eval()
    sync_cuda()
    start = time.perf_counter()
    with torch.no_grad():
        if is_pointnet:
            _ = model(data.x.unsqueeze(0))
        else:
            _ = model(data.x, data.edge_index)
    sync_cuda()
    return time.perf_counter() - start


def read_lidar_bin(path: Path) -> np.ndarray:
    """Read a KITTI/nuScenes-style binary LiDAR file and return x, y, z."""

    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
    if points.shape[0] == 0:
        raise ValueError(f"Empty LiDAR file: {path}")
    if np.any(np.isnan(points)):
        raise ValueError(f"NaN values detected in LiDAR file: {path}")
    return points


def compute_beam_indices(points: np.ndarray, cfg: ExperimentConfig) -> np.ndarray:
    """Estimate beam indices from vertical angles."""

    horizontal_range = np.linalg.norm(points[:, :2], axis=1)
    vertical_angles = np.degrees(np.arctan2(points[:, 2], horizontal_range))
    beam_edges = np.linspace(cfg.beam_min_deg, cfg.beam_max_deg, cfg.num_beams + 1)
    beam_indices = np.digitize(vertical_angles, beam_edges) - 1
    return np.clip(beam_indices, 0, cfg.num_beams - 1)


def stratified_sample(
    points: np.ndarray,
    beam_indices: np.ndarray,
    num_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points while preserving approximate representation from each beam."""

    unique_beams = np.unique(beam_indices)
    target_per_beam = max(1, num_points // max(1, len(unique_beams)))

    selected_indices: List[int] = []
    for beam in unique_beams:
        candidates = np.where(beam_indices == beam)[0]
        if len(candidates) > target_per_beam:
            chosen = np.random.choice(candidates, target_per_beam, replace=False)
        else:
            chosen = candidates
        selected_indices.extend(chosen.tolist())

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    if len(selected_indices) > num_points:
        selected_indices = np.random.choice(selected_indices, num_points, replace=False)
    elif len(selected_indices) < num_points:
        extra = np.random.choice(
            np.arange(len(points)),
            num_points - len(selected_indices),
            replace=True,
        )
        selected_indices = np.concatenate([selected_indices, extra])

    return points[selected_indices], beam_indices[selected_indices]


def structured_dropout_mask(beam_indices: np.ndarray, drop_every: int) -> np.ndarray:
    """Return a mask for retained points after structured beam dropout.

    Example: drop_every=4 removes beam indices divisible by 4, giving an
    approximate 25% structured beam dropout on a 64-beam scan.
    """

    if drop_every <= 1:
        raise ValueError("drop_every must be greater than 1")
    return (beam_indices % drop_every) != 0


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def compute_rmse_per_coord(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float, float]:
    """Return RMSE for x, y, and z coordinates."""

    errors = (pred - gt) ** 2
    rmse_x = torch.sqrt(torch.mean(errors[:, 0])).item()
    rmse_y = torch.sqrt(torch.mean(errors[:, 1])).item()
    rmse_z = torch.sqrt(torch.mean(errors[:, 2])).item()
    return rmse_x, rmse_y, rmse_z


def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute symmetric Chamfer distance using Euclidean distances."""

    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    dist_pred_to_gt = cdist(pred_np, gt_np)
    dist_gt_to_pred = cdist(gt_np, pred_np)
    return float(np.min(dist_pred_to_gt, axis=1).mean() + np.min(dist_gt_to_pred, axis=1).mean())


def surface_normal_consistency(pred: torch.Tensor, gt: torch.Tensor, k: int = 5) -> float:
    """Estimate local normals and return mean absolute normal agreement."""

    try:
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        def compute_normals(points: np.ndarray) -> np.ndarray:
            nn_model = NearestNeighbors(n_neighbors=k + 1).fit(points)
            indices = nn_model.kneighbors(points, return_distance=False)[:, 1:]
            normals = []
            for i, neighbor_idx in enumerate(indices):
                neighbors = points[neighbor_idx]
                if len(neighbors) >= 3:
                    centered = neighbors - points[i]
                    _, _, vh = np.linalg.svd(centered, full_matrices=False)
                    normal = vh[-1]
                    normal = normal / (np.linalg.norm(normal) + 1e-8)
                    normals.append(normal)
                else:
                    normals.append(np.array([0.0, 0.0, 1.0]))
            return np.asarray(normals)

        pred_normals = compute_normals(pred_np)
        gt_normals = compute_normals(gt_np)
        return float(np.abs(np.sum(pred_normals * gt_normals, axis=1)).mean())
    except Exception:
        return float("nan")


def summarize_prediction(pred_pos: torch.Tensor, gt_pos: torch.Tensor) -> Dict[str, float]:
    """Compute all metrics for predicted and ground-truth point sets."""

    rmse_x, rmse_y, rmse_z = compute_rmse_per_coord(pred_pos, gt_pos)
    rmse_xyz = math.sqrt((rmse_x**2 + rmse_y**2 + rmse_z**2) / 3.0)
    return {
        "rmse_xyz": rmse_xyz,
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_z": rmse_z,
        "chamfer": chamfer_distance(pred_pos, gt_pos),
        "normal_consistency": surface_normal_consistency(pred_pos, gt_pos),
    }


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class EnhancedPointNet(nn.Module):
    """PointNet-style baseline for point-wise z regression."""

    def __init__(self, cfg: ExperimentConfig, in_channels: int = 4):
        super().__init__()
        hidden = cfg.hidden_size
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, hidden, 1)
        self.conv3 = nn.Conv1d(hidden, hidden, 1)
        self.conv4 = nn.Conv1d(hidden, 1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(F.relu(self.bn2(self.conv3(x))))
        return self.conv4(x).transpose(2, 1)


class SuperiorGAT(nn.Module):
    """SuperiorGAT with single graph-attention layer, gated residual fusion, and FFN refinement."""

    def __init__(self, cfg: ExperimentConfig, in_channels: int = 4):
        super().__init__()
        hidden = cfg.hidden_size
        heads = cfg.gat_heads

        self.input_proj = nn.Linear(in_channels, hidden)
        self.gat = GATConv(
            hidden,
            hidden // heads,
            heads=heads,
            dropout=cfg.gat_dropout,
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Scalar gate used in the manuscript to blend attention output with identity features.
        self.gate = nn.Parameter(torch.tensor(0.5))

        self.ffn = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.LeakyReLU(0.2),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * hidden, hidden),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.input_proj(x))

        residual = x
        attention = F.leaky_relu(self.gat(x, edge_index), 0.2)
        x = self.norm1(self.gate * attention + (1.0 - self.gate) * residual)

        residual = x
        x = self.norm2(self.ffn(x) + residual)
        return self.output(x)


class BaselineGAT(nn.Module):
    """Vanilla multi-layer GAT baseline without gated residual or FFN refinement."""

    def __init__(self, cfg: ExperimentConfig, in_channels: int = 4):
        super().__init__()
        hidden = cfg.hidden_size
        heads = cfg.gat_heads
        self.input_proj = nn.Linear(in_channels, hidden)
        self.gat1 = GATConv(hidden, hidden // heads, heads=heads, dropout=cfg.gat_dropout)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=cfg.gat_dropout)
        self.gat_out = GATConv(hidden, hidden, heads=1, dropout=cfg.gat_dropout)
        self.output = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = F.leaky_relu(self.gat1(x, edge_index), 0.2)
        x = F.leaky_relu(self.gat2(x, edge_index), 0.2)
        x = F.leaky_relu(self.gat_out(x, edge_index), 0.2)
        return self.output(x)


class SimpleGCN(nn.Module):
    """Three-layer GCN baseline for point-wise z regression."""

    def __init__(self, cfg: ExperimentConfig, in_channels: int = 4):
        super().__init__()
        hidden = cfg.hidden_size
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, 1)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.dropout(F.relu(self.conv2(x, edge_index)))
        return self.conv3(x, edge_index)


# -----------------------------------------------------------------------------
# Reconstruction methods
# -----------------------------------------------------------------------------


def train_neural_network(
    model: nn.Module,
    data: Data,
    target: torch.Tensor,
    masked_points: torch.Tensor,
    cfg: ExperimentConfig,
    learning_rate: float,
    is_pointnet: bool = False,
) -> Tuple[nn.Module, List[float]]:
    """Optimize a neural model for one frame using masked z-coordinates as targets."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    losses: List[float] = []
    best_loss = float("inf")
    patience_counter = 0

    def scheduled_lr(epoch: int) -> float:
        if epoch < cfg.warmup_epochs:
            return learning_rate * (epoch + 1) / max(1, cfg.warmup_epochs)
        progress = (epoch - cfg.warmup_epochs) / max(1, cfg.num_epochs - cfg.warmup_epochs)
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))

    model.train()
    for epoch in range(cfg.num_epochs):
        for group in optimizer.param_groups:
            group["lr"] = scheduled_lr(epoch)

        optimizer.zero_grad()
        if is_pointnet:
            pred = model(data.x.unsqueeze(0)).squeeze(0)
        else:
            pred = model(data.x, data.edge_index)

        loss = loss_fn(pred[masked_points], target[masked_points].unsqueeze(1))
        if torch.isnan(loss):
            print(f"Warning: NaN loss at epoch {epoch}")
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))

        if loss.item() < best_loss:
            best_loss = float(loss.item())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                break

    return model, losses


def run_interpolation_baselines(
    points: np.ndarray,
    keep_mask: np.ndarray,
    pts_keep: np.ndarray,
    pts_skip: np.ndarray,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Run linear interpolation and nearest-neighbor interpolation baselines."""

    results: Dict[str, Dict[str, float]] = {}
    gt_skip = torch.tensor(pts_skip, dtype=torch.float32, device=device)

    # Linear interpolation estimates z at the dropped points' known x-y locations.
    try:
        start = time.perf_counter()
        interp_z = griddata(
            pts_keep[:, :2],
            pts_keep[:, 2],
            pts_skip[:, :2],
            method="linear",
            fill_value=np.nan,
        )
        valid = ~np.isnan(interp_z)
        if valid.sum() == 0:
            raise ValueError("Linear interpolation produced no valid predictions")

        pred = pts_skip[valid].copy()
        pred[:, 2] = interp_z[valid]
        pred_tensor = torch.tensor(pred, dtype=torch.float32, device=device)
        gt_tensor = torch.tensor(pts_skip[valid], dtype=torch.float32, device=device)

        metrics = summarize_prediction(pred_tensor, gt_tensor)
        metrics["time_total_sec"] = time.perf_counter() - start
        metrics["time_inference_sec"] = float("nan")
        results["linear_interp"] = metrics
    except Exception as exc:
        print(f"Linear interpolation failed: {exc}")
        results["linear_interp"] = nan_metrics()

    # Nearest-neighbor baseline selects the nearest observed point using x-y proximity.
    # The full nearest observed 3D point is used as the baseline prediction, matching a
    # simple local interpolation strategy without using missing z-values for neighbor search.
    try:
        start = time.perf_counter()
        nn_model = NearestNeighbors(n_neighbors=1).fit(pts_keep[:, :2])
        _, indices = nn_model.kneighbors(pts_skip[:, :2])
        pred = pts_keep[indices[:, 0]]
        pred_tensor = torch.tensor(pred, dtype=torch.float32, device=device)

        metrics = summarize_prediction(pred_tensor, gt_skip)
        metrics["time_total_sec"] = time.perf_counter() - start
        metrics["time_inference_sec"] = float("nan")
        results["nearest_neighbor"] = metrics
    except Exception as exc:
        print(f"Nearest-neighbor interpolation failed: {exc}")
        results["nearest_neighbor"] = nan_metrics()

    return results


def nan_metrics() -> Dict[str, float]:
    """Return a metric dictionary populated with NaN values."""

    return {
        "rmse_xyz": float("nan"),
        "rmse_x": float("nan"),
        "rmse_y": float("nan"),
        "rmse_z": float("nan"),
        "chamfer": float("nan"),
        "normal_consistency": float("nan"),
        "time_total_sec": float("nan"),
        "time_inference_sec": float("nan"),
        "convergence_epoch": float("nan"),
    }


def prepare_frame_data(
    points: np.ndarray,
    keep_mask: np.ndarray,
    beam_indices: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Tuple[Data, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build normalized features, targets, mask, and graph for one frame."""

    pos = torch.tensor(points[:, :3], dtype=torch.float32, device=device)
    beam_tensor = torch.tensor(beam_indices, dtype=torch.float32, device=device).unsqueeze(1)
    beam_tensor = beam_tensor / max(1, cfg.num_beams - 1)

    input_features = pos.clone()
    z_masked = torch.zeros(points.shape[0], dtype=torch.float32, device=device)
    z_masked[keep_mask] = pos[keep_mask, 2]
    input_features[:, 2] = z_masked
    input_features = torch.cat([input_features, beam_tensor], dim=1)

    feature_mean = input_features.mean(dim=0)
    feature_std = input_features.std(dim=0) + 1e-6
    input_norm = (input_features - feature_mean) / feature_std

    z_gt = pos[:, 2]
    z_mean = z_gt.mean()
    z_std = z_gt.std() + 1e-6
    target_norm = (z_gt - z_mean) / z_std

    masked_points = torch.tensor(~keep_mask, dtype=torch.bool, device=device)
    edge_index = knn_graph(pos, k=cfg.k, loop=False).to(device)
    data = Data(x=input_norm, edge_index=edge_index, y=target_norm)

    return data, pos, target_norm, masked_points, z_mean, z_std


def evaluate_neural_model(
    model: nn.Module,
    data: Data,
    gt_skip: torch.Tensor,
    masked_points: torch.Tensor,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    is_pointnet: bool = False,
) -> Dict[str, float]:
    """Run model inference and compute metrics on the masked points."""

    model.eval()
    with torch.no_grad():
        if is_pointnet:
            pred_norm = model(data.x.unsqueeze(0)).squeeze(0)
        else:
            pred_norm = model(data.x, data.edge_index)
        pred_z = pred_norm[:, 0] * z_std + z_mean

    pred_skip = gt_skip.clone()
    pred_skip[:, 2] = pred_z[masked_points]
    return summarize_prediction(pred_skip, gt_skip)


def run_neural_models(
    points: np.ndarray,
    keep_mask: np.ndarray,
    pts_skip: np.ndarray,
    beam_indices: np.ndarray,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """Train and evaluate all neural reconstruction models for one frame."""

    results: Dict[str, Dict[str, float]] = {}
    gt_skip = torch.tensor(pts_skip, dtype=torch.float32, device=device)
    data, _, target_norm, masked_points, z_mean, z_std = prepare_frame_data(
        points, keep_mask, beam_indices, cfg, device
    )

    model_specs = {
        "superior_gat": (SuperiorGAT(cfg).to(device), cfg.learning_rates["superior_gat"], False),
        "gat_baseline": (BaselineGAT(cfg).to(device), cfg.learning_rates["gat_baseline"], False),
        "enhanced_pointnet": (EnhancedPointNet(cfg).to(device), cfg.learning_rates["pointnet"], True),
        "simple_gcn": (SimpleGCN(cfg).to(device), cfg.learning_rates["gcn"], False),
    }

    for method_name, (model, lr, is_pointnet) in model_specs.items():
        try:
            start_total = time.perf_counter()
            model, losses = train_neural_network(
                model=model,
                data=data,
                target=target_norm,
                masked_points=masked_points,
                cfg=cfg,
                learning_rate=lr,
                is_pointnet=is_pointnet,
            )
            inference_sec = time_forward_pass(model, data, is_pointnet=is_pointnet)
            metrics = evaluate_neural_model(
                model=model,
                data=data,
                gt_skip=gt_skip,
                masked_points=masked_points,
                z_mean=z_mean,
                z_std=z_std,
                is_pointnet=is_pointnet,
            )
            metrics["time_total_sec"] = time.perf_counter() - start_total
            metrics["time_inference_sec"] = inference_sec
            metrics["convergence_epoch"] = float(len(losses))
            results[method_name] = metrics
            print(
                f"  {method_name}: RMSE_Z={metrics['rmse_z']:.4f}, "
                f"Chamfer={metrics['chamfer']:.4f}, "
                f"Infer={1000 * inference_sec:.2f} ms"
            )
        except Exception as exc:
            print(f"  {method_name} failed: {exc}")
            results[method_name] = nan_metrics()
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


# -----------------------------------------------------------------------------
# Experiment loop
# -----------------------------------------------------------------------------


def flatten_results(frame_name: str, method_results: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    """Convert nested method results into one row per frame and method."""

    rows = []
    for method, metrics in method_results.items():
        row = {"frame": frame_name, "method": method}
        row.update(metrics)
        row["time_inference_ms"] = (
            metrics["time_inference_sec"] * 1000.0
            if not np.isnan(metrics.get("time_inference_sec", float("nan")))
            else float("nan")
        )
        row["time_total_ms"] = (
            metrics["time_total_sec"] * 1000.0
            if not np.isnan(metrics.get("time_total_sec", float("nan")))
            else float("nan")
        )
        rows.append(row)
    return rows


def run_experiment(cfg: ExperimentConfig) -> pd.DataFrame:
    """Run reconstruction experiments over all selected LiDAR frames."""

    set_random_seed(cfg.random_seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    lidar_files = sorted(Path(cfg.data_path).glob("*.bin"))
    if cfg.max_frames is not None:
        lidar_files = lidar_files[: cfg.max_frames]
    if not lidar_files:
        raise FileNotFoundError(f"No .bin LiDAR files found in {cfg.data_path}")

    all_rows: List[Dict[str, float]] = []
    for frame_idx, lidar_file in enumerate(lidar_files, start=1):
        print(f"\nFrame {frame_idx}/{len(lidar_files)}: {lidar_file.name}")
        try:
            points = read_lidar_bin(lidar_file)
            if points.shape[0] < cfg.min_points:
                raise ValueError(f"Point cloud too small: {points.shape[0]} points")

            beam_indices = compute_beam_indices(points, cfg)
            if cfg.num_points is not None and len(points) > cfg.num_points:
                points, beam_indices = stratified_sample(points, beam_indices, cfg.num_points)

            keep_mask = structured_dropout_mask(beam_indices, cfg.drop_every)
            skip_mask = ~keep_mask
            if skip_mask.sum() < cfg.min_points or keep_mask.sum() < cfg.min_points:
                raise ValueError(
                    f"Insufficient points after dropout: kept={keep_mask.sum()}, dropped={skip_mask.sum()}"
                )

            pts_keep = points[keep_mask]
            pts_skip = points[skip_mask]
            print(
                f"  points={len(points)}, retained={keep_mask.sum()} "
                f"({keep_mask.sum() / len(points):.1%}), dropped beams={len(np.unique(beam_indices[skip_mask]))}"
            )

            baseline_results = run_interpolation_baselines(points, keep_mask, pts_keep, pts_skip, device)
            neural_results = run_neural_models(points, keep_mask, pts_skip, beam_indices, cfg, device)

            combined = {**baseline_results, **neural_results}
            all_rows.extend(flatten_results(lidar_file.name, combined))
            gc.collect()
        except Exception as exc:
            print(f"  Frame failed: {exc}")
            all_rows.append({"frame": lidar_file.name, "method": "error", "error": str(exc)})

    results_df = pd.DataFrame(all_rows)
    output_csv = cfg.output_dir / "reconstruction_results_by_frame.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved frame-level results to {output_csv}")

    summary = summarize_dataframe(results_df)
    summary_csv = cfg.output_dir / "reconstruction_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Saved summary results to {summary_csv}")
    print(summary.to_string(index=False))

    return results_df


def summarize_dataframe(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and standard deviation for each metric by method."""

    metric_cols = [
        "rmse_xyz",
        "rmse_x",
        "rmse_y",
        "rmse_z",
        "chamfer",
        "normal_consistency",
        "time_inference_ms",
        "time_total_ms",
        "convergence_epoch",
    ]
    valid = results_df[results_df["method"] != "error"].copy()
    available_metrics = [col for col in metric_cols if col in valid.columns]
    summary = valid.groupby("method")[available_metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.to_flat_index()]
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="SuperiorGAT LiDAR beam dropout reconstruction")
    parser.add_argument("--data-path", required=True, type=Path, help="Directory containing LiDAR .bin files")
    parser.add_argument("--output-dir", default=Path("results"), type=Path, help="Directory for CSV outputs")
    parser.add_argument("--max-frames", default=None, type=int, help="Maximum number of frames to process")
    parser.add_argument("--num-points", default=50_000, type=int, help="Maximum points per frame after sampling")
    parser.add_argument("--k", default=10, type=int, help="k-nearest-neighbor graph size")
    parser.add_argument("--num-beams", default=64, type=int, help="Number of LiDAR beams used for beam indexing")
    parser.add_argument("--beam-min-deg", default=-24.8, type=float, help="Minimum vertical beam angle")
    parser.add_argument("--beam-max-deg", default=2.0, type=float, help="Maximum vertical beam angle")
    parser.add_argument("--drop-every", default=4, type=int, help="Drop beams divisible by this value")
    parser.add_argument("--num-epochs", default=60, type=int, help="Maximum training epochs per frame")
    parser.add_argument("--patience", default=15, type=int, help="Early stopping patience based on reconstruction loss")
    parser.add_argument("--hidden-size", default=256, type=int, help="Hidden feature size")
    parser.add_argument("--random-seed", default=42, type=int, help="Random seed")

    args = parser.parse_args()
    return ExperimentConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        num_points=args.num_points,
        k=args.k,
        num_beams=args.num_beams,
        beam_min_deg=args.beam_min_deg,
        beam_max_deg=args.beam_max_deg,
        drop_every=args.drop_every,
        num_epochs=args.num_epochs,
        patience=args.patience,
        hidden_size=args.hidden_size,
        random_seed=args.random_seed,
    )


if __name__ == "__main__":
    config = parse_args()
    run_experiment(config)
