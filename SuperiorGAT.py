#                     THIS IS LAST VERSION 08/12/2025 WHERE GAT OUTPERFORM OTHER METHODS.
#                    BASICALLY, IT SHOULD BE READY TO USE FOR MY MANUSCRIPT.
#                   In 12/21/2025 I implement Superior GAT with one Layer and adding FFN, gate residual.

# LiDAR Point Cloud Reconstruction for Autonomous Applications
# This script processes KITTI LiDAR dataset frames to simulate sparsity (beam dropout)
# and reconstruct missing points' z-coordinates using baseline methods (linear interpolation,
# nearest neighbors) and neural network models (SuperiorGAT, EnhancedPointNet, SimpleGCN).
# Purpose: Evaluate reconstruction accuracy for cost-effective LiDAR in autonomous vehicles.
# Key metrics: RMSE_XYZ, Chamfer distance, surface normal consistency.
# Dataset: KITTI velodyne points (~100k points/frame, subsampled to 50k).
# Sparsity: Drop every 4th beam (~25% points dropped) to mimic low-res sensors.


import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, GATConv, GCNConv
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import gc
import time
from typing import Dict, List, Tuple, Optional

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Complete configuration with all required keys
CONFIG = {
    #'data_path': "/kaggle/input/lidar-dataset/2011_09_26_drive_0001_sync/2011_09_26_drive_0001_sync/velodyne_points/data",
    #
    #'data_path': "/kaggle/input/lidar-dataset-2011-09-29/LiDAR_Dataset_2011_09_29/2011_09_29_drive_0071_sync/velodyne_points/data",
    #'data_path': "/kaggle/input/campus-dataset/2011_09_28/2011_09_28_drive_0038_sync/velodyne_points/data",
    #'data_path': "/kaggle/input/person-dataset/2011_09_28/2011_09_28_drive_0209_sync/velodyne_points/data",
    'data_path': "/kaggle/input/person-dataset/2011_09_28/2011_09_28_drive_0209_sync/velodyne_points/data",
    'k': 6,  # Increased from 5 to capture more neighborhood information
    'num_points': 50000,
    'num_epochs': 60,  # Increased slightly to allow more training if needed
    'min_points': 100,
    'learning_rate': 0.001,  # Single learning rate for all models
    'learning_rates': {      # Individual learning rates (if needed)
        'superior_gat': 0.0005,  # Reduced LR for GAT to stabilize training
        'pointnet': 0.001,
        'gcn': 0.001
    },
    'hidden_size': 256,  # Increased hidden size for more capacity
    'patience': 15,  # Increased patience for early stopping
    'weight_decay': 5e-5,  # Adjusted weight decay
    'dropout': 0.2,  # Reduced dropout
    'gat_dropout': 0.2,     # Reduced GAT dropout
    'warmup_epochs': 10,  # Increased warmup for better initial learning
    'random_seed': 42,
    'gat_heads': 8
}

# Set random seeds
np.random.seed(CONFIG['random_seed'])
torch.manual_seed(CONFIG['random_seed'])

class EnhancedPointNet(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, CONFIG['hidden_size'], 1)
        self.conv3 = nn.Conv1d(CONFIG['hidden_size'], CONFIG['hidden_size'], 1)
        self.conv4 = nn.Conv1d(CONFIG['hidden_size'], 1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(CONFIG['hidden_size'])
        self.dropout = nn.Dropout(CONFIG['dropout'])
        
    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(F.relu(self.bn2(self.conv3(x))))
        return self.conv4(x).transpose(2, 1)
############################################################################################################################

'''
THIS IS ONE LAYER SUPERIOR GAT WITHOUT ANY CHANGE FOR MULTIPLE LAYERS 11/12/2025

class SuperiorGAT(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, CONFIG['hidden_size'])

        # Single GAT layer
        self.gat = GATConv(
            CONFIG['hidden_size'],
            CONFIG['hidden_size'] // CONFIG['gat_heads'],
            heads=CONFIG['gat_heads'],
            dropout=CONFIG['gat_dropout']
        )

        # One normalization layer
        self.norm = nn.LayerNorm(CONFIG['hidden_size'])

        # Output head (same as before)
        self.output = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], CONFIG['hidden_size'] // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(CONFIG['hidden_size'] // 2, 1)
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.norm(x)

        # GAT layer with residual
        x_res = x
        x = F.leaky_relu(self.gat(x, edge_index), 0.2)
        x = self.norm(x) + x_res  # normalization + residual

        return self.output(x)



'''

########################################################################################################################

# I MADE SOME CHANGES 
#(1) A Feed-Forward Network (FFN) sub-block after attention.
#(2) A Residual Gate to softly blend attention and identity.

class SuperiorGAT(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        H = CONFIG['hidden_size']
        self.input_proj = nn.Linear(in_channels, H)

        # ---- GAT layer ----
        self.gat = GATConv(
            H,
            H // CONFIG['gat_heads'],
            heads=CONFIG['gat_heads'],
            dropout=CONFIG['gat_dropout']
        )

        # ---- Layer norms ----
        self.norm1 = nn.LayerNorm(H)
        self.norm2 = nn.LayerNorm(H)

        # ---- Residual gate ----
        self.gate = nn.Parameter(torch.tensor(0.5))  # learnable scalar in [0,1]

        # ---- Feed-forward sub-block (Transformer-style) ----
        self.ffn = nn.Sequential(
            nn.Linear(H, 2 * H),
            nn.LeakyReLU(0.2),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(2 * H, H)
        )

        # ---- Output head ----
        self.output = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(H // 2, 1)
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.norm1(x)

        # ---- GAT + gated residual ----
        x_res = x
        attn_out = F.leaky_relu(self.gat(x, edge_index), 0.2)
        x = self.norm1(self.gate * attn_out + (1 - self.gate) * x_res)

        # ---- Feed-forward + residual ----
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x + x_res)

        return self.output(x)




###################################################################################################################

'''

class SuperiorGAT(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, CONFIG['hidden_size'])
        
        self.gat1 = GATConv(
            CONFIG['hidden_size'], 
            CONFIG['hidden_size'] // CONFIG['gat_heads'], 
            heads=CONFIG['gat_heads'],
            dropout=CONFIG['gat_dropout']
        )
        self.gat2 = GATConv(
            CONFIG['hidden_size'], 
            CONFIG['hidden_size'] // CONFIG['gat_heads'], 
            heads=CONFIG['gat_heads'],
            dropout=CONFIG['gat_dropout']
        )
        self.gat3 = GATConv(
            CONFIG['hidden_size'], 
            CONFIG['hidden_size'],
            heads=1,
            dropout=CONFIG['gat_dropout']
        )
        
        self.norm1 = nn.LayerNorm(CONFIG['hidden_size'])
        self.norm2 = nn.LayerNorm(CONFIG['hidden_size'])
        self.norm3 = nn.LayerNorm(CONFIG['hidden_size'])
        
        self.output = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], CONFIG['hidden_size'] // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(CONFIG['hidden_size'] // 2, 1)
        )
        
    def forward(self, x, edge_index):
        x = self.input_proj(x)
        x = self.norm1(x)
        
        x_res = x
        x = F.leaky_relu(self.gat1(x, edge_index), 0.2)
        x = self.norm2(x) + x_res
        
        x_res = x
        x = F.leaky_relu(self.gat2(x, edge_index), 0.2)
        x = self.norm3(x) + x_res
        
        x = F.leaky_relu(self.gat3(x, edge_index), 0.2)
        return self.output(x)
  '''      
#######################################################################



class BaselineGAT(nn.Module):
    """
    Vanilla GAT (no LayerNorm, no residual). Same I/O as SuperiorGAT.
    Two multi-head layers + one single-head projection, followed by the same MLP decoder.
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, CONFIG['hidden_size'])

        # Two GATConv layers with multi-head attention (concat)
        self.gat1 = GATConv(
            CONFIG['hidden_size'],
            CONFIG['hidden_size'] // CONFIG['gat_heads'],
            heads=CONFIG['gat_heads'],
            dropout=CONFIG['gat_dropout']
        )
        self.gat2 = GATConv(
            CONFIG['hidden_size'],
            CONFIG['hidden_size'] // CONFIG['gat_heads'],
            heads=CONFIG['gat_heads'],
            dropout=CONFIG['gat_dropout']
        )

        # Final single-head projection to hidden size
        self.gat_out = GATConv(
            CONFIG['hidden_size'],
            CONFIG['hidden_size'],
            heads=1,
            dropout=CONFIG['gat_dropout']
        )

        # Same decoder head used in SuperiorGAT for fair comparison
        self.output = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], CONFIG['hidden_size'] // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(CONFIG['dropout']),
            nn.Linear(CONFIG['hidden_size'] // 2, 1)
        )

    def forward(self, x, edge_index):
        x = self.input_proj(x)                    # N x H
        x = F.leaky_relu(self.gat1(x, edge_index), 0.2)
        x = F.leaky_relu(self.gat2(x, edge_index), 0.2)

        #x = F.leaky_re_lu(self.gat2(x, edge_index), 0.2)
        x = F.leaky_relu(self.gat_out(x, edge_index), 0.2)
        return self.output(x)                     # N x 1





##################################################################
class SimpleGCN(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv1 = GCNConv(in_channels, CONFIG['hidden_size'])
        self.conv2 = GCNConv(CONFIG['hidden_size'], CONFIG['hidden_size'])
        self.conv3 = GCNConv(CONFIG['hidden_size'], 1)
        self.dropout = nn.Dropout(CONFIG['dropout'])
        
    def forward(self, x, edge_index):
        x = self.dropout(F.relu(self.conv1(x, edge_index)))
        x = self.dropout(F.relu(self.conv2(x, edge_index)))
        return self.conv3(x, edge_index)

def chamfer_distance(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    dist_pred_to_gt = cdist(pred_np, gt_np)
    min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)
    dist_gt_to_pred = cdist(gt_np, pred_np)
    min_dist_gt_to_pred = np.min(dist_gt_to_pred, axis=1)
    chamfer = np.mean(min_dist_pred_to_gt) + np.mean(min_dist_gt_to_pred)
    return chamfer

def surface_normal_consistency(pred: torch.Tensor, gt: torch.Tensor, k: int = 5) -> float:
    try:
        pred_np = pred.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()
        def compute_normals(points):
            nn = NearestNeighbors(n_neighbors=k+1).fit(points)
            indices = nn.kneighbors(points, return_distance=False)[:, 1:]
            normals = []
            for i in range(len(points)):
                neighbors = points[indices[i]]
                if len(neighbors) >= 3:
                    centered = neighbors - points[i]
                    _, _, vh = np.linalg.svd(centered)
                    normal = vh[-1] / np.linalg.norm(vh[-1] + 1e-8)
                    normals.append(normal)
                else:
                    normals.append([0, 0, 1])
            return np.array(normals)
        pred_normals = compute_normals(pred_np)
        gt_normals = compute_normals(gt_np)
        dots = np.abs(np.sum(pred_normals * gt_normals, axis=1))
        return np.mean(dots)
    except:
        return np.nan

def compute_rmse_per_coord(pred: torch.Tensor, gt: torch.Tensor) -> Tuple[float, float, float]:
    errors = (pred - gt) ** 2
    rmse_x = torch.sqrt(torch.mean(errors[:, 0])).item()
    rmse_y = torch.sqrt(torch.mean(errors[:, 1])).item()
    rmse_z = torch.sqrt(torch.mean(errors[:, 2])).item()
    return rmse_x, rmse_y, rmse_z

def save_sample_points(pred_pos: torch.Tensor, gt_pos: torch.Tensor, filename: str):
    os.makedirs("results", exist_ok=True)
    with open(filename, 'w') as f:
        f.write("Sample Predicted vs. Ground Truth Points (First 5):\n")
        for p, g in zip(pred_pos[:5].cpu().numpy(), gt_pos[:5].cpu().numpy()):
            f.write(f"Pred: {p}, GT: {g}\n")

def train_neural_network(model: nn.Module, data: Data, target: torch.Tensor, skip_mask: torch.Tensor,
                        num_epochs: int, lr: float, model_name: str) -> Tuple[nn.Module, List[float]]:
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=CONFIG['weight_decay']
    )
    
    def get_lr(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return lr * (epoch + 1) / CONFIG['warmup_epochs']
        progress = (epoch - CONFIG['warmup_epochs']) / (num_epochs - CONFIG['warmup_epochs'])
        return lr * 0.5 * (1 + math.cos(math.pi * progress))
    
    loss_fn = nn.MSELoss()
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    model.train()
    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(epoch)
            
        optimizer.zero_grad()
        
        if 'PointNet' in model_name:
            pred = model(data.x.unsqueeze(0)).squeeze(0)
        else:
            pred = model(data.x, data.edge_index)
            
        pred_skip = pred[skip_mask]
        target_skip = target[skip_mask].unsqueeze(1)
        loss = loss_fn(pred_skip, target_skip)
        
        if torch.isnan(loss):
            print(f"⚠️ NaN loss at epoch {epoch}")
            break
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                print(f"🛑 Early stopping at epoch {epoch}")
                break
                
    return model, losses

def run_baseline_methods(points: np.ndarray, keep_mask: np.ndarray, pts_keep: np.ndarray, pts_skip: np.ndarray, frame: str) -> Dict:
    results = {}
    gt_pos_skip = torch.tensor(pts_skip, dtype=torch.float32).to(device)
    
    # Linear interpolation
    try:
        start_time = time.time()
        interp_coords = griddata(pts_keep[:, :2], pts_keep[:, 2], pts_skip[:, :2], method='linear', fill_value=np.nan)
        valid_mask = ~np.isnan(interp_coords)
        if valid_mask.sum() > 0:
            interp_pred = pts_skip[valid_mask].copy()
            interp_pred[:, 2] = interp_coords[valid_mask]
            interp_gt = pts_skip[valid_mask]
            pred_pos = torch.tensor(interp_pred, dtype=torch.float32).to(device)
            rmse_z = np.sqrt(np.mean((interp_pred[:, 2] - interp_gt[:, 2])**2))
            rmse_xyz = rmse_z / np.sqrt(3)
            rmse_x, rmse_y, rmse_z = 0.0, 0.0, rmse_z
            results['linear_interp'] = {
                'rmse_xyz': rmse_xyz,
                'rmse_x': rmse_x,
                'rmse_y': rmse_y,
                'rmse_z': rmse_z,
                'time': time.time() - start_time,
                'chamfer': chamfer_distance(pred_pos, gt_pos_skip[valid_mask]),
                'normal_consistency': surface_normal_consistency(pred_pos, gt_pos_skip[valid_mask])
            }
            save_sample_points(pred_pos, gt_pos_skip[valid_mask], f"results/{frame}_linear_interp.txt")
        else:
            results['linear_interp'] = {'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan, 'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan}
    except Exception as e:
        print(f"❌ Linear interpolation failed: {e}")
        results['linear_interp'] = {'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan, 'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan}
    
    # Nearest neighbor
    try:
        start_time = time.time()
        nn_model = NearestNeighbors(n_neighbors=1)
        nn_model.fit(pts_keep)
        _, indices = nn_model.kneighbors(pts_skip)
        nn_pred = pts_keep[indices[:, 0]]
        pred_pos = torch.tensor(nn_pred, dtype=torch.float32).to(device)
        rmse_xyz = np.sqrt(np.mean((nn_pred - pts_skip)**2))
        rmse_x, rmse_y, rmse_z = compute_rmse_per_coord(pred_pos, gt_pos_skip)
        results['nearest_neighbor'] = {
            'rmse_xyz': rmse_xyz,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'time': time.time() - start_time,
            'chamfer': chamfer_distance(pred_pos, gt_pos_skip),
            'normal_consistency': surface_normal_consistency(pred_pos, gt_pos_skip)
        }
        save_sample_points(pred_pos, gt_pos_skip, f"results/{frame}_nearest_neighbor.txt")
    except Exception as e:
        print(f"❌ NN failed: {e}")
        results['nearest_neighbor'] = {'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan, 'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan}
    
    return results

def run_neural_methods(points: np.ndarray, keep_mask: np.ndarray, pts_skip: np.ndarray, beam_indices: np.ndarray, frame: str) -> Dict:
    results = {}
    gt_pos_skip = torch.tensor(pts_skip, dtype=torch.float32).to(device)
    
    pos = torch.tensor(points[:, :3], dtype=torch.float32).to(device)
    beam_indices_tensor = torch.tensor(beam_indices, dtype=torch.float32).to(device).unsqueeze(1) / 64.0
    
    input_features = pos.clone()
    z_masked = torch.zeros((points.shape[0]), dtype=torch.float32).to(device)
    z_masked[keep_mask] = pos[keep_mask, 2]
    input_features[:, 2] = z_masked
    input_features = torch.cat((input_features, beam_indices_tensor), dim=1)
    
    # Normalization
    feature_mean = input_features.mean(dim=0)
    feature_std = input_features.std(dim=0) + 1e-6
    input_norm = (input_features - feature_mean) / feature_std
    
    z_gt = pos[:, 2]
    z_mean = z_gt.mean()
    z_std = z_gt.std() + 1e-6
    y_norm = (z_gt - z_mean) / z_std
    
    skip_mask_tensor = torch.tensor(~keep_mask, dtype=torch.bool).to(device)
    
    # SuperiorGAT
    try:
        start_time = time.time()
        edge_index = knn_graph(pos, k=CONFIG['k'], loop=False).to(device)
        data = Data(x=input_norm, edge_index=edge_index, y=y_norm)
        
        model = SuperiorGAT(
            in_channels=4
        ).to(device)
        
        model, losses = train_neural_network(
            model, data, y_norm, skip_mask_tensor,
            CONFIG['num_epochs'], CONFIG['learning_rates']['superior_gat'],
            "SuperiorGAT"
        )
        
        model.eval()
        with torch.no_grad():
            pred_norm = model(data.x, data.edge_index)
            pred_z = pred_norm[:, 0] * z_std + z_mean
            
        pred_pos_skip = gt_pos_skip.clone()
        pred_pos_skip[:, 2] = pred_z[skip_mask_tensor]
        
        rmse_xyz = torch.sqrt(torch.mean((pred_pos_skip - gt_pos_skip) ** 2)).item()
        rmse_x, rmse_y, rmse_z = compute_rmse_per_coord(pred_pos_skip, gt_pos_skip)
        chamfer = chamfer_distance(pred_pos_skip, gt_pos_skip)
        normal_consistency = surface_normal_consistency(pred_pos_skip, gt_pos_skip)
        
        results['superior_gat'] = {
            'rmse_xyz': rmse_xyz,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'time': time.time() - start_time,
            'chamfer': chamfer,
            'normal_consistency': normal_consistency,
            'convergence_epoch': len(losses)
        }
        
        print(f"✅ SuperiorGAT RMSE_XYZ: {rmse_xyz:.4f}, Chamfer: {chamfer:.4f}")
        save_sample_points(pred_pos_skip, gt_pos_skip, f"results/{frame}_superior_gat.txt")
        
        del model, data, pred_norm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ SuperiorGAT failed: {str(e)}")
        results['superior_gat'] = {
            'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan,
            'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan,
            'convergence_epoch': np.nan
        }

    #########################################################################################################################

    # Baseline GAT (vanilla, no residual/LayerNorm)
    try:
        start_time = time.time()
        # You may reuse the earlier edge_index; recompute here for clarity
        edge_index = knn_graph(pos, k=CONFIG['k'], loop=False).to(device)
        data = Data(x=input_norm, edge_index=edge_index, y=y_norm)

        model = BaselineGAT(in_channels=4).to(device)
        # Use the same LR as SuperiorGAT for fairness, or set a local one (e.g., 0.001)
        lr_gat_baseline = CONFIG['learning_rates'].get('superior_gat', 0.0005)

        model, losses = train_neural_network(
            model, data, y_norm, skip_mask_tensor,
            CONFIG['num_epochs'], lr_gat_baseline,
            "Baseline_GAT"
        )

        model.eval()
        with torch.no_grad():
            pred_norm = model(data.x, data.edge_index)
            pred_z = pred_norm[:, 0] * z_std + z_mean

        pred_pos_skip = gt_pos_skip.clone()
        pred_pos_skip[:, 2] = pred_z[skip_mask_tensor]

        rmse_xyz = torch.sqrt(torch.mean((pred_pos_skip - gt_pos_skip) ** 2)).item()
        rmse_x, rmse_y, rmse_z = compute_rmse_per_coord(pred_pos_skip, gt_pos_skip)
        chamfer = chamfer_distance(pred_pos_skip, gt_pos_skip)
        normal_consistency = surface_normal_consistency(pred_pos_skip, gt_pos_skip)

        results['gat_baseline'] = {
            'rmse_xyz': rmse_xyz,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'time': time.time() - start_time,
            'chamfer': chamfer,
            'normal_consistency': normal_consistency,
            'convergence_epoch': len(losses)
        }

        print(f"✅ Baseline GAT RMSE_XYZ: {rmse_xyz:.4f}, Chamfer: {chamfer:.4f}")
        save_sample_points(pred_pos_skip, gt_pos_skip, f"results/{frame}_gat_baseline.txt")

        del model, data, pred_norm
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Baseline GAT failed: {str(e)}")
        results['gat_baseline'] = {
            'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan,
            'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan,
            'convergence_epoch': np.nan
        }







    ##########################################################################################################################
    
    # Enhanced PointNet
    try:
        start_time = time.time()
        model = EnhancedPointNet(in_channels=4).to(device)
        data = Data(x=input_norm, y=y_norm)
        model, losses = train_neural_network(
            model, data, y_norm, skip_mask_tensor, CONFIG['num_epochs'], 
            CONFIG['learning_rates']['pointnet'], "Enhanced_PointNet"
        )
        model.eval()
        with torch.no_grad():
            pred_norm = model(input_norm.unsqueeze(0)).squeeze(0)
            pred_z = pred_norm[:, 0] * z_std + z_mean
        pred_pos_skip = gt_pos_skip.clone()
        pred_pos_skip[:, 2] = pred_z[skip_mask_tensor]
        rmse_xyz = torch.sqrt(torch.mean((pred_pos_skip - gt_pos_skip) ** 2)).item()
        rmse_x, rmse_y, rmse_z = compute_rmse_per_coord(pred_pos_skip, gt_pos_skip)
        chamfer = chamfer_distance(pred_pos_skip, gt_pos_skip)
        normal_consistency = surface_normal_consistency(pred_pos_skip, gt_pos_skip)
        results['enhanced_pointnet'] = {
            'rmse_xyz': rmse_xyz,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'time': time.time() - start_time,
            'chamfer': chamfer,
            'normal_consistency': normal_consistency
        }
        print(f"✅ Enhanced PointNet RMSE_XYZ: {rmse_xyz:.4f}, Chamfer: {chamfer:.4f}")
        save_sample_points(pred_pos_skip, gt_pos_skip, f"results/{frame}_enhanced_pointnet.txt")
        del model, data, pred_norm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Enhanced PointNet failed: {e}")
        results['enhanced_pointnet'] = {'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan, 'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan}
    
    # Simple GCN
    try:
        start_time = time.time()
        edge_index = knn_graph(pos, k=CONFIG['k'], loop=False).to(device)
        data = Data(x=input_norm, edge_index=edge_index, y=y_norm)
        model = SimpleGCN(in_channels=4).to(device)
        model, losses = train_neural_network(
            model, data, y_norm, skip_mask_tensor, CONFIG['num_epochs'], 
            CONFIG['learning_rates']['gcn'], "Simple_GCN"
        )
        model.eval()
        with torch.no_grad():
            pred_norm = model(data.x, data.edge_index)
            pred_z = pred_norm[:, 0] * z_std + z_mean
        pred_pos_skip = gt_pos_skip.clone()
        pred_pos_skip[:, 2] = pred_z[skip_mask_tensor]
        rmse_xyz = torch.sqrt(torch.mean((pred_pos_skip - gt_pos_skip) ** 2)).item()
        rmse_x, rmse_y, rmse_z = compute_rmse_per_coord(pred_pos_skip, gt_pos_skip)
        chamfer = chamfer_distance(pred_pos_skip, gt_pos_skip)
        normal_consistency = surface_normal_consistency(pred_pos_skip, gt_pos_skip)
        results['simple_gcn'] = {
            'rmse_xyz': rmse_xyz,
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'rmse_z': rmse_z,
            'time': time.time() - start_time,
            'chamfer': chamfer,
            'normal_consistency': normal_consistency
        }
        print(f"✅ Simple GCN RMSE_XYZ: {rmse_xyz:.4f}, Chamfer: {chamfer:.4f}")
        save_sample_points(pred_pos_skip, gt_pos_skip, f"results/{frame}_simple_gcn.txt")
        del model, data, pred_norm
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Simple GCN failed: {e}")
        results['simple_gcn'] = {'rmse_xyz': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan, 'time': np.nan, 'chamfer': np.nan, 'normal_consistency': np.nan}
    
    return results

def stratified_sample(points: np.ndarray, beam_indices: np.ndarray, num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    unique_beams = np.unique(beam_indices)
    points_per_beam = len(points) // max(1, len(unique_beams))
    target_per_beam = num_points // max(1, len(unique_beams))
    indices = []
    for beam in unique_beams:
        beam_mask = beam_indices == beam
        beam_indices_i = np.where(beam_mask)[0]
        if len(beam_indices_i) > target_per_beam:
            selected = np.random.choice(beam_indices_i, target_per_beam, replace=False)
        else:
            selected = beam_indices_i
        indices.extend(selected)
    indices = np.array(indices)
    if len(indices) > num_points:
        indices = np.random.choice(indices, num_points, replace=False)
    elif len(indices) < num_points:
        extra = np.random.choice(np.arange(len(points)), num_points - len(indices), replace=True)
        indices = np.concatenate([indices, extra])
    return points[indices], beam_indices[indices]

def main():
    lidar_files = sorted([f for f in os.listdir(CONFIG['data_path']) if f.endswith('.bin')])
    os.makedirs("results", exist_ok=True)
    all_results = []
    print(f"🚀 Starting LiDAR reconstruction analysis")

    '''
    THIS NEXT LINE IF YOU WANT TO LIMIT NUMBER OF FRAMES RGIHT NOW IS 100
    '''
    lidar_files = lidar_files[:108]
    
    for idx, fname in enumerate(lidar_files):
        print(f"\n🌀 Frame {idx+1}/{len(lidar_files)}: {fname}")
        path = os.path.join(CONFIG['data_path'], fname)
        try:
            points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)[:, :3]
            if points.shape[0] < CONFIG['min_points']:
                raise ValueError(f"Point cloud too small: {points.shape[0]} points")
            if np.any(np.isnan(points)):
                raise ValueError("NaN values detected in point cloud")
            
            vertical_angles = np.degrees(np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], axis=1)))
            beam_angles = np.linspace(-24.8, 2.0, 65)
            beam_indices = np.digitize(vertical_angles, beam_angles) - 1
            beam_indices = np.clip(beam_indices, 0, 63)
            
            if CONFIG['num_points'] is not None and len(points) > CONFIG['num_points']:
                points, beam_indices = stratified_sample(points, beam_indices, CONFIG['num_points'])
            
            keep_mask = (beam_indices % 4) != 0
            skip_mask = ~keep_mask
            
            if skip_mask.sum() < CONFIG['min_points'] or keep_mask.sum() < CONFIG['min_points']:
                raise ValueError(f"Insufficient points: kept={keep_mask.sum()}, dropped={skip_mask.sum()}")
            
            pts_keep = points[keep_mask]
            pts_skip = points[skip_mask]
            
            print(f"Points before dropout: {len(points)}")
            print(f"Points after dropout: {keep_mask.sum()} ({keep_mask.sum()/len(points):.1%} remaining)")
            print(f"Beams dropped: {len(np.unique(beam_indices[skip_mask]))}")
            
            baseline_results = run_baseline_methods(points, keep_mask, pts_keep, pts_skip, fname)
            neural_results = run_neural_methods(points, keep_mask, pts_skip, beam_indices, fname)
            
            frame_result = {'frame': fname}
            frame_result.update(baseline_results)
            frame_result.update(neural_results)
            all_results.append(frame_result)
            
            print(f"📈 Frame {idx+1} Summary:")
            for method, metrics in {**baseline_results, **neural_results}.items():
                if isinstance(metrics, dict) and 'rmse_xyz' in metrics:
                    rmse_xyz = metrics['rmse_xyz']
                    time_taken = metrics['time']
                    rmse_x = metrics.get('rmse_x', np.nan)
                    rmse_y = metrics.get('rmse_y', np.nan)
                    rmse_z = metrics.get('rmse_z', np.nan)
                    rmse_str = f"{rmse_xyz:.4f}" if not np.isnan(rmse_xyz) else "nan"
                    time_str = f"{time_taken:.2f}" if not np.isnan(time_taken) else "nan"
                    print(f"  {method}: RMSE_XYZ={rmse_str}, RMSE_X={rmse_x:.4f}, RMSE_Y={rmse_y:.4f}, RMSE_Z={rmse_z:.4f}, Time={time_str}s")
            
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error in {fname}: {e}")
            all_results.append({'frame': fname, 'error': str(e)})
    
    df = pd.DataFrame(all_results)
    df.to_csv("results/reconstruction_analysis.csv", index=False)
    print(f"\n📊 ANALYSIS COMPLETE")
    print(f"💾 Results saved to: results/reconstruction_analysis.csv")
    methods = [col for col in df.columns if col not in ['frame', 'error']]
    for method in methods:
        try:
            avg_rmse_xyz = df.apply(lambda row: row[method]['rmse_xyz'] if method in row and isinstance(row[method], dict) and 'rmse_xyz' in row[method] else np.nan, axis=1).mean()
            print(f"  {method}: average RMSE_XYZ = {avg_rmse_xyz:.4f}")
        except:
            pass

if __name__ == "__main__":
    main()
