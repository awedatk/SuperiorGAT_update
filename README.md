# SuperiorGAT: Graph Attention Networks for Sparse LiDAR Point Cloud Reconstruction

This repository contains the reference implementation of **SuperiorGAT**, a graph attention network framework for reconstructing missing elevation information in sparse LiDAR point clouds under structured beam dropout.

## Overview

The proposed method represents LiDAR scans as beam-aware graphs and combines graph attention with gated residual fusion and a lightweight feed-forward refinement module to recover missing vertical geometry.

The implementation accompanies the manuscript:

**"SuperiorGAT: Graph Attention Networks for Sparse LiDAR Point Cloud Reconstruction in Autonomous Systems"**

## Features

* Structured beam dropout simulation
* SuperiorGAT model
* Baseline GAT implementation
* Simple GCN baseline
* Enhanced PointNet baseline
* Linear interpolation baseline
* Nearest neighbor baseline
* Evaluation using:

  * RMSE
  * Chamfer Distance
  * Surface Normal Consistency

## Datasets

This code was developed and evaluated using publicly available datasets:

* KITTI Vision Benchmark Suite
* nuScenes Dataset

The datasets are not included in this repository and must be obtained from their official sources.

## Requirements

* Python 3.10+
* PyTorch
* PyTorch Geometric
* NumPy
* SciPy
* scikit-learn
* pandas

## Citation

If you use this code, please cite the associated publication.

## License

This repository is provided for academic and research purposes.
