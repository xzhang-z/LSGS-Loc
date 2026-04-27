# **LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios**

📄 Paper: [PDF](https://arxiv.org/abs/2604.05402)

![LSGS-Loc Workflow](assets/workflow.png)

This repository contains the code for "LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios".

## Environment Setup

```bash
# 1. Create environment
conda create -n lsgs_loc python=3.10 -y
conda activate lsgs_loc

# 2. Install PyTorch with CUDA 11.8
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Install gsplat example dependencies
cd third_party/gsplat/examples
pip install -r requirements.txt

# 4. Install gsplat
pip install gsplat==1.5.3

# 5. Install remaining dependencies
pip install scikit-image natsort safetensors ninja kornia huggingface_hub einops fast-pytorch-kmeans
```

After finishing the environment setup, please also follow the AnyLoc demo instructions to download and prepare **Cluster Centers**:

- [AnyLoc Demo (Cluster Centers preparation)](https://github.com/AnyLoc/AnyLoc/tree/main/demo)

## Data Preparation

We use the [GauUScene](https://saliteta.github.io/anonymous_project/) dataset.

- Project page: [GauUScene Dataset](https://saliteta.github.io/anonymous_project/)
- After downloading, place the dataset under the `data/` folder.

## Main Scripts

- `scripts/lsgs_loc_demo.py`: full pipeline of the paper method .
- `scripts/lsgs_loc_full_pipeline.py`: end-to-end pipeline including data preprocessing (train/test split) and 3DGS training.

## Usage

### Option A: Run full pipeline from raw COLMAP project

You can directly run `scripts/lsgs_loc_full_pipeline.py` to obtain localization results from a raw COLMAP dataset.

```bash
python scripts/lsgs_loc_full_pipeline.py \
  --colmap_project /path/to/your_colmap_scene \
  --output_root /path/to/output \
  --trainer_cuda_visible_devices 0
```

Required arguments:

- `--colmap_project`: input COLMAP project root, must contain `images/` and `sparse/0` (or `sparse/`).
- `--output_root`: output root directory.

<details>
<summary>Commonly used optional arguments</summary>

- `--trainer_cuda_visible_devices`: GPU id(s) used by training/inference subprocesses (for example `0` or `0,1`).
- `--do_resize`: enable COLMAP/image resizing before split.
- `--resize_width`: max edge when resizing (default `1600`, currently only `1600` is supported/recommended).
- `--image_mode`: image materialization mode for split subsets, `copy` or `symlink` (default `copy`).
- `--retrieval_top_k`: retrieval top-k (default `3`).
- `--render_opt_image_size`: render optimization size, `original` or `1600`.
- `--force_restart_all`: delete existing intermediates and rerun from scratch.

</details>

Notes on settings:

- `--trainer_cuda_visible_devices` and `CUDA_VISIBLE_DEVICES` are usually interchangeable for this script.
- If `--trainer_cuda_visible_devices` is set, you typically do not need to additionally set `CUDA_VISIBLE_DEVICES`.
- `output_root` will contain split data, gsplat checkpoints, and final pose results.

### Option B  (Recommended): Run step-by-step

You can also run each stage manually.

#### Step 1: Data preprocessing (split raw COLMAP into train/test)

```bash
python -m src.utils.split_colmap \
	--colmap_project /path/to/your_colmap_scene \
	--output_root /path/to/split_output \
	--image_mode copy
```

<details>
<summary>Preprocessing script arguments</summary>

- `--colmap_project` (required): input COLMAP project directory, must contain `images/` and `sparse/0` (or `sparse/`).
- `--output_root` (required): output directory where `train/`, `test/`, and `list_test.txt` will be generated.
- `--image_mode` (optional): `copy` or `symlink`, default is `copy`.

</details>

This step will generate:

- `train/` COLMAP subset
- `test/` COLMAP subset
- `list_test.txt` test image list

#### Step 2: Train 3DGS scene on the training split

After splitting, follow the [gsplat](https://github.com/nerfstudio-project/gsplat) project practice to train a 3DGS scene on the `train/` subset and obtain the 3DGS checkpoint/scene for localization.


#### Step 3: Run LSGS-Loc workflow

After preparing `train/`, `test/`, and a trained 3DGS representation, run the workflow script directly.

Use checkpoint input:

```bash
python scripts/lsgs_loc_demo.py \
  --database_root /path/to/split_output/train \
  --query_root /path/to/split_output/test \
  --output_dir /path/to/workflow_output \
  --ckpt /path/to/3dgs_ckpt.pt
```

Or use PLY input:

```bash
python scripts/lsgs_loc_demo.py \
  --database_root /path/to/split_output/train \
  --query_root /path/to/split_output/test \
  --output_dir /path/to/workflow_output \
  --ply /path/to/point_cloud_30000.ply
```

Notes:

- Exactly one of `--ckpt` or `--ply` must be provided.
- `--ply` expects the gsplat export format (for example, files saved by `simple_trainer.py` via `save_ply`).
- `scripts/lsgs_loc_demo.py` is the user-facing alias of the same workflow implementation.

## License

This project is licensed under MIT. See [LICENSE](LICENSE) for details.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{zhang2026lsgs,
  title={LSGS-Loc: Towards Robust 3DGS-Based Visual Localization for Large-Scale UAV Scenarios},
  author={Zhang, Xiang and Wang, Tengfei and Xu, Fang and Wang, Xin and Zhan, Zongqian},
  journal={arXiv preprint arXiv:2604.05402},
  year={2026}
}
```

## Acknowledgements

We gratefully acknowledge the following projects:

- [gsplat](http://www.gsplat.studio/)
- [AnyLoc](https://anyloc.github.io/)
- [ReLoc3r](https://github.com/ffrivera0/reloc3r)
