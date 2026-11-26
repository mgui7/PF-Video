<h1 align="center"> JVT: Just Video Transformer makes
pixel-space video diffusion simple</h1>

<div align="center">
    <p>
        <a href="https://zhoutimemachine.github.io">Jianjun Zhou</a><sup>123*</sup>&nbsp;&nbsp;
        <a href="https://github.com/Yusux">Hua Geng</a><sup>1*</sup>&nbsp;&nbsp;
        <a href="https://github.com/ghy0324">Haoyu Guo</a><sup>3†</sup>
    </p>
    <p>
        <sup>1</sup>Zhejiang University &nbsp;&nbsp;&nbsp;
        <sup>2</sup>Shanghai Innovation Institute &nbsp;&nbsp;&nbsp;
        <sup>3</sup>Shanghai AI Laboratory
    </p>
    <p>
        <sup>*</sup> Equal Contribution &nbsp;&nbsp;&nbsp;
        <sup>†</sup> Corresponding Author
    </p>
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2611.18549" rel="external nofollow noopener" target="_blank"> <img src="https://img.shields.io/badge/arXiv-2611.18549-b31b1b.svg" alt="arXiv"></a>
    <a href="https://zhoutimemachine.github.io/JVT/" rel="external nofollow noopener" target="_blank"> <img src="https://img.shields.io/badge/project_page-0F80C0?logo=rocket&amp;logoColor=white" alt="Project Page"></a>
    <a href="https://huggingface.co/ZhouTimeMachine/JVT" rel="external nofollow noopener" target="_blank"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-checkpoints-yellow" alt="HuggingFace Checkpoints"></a>
    <a href="https://modelscope.cn/models/ZhouTimeMachine/JVT" rel="external nofollow noopener" target="_blank">
    <img src="https://img.shields.io/badge/ModelScope-checkpoints-blueviolet?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDEuNSAyMS4zIDYuOHYxMC40TDEyIDIyLjUgMi43IDE3LjJWNi44TDEyIDEuNXoiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PHBhdGggZD0iTTIuNyA2LjggMTIgMTIuMWw5LjMtNS4zTTEyIDEyLjF2MTAuNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=&logoColor=white" alt="ModelScope Checkpoints">
    </a>
    <a href="https://huggingface.co/datasets/ZhouTimeMachine/JVT" rel="external nofollow noopener" target="_blank"> <img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-dataset-yellow" alt="HuggingFace Dataset"></a>
    <a href="https://modelscope.cn/datasets/ZhouTimeMachine/JVT" rel="external nofollow noopener" target="_blank">
    <img src="https://img.shields.io/badge/ModelScope-dataset-blueviolet?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTEyIDEuNSAyMS4zIDYuOHYxMC40TDEyIDIyLjUgMi43IDE3LjJWNi44TDEyIDEuNXoiIGZpbGw9Im5vbmUiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMS44IiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+PHBhdGggZD0iTTIuNyA2LjggMTIgMTIuMWw5LjMtNS4zTTEyIDEyLjF2MTAuNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIxLjgiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=&logoColor=white" alt="ModelScope Dataset">
    </a>
</p>

<div align="center">
    <a href="https://zhoutimemachine.github.io/jvt/">
        <img src="assets/main.png" width="90%">
    </a>
    <p>
        <i>JVT generates videos directly in pixel space with a simple DiT-style transformer</i>
    </p>
</div>


## 📣 News
- **[Nov 16, 2026]** 🚀 The training and inference code for class-conditioned, unconditional and camera control is released.

## ⏳ Preparation

### 1. clone & install dependencies

```bash
git clone https://github.com/ZhouTimeMachine/JVT.git
cd JVT

conda create -n jvt python=3.12
conda activate jvt
conda install "ffmpeg>=7,<8"
pip install -r requirements.txt
```

### 2. download necessary checkpoints

```bash
# pretrained models
hf download ZhouTimeMachine/JVT --local-dir checkpoints
# modelscope download --repo-type model ZhouTimeMachine/JVT --local_dir checkpoints
# ms-hub download --repo-type model ZhouTimeMachine/JVT --local-dir checkpoints

# rgbd training, pose estimation
hf download yyfz233/Pi3X --local-dir checkpoints/Pi3X

# FVD evaluation
wget -P checkpoints https://github.com/JunyaoHu/common_metrics_on_video_quality/raw/main/fvd/styleganv/i3d_torchscript.pt
```

### 3. datasets preparation

Downloading and preprocessing datasets are described in [docs/dataset_preparation.md](docs/dataset_preparation.md).

## ⭐️ Inference

### 1. unconditional generation on 4 datasets

```bash
DATA=ffs  # sky, taichi
python evaluation/infer_uncond.py \
    batch_size=1 \
    output_dir=outputs/infer_${DATA} \
    resume=checkpoints/${DATA}.safetensors

# DATA=ucf101, with register guidance
python evaluation/infer_uncond.py \
    batch_size=1 \
    output_dir=outputs/infer_ucf101-uncond \
    algorithm.backbone.in_context_len=32 \
    algorithm.backbone.in_context_start=9 \
    algorithm.rg_scale=0.8 \
    resume=checkpoints/ucf101-uncond.safetensors
```

### 2. class-conditioned generation on UCF-101

```bash
python evaluation/infer_c2v.py \
    batch_size=1 \
    output_dir=outputs/infer_c2v \
    resume=checkpoints/ucf101-c2v.safetensors 
```

### 3. camera-conditioned generation on RealEstate10K

```bash
# RGB-only generation
python evaluation/infer_camctrl.py \
    dataloader.batch_size=1 \
    evaluation=infer_camctrl \
    output_dir=outputs/infer_re10k-rgb \
    resume=checkpoints/re10k-rgb.safetensors

# RGB-disparity joint generation
python evaluation/infer_camctrl.py \
    dataloader.batch_size=1 \
    evaluation=infer_camctrl_joint \
    output_dir=outputs/infer_re10k-rgbd \
    resume=checkpoints/re10k-rgbd.safetensors
    
```

## 🔥 Training

For the local Kinetics-600 setup (16 frames, 4 fps, 256px), including multi-node
launching and GPU batch-size profiling, see [K600 training](docs/k600_training.md).

Our full-scale training needs 8 NVIDIA 80GB GPUs, with the following scripts:

```bash
# 1. unconditional generation on 4 datasets
bash scripts/train_jvtuncond_ffs.sh
bash scripts/train_jvtuncond_sky.sh
bash scripts/train_jvtuncond_taichi.sh
bash scripts/train_jvtuncond_ucf101.sh

# 2. class-conditioned generation on UCF-101
bash scripts/train_jvtc2v.sh

# 3. camera-conditioned generation on RealEstate10K
bash scripts/train_jvtcamctrl.sh
bash scripts/train_jvtcamctrl_joint.sh
```

## 📈 Evaluation

```bash
# 1. unconditional generation on 4 datasets
bash scripts/eval_ffs.sh
bash scripts/eval_sky.sh
bash scripts/eval_taichi.sh
bash scripts/eval_ucf101-uncond.sh

# 2. class-conditioned generation on UCF-101
bash scripts/eval_ucf101-c2v.sh

# 3. camera-conditioned generation on RealEstate10K
bash scripts/eval_re10k-rgb.sh
bash scripts/eval_re10k-rgbd.sh
```

Details of evaluation metrics are described in [docs/evaluation.md](docs/evaluation.md).

## 🙏 Acknowledgements

Our work builds upon several fantastic open-source projects. We'd like to express our gratitude to the authors of:

  * [DFoT](https://github.com/kwsong0113/diffusion-forcing-transformer)
  * [JiT](https://github.com/LTH14/JiT)
  * [Latte](https://github.com/Vchitect/Latte)
  * [Pi3](https://github.com/yyfz/Pi3)
  * [VGGT](https://github.com/facebookresearch/vggt)

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@article{zhou2026jvt,
  title   = {{JVT}: Just Video Transformer makes pixel-space video diffusion simple},
  author  = {Jianjun Zhou and Hua Geng and Haoyu Guo},
  year    = {2026}
  journal = {arXiv preprint arXiv: 2611.18549}
}
```
