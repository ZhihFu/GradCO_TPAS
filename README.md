<a id="top"></a>

<div align="center">
  <img src="gradco-logo.png" alt="GradCO logo" width="150" />
  <h1>GradCO: Behavior-Aware Gradient-Conflict Offloading for Text-Based Person Anomaly Search</h1>
  <p>
    <a href="https://arxiv.org/abs/2610.37267"><strong>Zhiheng Fu</strong></a>,
    Zixu Li, Zhiwei Chen, Jiale Huang, Fangxu Liu, Jinhe Bi, Yupeng Hu
  </p>
  <p>
    <a href="https://arxiv.org/abs/2610.37267"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.37267-b31b1b.svg"></a>
    <a href="https://github.com/ZhihFu/GradCO_TPAS"><img alt="GitHub stars" src="https://img.shields.io/github/stars/ZhihFu/GradCO_TPAS?style=social"></a>
    <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
    <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  </p>
  <p>
    <a href="https://arxiv.org/abs/2610.37267">Paper</a> ·
    <a href="#method">Method</a> ·
    <a href="#results">Results</a> ·
    <a href="#getting-started">Get Started</a>
  </p>
</div>

## Abstract

Text-based Person Anomaly Search (TPAS) retrieves pedestrians from natural-language descriptions of both appearance and behavior. It must preserve identity cues while recognizing fine-grained normal and anomalous actions. GradCO addresses two challenges in this setting: conflicting gradients from same-identity, different-behavior hard negatives, and appearance features overwhelming behavioral evidence during retrieval.

GradCO introduces three complementary components: Behavior-Aware Mixture-of-Experts (B-MoE) separates appearance, normal-behavior, and anomalous-behavior streams; Conflict Offload Learning (COL) uses pair-conditioned anchors to offload conflicting gradients; and Hierarchical Intent Alignment (HIA) aggregates expert features and similarities with query-conditioned weights.

## News

- **2026-10-27:** GradCO preprint released on [arXiv](https://arxiv.org/abs/2610.37267).

## Table of Contents

- [Abstract](#abstract)
- [News](#news)
- [Method](#method)
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Method

The implementation is in [`models/model_X2VLM.py`](models/model_X2VLM.py). GradCO builds on an X²-VLM vision-language backbone and applies behavior-aware experts to both image and text representations.

| Component | Role |
| --- | --- |
| **B-MoE** | Learns appearance, normal-behavior, and anomalous-behavior expert streams from retrieval objectives, without expert-specific semantic labels. |
| **COL** | Introduces pair-conditioned offload anchors in similarity computation to reduce hard-negative gradient conflict on the main text representations. |
| **HIA** | Uses query-conditioned weights to combine expert similarities and features for retrieval. |

The supplied configurations use BEiT-v2 as the image encoder and BERT as the text encoder. Training combines contrastive, image-text matching, and masked language modeling losses with conflict-offload regularization.

## Benchmarks

GradCO is evaluated on Text-based Person Anomaly Search (TPAS) and Text-to-Image Person Retrieval (TIPR):

| Task | Benchmark / setting | Configuration |
| --- | --- | --- |
| TPAS | PAB | [`configs/tpas_X2VLM.yaml`](configs/tpas_X2VLM.yaml) |
| TPAS | PAB Multi-Weather and UCC OOD | [`configs/tpas_X2VLM.yaml`](configs/tpas_X2VLM.yaml) |
| TIPR | CUHK-PEDES | [`configs/itr_cuhk_X2VLM.yaml`](configs/itr_cuhk_X2VLM.yaml) |
| TIPR | ICFG-PEDES | [`configs/itr_icfg_X2VLM.yaml`](configs/itr_icfg_X2VLM.yaml) |
| TIPR | RSTPReID | [`configs/itr_rstp_X2VLM.yaml`](configs/itr_rstp_X2VLM.yaml) |
| TIPR | UFine6926 and UFine3C | [`configs/itr_ufine_X2VLM.yaml`](configs/itr_ufine_X2VLM.yaml) |

PAB is the in-distribution TPAS benchmark. Multi-Weather evaluates robustness to weather and lighting changes, while UCC is used for out-of-distribution evaluation. UFineBench includes the UFine6926 and UFine3C settings.

## Results

The following headline results are reported in the [arXiv v1 paper](https://arxiv.org/abs/2610.37267). Retrieval scores are percentages; see the paper for full tables, baselines, and evaluation protocols.

### TPAS

| Setting | GradCO R@1 | CMP R@1 | GradCO mAP | CMP mAP |
| --- | ---: | ---: | ---: | ---: |
| PAB, 0.1M gallery | **85.39** | 83.06 | **92.13** | 90.41 |
| Multi-Weather mean | **72.20** | 67.12 | **80.67** | 77.44 |
| UCC OOD, 1M gallery | **62.53** | 55.23 | **52.01** | 44.35 |

### TIPR

The paper reports the following scores in its `Avg` column:

| Benchmark | GradCO | Strongest prior result in the paper |
| --- | ---: | ---: |
| CUHK-PEDES | **88.87** | 88.77 (Bi-IRRA) |
| ICFG-PEDES | **79.85** | 79.79 (Bi-IRRA) |
| RSTPReID | **84.73** | 84.17 (Bi-IRRA) |
| UFine6926 | 95.13 | **95.18** (Bi-IRRA) |
| UFine3C | **82.01** | 74.63 (CFAM) |

## Getting Started

### Requirements

- Python 3.10
- Linux with NVIDIA GPUs; distributed training uses NCCL and the launcher uses POSIX shell syntax
- A PyTorch/torchvision pair compatible with the installed CUDA runtime

Create an environment and install dependencies. The following PyTorch command is an example for CUDA 11.8; choose the matching build for your system.

```bash
conda create -n gradco python=3.10
conda activate gradco

pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python -m nltk.downloader wordnet omw-1.4
```

The dependency list reflects imports in this repository. A complete end-to-end training environment has not been validated; the custom text encoders use Transformers 4.x APIs.

### Pretrained Weights

Datasets and model weights are not included. Prepare an X2-VLM checkpoint compatible with the selected backbone and a BERT tokenizer/configuration directory. The supplied configurations use `configs/config_beit2_base.json` and `bert-base-uncased`.

```text
checkpoint/
  x2vlm_base_1b.th
bert-base-uncased/
  config.json
  vocab.txt
  tokenizer_config.json
```

Set model and tokenizer paths in the selected task YAML and pass the pretrained checkpoint with `--checkpoint`. For offline use, point `text_encoder` to a local directory whose name contains `bert-base-uncased`, as required by the current tokenizer selection logic. The pretrained loader expects projection weights such as `text_proj.weight` and `text_proj.bias`; review its loading messages to confirm checkpoint compatibility.

## Data Preparation

Obtain PAB images and annotations from the original [PAB/CMP project](https://github.com/Shuyu-XJTU/CMP) and follow its dataset terms. The supplied PAB configuration uses training shards 0 through 7.

```text
data/PAB/
├── annotation/
│   ├── train/
│   │   ├── attr_0.json
│   │   └── ... attr_7.json
│   ├── test/
│   │   ├── attr.json
│   │   └── ucc.json
│   └── multi-weather/
│       └── multi-weather_*.json
├── train/
├── test/
└── ucc/
```

PAB training annotations contain `image`, `caption`, and `image_id`. When hard-negative sampling is enabled, they also contain `hard_i`, `hard_c`, and `hard_i_id`; `hard_i` is resolved relative to `image_root/train/`. Pose images are only needed when `be_pose_img` is enabled; the supplied configurations disable it.

For each task, prepare the corresponding image files and JSON annotations, then update `image_root`, `image_root_test` (where present), `train_file`, `val_file`, and `test_file` in the YAML. Set `ifSpecial: True` in the UFine configuration for UFine6926, or `False` for UFine3C.

The standalone PAB evaluation loop uses `--data_root` for annotation paths; this does not rewrite image or training paths in the YAML. Pass it explicitly rather than relying on the `/data/PAB` default.

## Training

Run commands from the repository root. Single-GPU PAB training:

```bash
python run_TPS.py \
  --task tpas \
  --devices 0 \
  --nproc 1 \
  --checkpoint checkpoint/x2vlm_base_1b.th \
  --output_dir output/gradco_tpas \
  --data_root data/PAB \
  --num_sub_experts 4 \
  --col_lambda_dir 0.1
```

For four GPUs, use `--devices 0,1,2,3 --nproc 4`. Use one process per selected GPU. Other task names are `itr_cuhk`, `itr_icfg`, `itr_rstp`, and `itr_ufine`, each mapped by the launcher to its corresponding YAML configuration.

`--bs`, `--epo`, and `--lr` override the configured batch size, epoch count, and learning rate when positive. `--seed` defaults to 42. Use the distributed launcher even for one GPU because the training and evaluation loops call distributed barriers.

Training evaluates the configured test set each epoch and writes `log.txt`, `config.yaml`, and `checkpoint_best.pth` under `--output_dir`. The best checkpoint is selected by test R@1. The saved YAML is written before command-line overrides, so retain the launch command alongside your results.

## Evaluation

Training reports R@1, R@5, R@10, mAP, and mINP. The standalone `--evaluate` branch is currently PAB-specific: it iterates over the original, OOD, and configured Multi-Weather annotation files regardless of the selected task. It is not a standalone evaluation entry point for the other TIPR benchmarks.

The current evaluation entry point uses the pretrained-weight remapping loader rather than a dedicated trained-checkpoint loader. Since GradCO inherits the base text projection, its state dictionary includes `text_proj` keys; however, the loader also remaps text encoder weights and overwrites `tl_text_proj` from `text_proj`. Verify trained-checkpoint restoration before relying on standalone evaluation results.

## Project Structure

```text
GradCO_TPAS/
├── gradco-logo.png         # Project logo
├── configs/                # TPAS and TIPR experiment configurations
├── dataset/                # Text-based person retrieval data loaders
├── dataset_TPS/            # PAB / TPAS data loaders
├── models/
│   └── model_X2VLM.py       # GradCO model
├── Search_TPS_X2VLM.py      # Training and task orchestration
├── run_TPS.py               # Distributed launcher
├── eval.py                  # TIPR evaluation metrics
└── eval_TPS.py              # TPAS evaluation metrics
```

## Citation

If you use this work, please cite the paper:

```bibtex
@misc{fu2026gradco,
  title = {GradCO: Behavior-Aware Gradient-Conflict Offloading for Text-Based Person Anomaly Search},
  author = {Zhiheng Fu and Zixu Li and Zhiwei Chen and Jiale Huang and Fangxu Liu and Jinhe Bi and Yupeng Hu},
  year = {2026},
  eprint = {2610.37267},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2610.37267}
}
```

## Acknowledgements

This implementation builds on [X²-VLM](https://arxiv.org/abs/2211.12402) and uses the [PAB/CMP](https://github.com/Shuyu-XJTU/CMP) benchmark. Please follow the corresponding code, checkpoint, and dataset terms.

## License

This project is released under the [MIT License](LICENSE).

<div align="center">
  <a href="#top">Back to top</a>
</div>
