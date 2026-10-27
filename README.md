# GradCO

PyTorch implementation of GradCO for text-based person anomaly search and text-based person retrieval.

GradCO combines Behavior-Aware Mixture-of-Experts, Hierarchical Intent Alignment, and Conflict Offload Learning to model appearance, normal behavior, and anomalous behavior in image-text retrieval.

## Method

The model is implemented in `models/model_X2VLM.py` as `GradCO`. It uses separate visual and textual behavior-aware experts, hierarchical image-text alignment, and dynamic anchors with directional regularization for conflict offloading.

The supplied configurations use BEiT-v2 as the image encoder and BERT as the text encoder. The forward pass returns contrastive, image-text matching, and masked language modeling losses, with conflict-offload regularization added to the masked language modeling loss. The configurations also include image masking settings used by the backbone.

## Datasets

### PAB

PAB is used for Text-based Person Anomaly Search. Obtain the images and annotations from the original [PAB/CMP project](https://github.com/Shuyu-XJTU/CMP) and follow its dataset usage terms.

PAB training annotations contain `image`, `caption`, and `image_id`. When hard-negative sampling is enabled, they also contain `hard_i`, `hard_c`, and `hard_i_id`. The loader resolves `hard_i` relative to `image_root/train/`.

### Text-based Person Retrieval

The repository also provides configurations for CUHK-PEDES, ICFG-PEDES, RSTPReid, and UFine6926/UFine3C. Prepare the corresponding images and JSON annotations and update their paths in the selected configuration. The supplied UFine configuration sets `ifSpecial: True`, selecting the test split of UFine6926. Set `ifSpecial: False` to select the configured UFine3C annotations and image root instead.

## Models and Weights

Prepare an X2-VLM pretrained checkpoint compatible with the BEiT-v2/BERT model and a BERT tokenizer/configuration directory. Datasets and model weights are not included in this code package.

```text
checkpoint/
  x2vlm_base_1b.th
bert-base-uncased/
  config.json
  vocab.txt
  tokenizer_config.json
```

The text encoder can also be specified by the Hugging Face identifier `bert-base-uncased`. For offline use, set `text_encoder` in the task YAML to the local directory. Its name must contain `bert-base-uncased` for the current tokenizer selection logic.

The pretrained loader expects projection weights such as `text_proj.weight` and `text_proj.bias`, and remaps pretrained weights into the GradCO model. Check the loading messages to confirm checkpoint compatibility.

## Usage

### Install Requirements

Use Python 3.10 and a Linux environment with NVIDIA GPUs. The launcher uses POSIX shell syntax and distributed training uses NCCL.

```bash
conda create -n gradco python=3.10
conda activate gradco
```

Install a matching PyTorch and torchvision pair for your CUDA version. For example, for CUDA 11.8:

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Download the WordNet resources used for text augmentation:

```bash
python -m nltk.downloader wordnet omw-1.4
```

The dependency list reflects imports in this repository. Exact package compatibility has not been validated in a complete training run; the custom text encoders use Transformers 4.x APIs.

### Parameter Initialization

Set the model and tokenizer paths in `configs/<task>_X2VLM.yaml` and pass the combined pretrained checkpoint with `--checkpoint`. The supplied task configurations select `configs/config_beit2_base.json` and disable separate backbone parameter loading during GradCO construction.

Use `--num_sub_experts 4` and `--col_lambda_dir 0.1` for the settings used by the supplied launcher. These arguments override the corresponding YAML settings.

### Datasets Prepare

Organize PAB as follows:

```text
 data/PAB/
   annotation/
     train/
       attr_0.json
       ...
       attr_7.json
     test/
       attr.json
       ucc.json
     multi-weather/
       multi-weather_wind.json
       multi-weather_rain.json
       multi-weather_snow.json
       multi-weather_rain_snow.json
       multi-weather_dark.json
       multi-weather_dark_wind.json
       multi-weather_dark_rain.json
       multi-weather_dark_snow.json
       multi-weather_light.json
   train/
   test/
   ucc/
```

The supplied PAB configuration enables training shards 0 through 7. Add other shards to `train_file` if required. Pose images are needed only when `be_pose_img` is enabled; the supplied configurations disable it.

Update `image_root`, `image_root_test` (where present), `train_file`, `val_file`, and `test_file` in the task YAML. Run commands from the repository root. `--data_root` controls annotation paths in the standalone PAB evaluation loop and does not rewrite the YAML image or training paths; pass it explicitly rather than relying on the current `/data/PAB` default.

| Task | Configuration |
| --- | --- |
| PAB | `configs/tpas_X2VLM.yaml` |
| CUHK-PEDES | `configs/itr_cuhk_X2VLM.yaml` |
| ICFG-PEDES | `configs/itr_icfg_X2VLM.yaml` |
| RSTPReid | `configs/itr_rstp_X2VLM.yaml` |
| UFine6926 / UFine3C | `configs/itr_ufine_X2VLM.yaml` |

### Training

Single-GPU PAB training:

```bash
python run_TPS.py --task tpas --devices 0 --nproc 1 --checkpoint checkpoint/x2vlm_base_1b.th --output_dir output/gradco_tpas --data_root data/PAB --num_sub_experts 4 --col_lambda_dir 0.1
```

For four GPUs, replace `--devices 0 --nproc 1` with `--devices 0,1,2,3 --nproc 4`. Use one process per selected GPU. Other datasets use the task names `itr_cuhk`, `itr_icfg`, `itr_rstp`, and `itr_ufine` with their corresponding YAML configurations.

`--bs`, `--epo`, and `--lr` override the configured batch size, epoch count, and learning rate when positive. `--seed` defaults to 42. Use the distributed launcher even for one GPU because the training and evaluation loops call distributed barriers.

Training evaluates the configured test set each epoch and saves `log.txt`, `config.yaml`, and `checkpoint_best.pth` under `--output_dir`. The best checkpoint is selected by test R@1. The YAML copy is written before command-line overrides, so retain the launch command with your results.

### Evaluation

Training performs retrieval evaluation at each epoch and reports R@1, R@5, R@10, mAP, and mINP.

The standalone `--evaluate` branch is currently specific to PAB: it iterates over the original, OOD, and all nine multi-weather annotation files listed above, irrespective of the selected task. It is not a standalone evaluation entry point for the other retrieval benchmarks.

The evaluation entry point currently uses the pretrained-weight remapping loader rather than a dedicated trained-checkpoint loader. GradCO inherits the base text projection, so its state dictionary includes `text_proj` keys. However, the loader also remaps text encoder weights and overwrites `tl_text_proj` from `text_proj`; verify trained-checkpoint restoration before using standalone evaluation results.
