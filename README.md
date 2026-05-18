
<div align="center">
<h2><center>[ICML 2026] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation </h2>

[Sizhe Zhao<sup>1</sup>](), [Shengping Zhang<sup>1,2✉️</sup>](https://homepage.hit.edu.cn/zhangshengping),  [Shuo Yang<sup>1</sup>](), [Weiyu Zhao<sup>1</sup>](https://although-not-but.github.io/weiyu.github.io/), [Shuigen Wang<sup>3</sup>](), [Xiangyang Ji<sup>4</sup>]()

1 Harbin Institute of Technology,  2 Harbin Institute of Technology (Weihai) Qingdao Research Institute,  
3 Iray Technology co., Ltd.,  4. Tsinghua University

[![Project](https://img.shields.io/badge/Project-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://aipixel.github.io/TapSampling) [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)]() [![Hugging Face Collection](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/SizheZhao/tapsampling)

</div>

## 🛠️ Installation

```bash
git clone https://github.com/aipixel/TapSampling.git
cd TapSampling
export TAPS_PATH="$(pwd)"

conda create -n taps python==3.10
conda activate taps
cd calvin
bash install.sh
pip install -r requirements.txt
pip install "dlimp @ git+https://github.com/kvablack/dlimp.git"
pip install "flash-attn==2.5.5" --no-build-isolation
```

## 📦 Dataset/Checkpoints Download

### Download CALVIN ABC->D dataset
If you want to train the model, the full CALVIN ABC->D dataset (~517 GB) should be downloaded. For testing with the released checkpoint, only a subset of the dataset needs to be downloaded.

```bash
# Option 1: Download the full dataset
cd calvin/dataset
bash download_data.sh ABC

# Option 2: Download only the subset required for inference
cd calvin/dataset
bash download_part_data.sh      # TODO: Upload, Create this script!
```

After the download is complete, the dataset directory structure should be:

```text
calvin/dataset/task_ABC_D
├── training
└── validation
```

### Download Policy (VPP) Checkpoints
TapSampling is a policy-agnostic inference-time sampling framework. We take the VPP policy as an example.

```bash
# Download the VPP policy checkpoints
python video-prediction-policy/download_vpp_checkpoints.py
```

After the download is complete, the VPP checkpoints directory structure should be:

```text
video-prediction-policy/official_checkpoints
├── clip-vit-base-patch32/
├── dp-calvin/
└── svd-robot-calvin-ft/
```

### Download TapSampling Checkpoints

```bash
# Download the TapSampling checkpoints
python tapsampling/download_base_model_checkpoints.py
python tapsampling/download_tapsampling_checkpoints.py
```

After the download is complete, the checkpoints directory structure should be:

```text
tapsampling/pretrained_models
├── configs/
├── prism-qwen25-extra-dinosiglip-224px-0_5b/
    └── checkpoints/step-020792-epoch-01-loss=0.5268.pt
├── Qwen2.5-0.5B/
├── vit_large_patch14_reg4_dinov2.lvd142m/
└── ViT-SO400M-14-SigLip/
```

```text
action_vae/mvae/mvae_24_split
├── checkpoint_50000.pt
└── config.yaml
```

```text
tapsampling/official_checkpoint/last
├── lora_adapter/
├── action_head--65000_checkpoint.pt
└── (other files)
```

## 🚀 Training

### Training the Action-VAE
```text
cd "$TAPS_PATH/action_vae"

# Only run once to create an action file: $TAPS_PATH/action_vae/actions.h5
python prepare_actions.py --dataset_dir ../calvin/dataset/task_ABC_D/training

# Train. Check the script for detail configurations (e.g. checkpoint path and output path).
bash train_vae.sh
```

### Training the TapSampling Verifier
```text
cd "$TAPS_PATH/tapsampling"

# Only run once to create an annotate file: $TAPS_PATH/tapsampling/complete_percentage.pkl
python tapsampling/annotate_complete_percentage.py --dataset_root ../calvin/dataset/task_ABC_D

# Train. Check the script for detail configurations (e.g. checkpoint path and output path).
bash train_calvin_sp.sh
```

## 🔍 Inference with Released Checkpoint

### Deploy the Action-VAE
```text
cd "$TAPS_PATH/action_vae"

# Deploy.
bash deploy_action_vae.sh
```

### Evaluate on CALVIN with **VPP + TapSampling Verifier**

If the Action-VAE port changed, maybe you need to update server configuration in the ```video-prediction-policy/policy_evaluation/sampling_wrapper.py```.

Importantly, check the model path in the ```video-prediction-policy/eval.sh``` before running.

```text
cd "$TAPS_PATH/video-prediction-policy"
bash eval.sh
```



## 🙏 Acknowledgements

We thank [DART](https://github.com/zkf1997/DART), [VPP](https://github.com/roboterax/video-prediction-policy), [CALVIN](https://github.com/mees/calvin), and [VLA-Adapter](https://github.com/OpenHelix-Team/VLA-Adapter) for their excellent open-source work.

## 📖 Citation

```bibtex
@inproceedings{zhao2026tapsampling,
  title={{T}ap{S}ampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation},
  author={Sizhe Zhao and Shengping Zhang and Shuo Yang and Weiyu Zhao and Shuigen Wang and Xiangyang Ji},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
