# generative-ai-project

This project trains a Variational Autoencoder (VAE) in PyTorch to generate synthetic traffic sign
images, with data augmentation for a sign classifier as the motivating use case.

**Dataset:** [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign)
(Ertler et al., 2020), full street-level photographs with bounding-box annotations for every sign in
frame, a different dataset and format from GTSRB (used in Project 4). The data is loaded from the
[`ThankGod/mapillary_traffic_sign_dataset`](https://huggingface.co/datasets/ThankGod/mapillary_traffic_sign_dataset)
Hugging Face mirror. The notebook crops individual sign patches out of two shards' worth of
photographs (1,154 images) using the provided bounding boxes, producing 1,721 32x32 sign crops.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

```bash
from huggingface_hub import hf_hub_download

REPO_ID = "ThankGod/mapillary_traffic_sign_dataset"
SHARD_FILES = [
    "train_mtsd/train-00000-of-00045.parquet",
    "train_mtsd/train-00001-of-00045.parquet",
]

hf_hub_download(repo_id=REPO_ID, filename="id2labels.txt", repo_type="dataset")
for shard_file in SHARD_FILES:
    path = hf_hub_download(repo_id=REPO_ID, filename=shard_file, repo_type="dataset")
    print(f"cached {shard_file} -> {path}")

```

Downloads and caches the two MTSD parquet shards used by the notebook. The notebook itself performs
the bounding-box cropping, so the cropping pipeline is visible and reproducible from the notebook.

## Running the notebook

```bash
source .venv/bin/activate
jupyter notebook generative_model.ipynb
```

Run all cells top to bottom to reproduce the dataset loading/cropping, VAE training, and generated
sample evaluation.

## Results

150 epochs, 1,721 training crops (90/10 train/validation split). Reconstruction loss and KL
divergence both converge smoothly with no widening train/validation gap. The model reconstructs the
overall shape and color of real signs but loses fine detail. Samples generated from the prior
reliably reproduce plausible sign shapes and color schemes (strongest: samples 15, 19, 20, sharp
red-and-white circular rims) but never render a legible symbol or number, and some samples are
outright failures with no visible sign structure (samples 18, 22).

## Report

`module_summary.pdf` is the written Generative AI Applications Analysis Report (built from
`module_summary.html` via WeasyPrint), covering dataset sourcing and preprocessing, model design and
training, output evaluation with concrete generated examples, ethical considerations, limitations,
and references.
