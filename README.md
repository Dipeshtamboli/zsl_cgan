# zsl_cgan

Zero-shot video action recognition on UCF101, where a conditional GAN synthesizes visual features for **unseen** action classes from their class-level semantics.

## Overview

This repository implements a feature-generating approach to zero-shot action recognition on the UCF101 dataset. The pipeline has three stages: (1) a ResNet-152 + Bi-LSTM (with temporal attention) is trained on the **seen** classes and used to extract 2048-dimensional clip features; (2) a **conditional GAN** is trained to synthesize those 2048-d visual features from a 300-d class semantic embedding plus noise, so that plausible features can be generated for classes that were never seen during training; (3) a fusion classifier combining the synthesized visual features, the semantic embeddings, and a graph-convolutional (GCN) view of the class relationship graph is trained over the combined seen + unseen feature set.

The 101 UCF101 classes are partitioned into 51 seen and 50 unseen classes (the runnable scripts here work with the 51 seen classes plus 10 unseen classes). The training code is research/experiment code: it targets a CUDA GPU and several input/output paths are hard-coded, so expect to edit paths before running.

## Repository layout

The canonical three-stage pipeline lives in these folders:

| Path | Role |
| --- | --- |
| `seen51_training/` | Train the Bi-LSTM feature extractor on the seen classes and extract 2048-d features to `.npy` files. |
| `cgan_training/` | Build class semantic vectors, train the conditional GAN, and generate synthetic features for seen + unseen classes. |
| `zsl/` | Train the final GCN + semantic + visual fusion classifier over real + generated features. |

Additional, auxiliary experiment code is also present:

- `new_model/` — alternative and incremental-training scripts, plus feature visualization (t-SNE).
- `LWF/` — a Learning-without-Forgetting style incremental-learning experiment.
- Top-level scripts (`gcn_model.py`, `gcn_vis_train.py`, `lstm_models.py`, `old_16_frames.py`, `video_data_loader.py`, `split_ucf.py`, `splitting_the_data.py`) are earlier standalone copies of the models and data-splitting utilities; the maintained pipeline is the three folders above.

## Requirements

There is no `requirements.txt` or `environment.yml` in the repo. Based on the imports, you need Python 3 with an NVIDIA GPU (CUDA) and:

- `torch` and `torchvision` (ResNet-152 pretrained weights, LSTM, GAN training)
- `numpy`
- `scipy` (reads the `.mat` semantic splits)
- `scikit-learn` (confusion matrix, train/test split)
- `opencv-python` (`cv2`, video frame extraction)
- `tensorboardX` (logging)
- `tqdm`
- `matplotlib`
- `requests`

## Datasets

- **UCF101** — 101 human-action video classes. Official page: https://www.crcv.ucf.edu/data/UCF101.php
- **Class semantics** — 300-d class embeddings are read from `cgan_training/att_splits.mat` (included in the repo) together with the seen/unseen mapping in `cgan_training/all_seen_unseen_labs.json`.

The data loader (`seen51_training/old_16_frames.py`) expects videos already decoded into per-clip frame folders, organized as `output_dir/<split>/<class_name>/<clip>/*.jpg`, with splits named `train`, `test_seen`, and `test_unseen`. The `VideoDataset` class can perform this frame extraction (see its `preprocess`/`process_video` methods). Update `root_dir` (raw UCF101 location) and `output_dir` (frame output location) inside `Path.db_dir` before running.

## Usage

All stages assume a CUDA-capable GPU. Because intermediate artifacts are passed between stages as `.npy` files with fixed names/paths, check the load/save paths in each script and align them (notes below).

### 1. Build the class semantic vectors

From `cgan_training/`, convert the attribute `.mat` file into per-class semantic vectors:

```bash
cd cgan_training
python mat_split.py
```

This reads `att_splits.mat` and `all_seen_unseen_labs.json` and writes `seen_semantic_51.npy` (51x300), `unseen_semantic_50.npy` (50x300), and `label_to_sem_vec.json`.

### 2. Train the Bi-LSTM feature extractor (seen classes)

From `seen51_training/`, after setting the dataset paths in `old_16_frames.py`:

```bash
cd seen51_training
python lstm_train.py --gpu 0 --logfile_name bi-lstm_seen51_training
```

This trains a `ConvLSTM` (ResNet-152 encoder + bidirectional LSTM + attention) on the 51 seen classes and saves checkpoints under `run/<logfile_name>/`.

### 3. Extract 2048-d clip features

Still in `seen51_training/`, extract features from a trained checkpoint into a `.npy` file:

```bash
python test_lstm_51.py --gpu 0
```

Edit `load_Path` (the checkpoint to load), the dataset `split` (e.g. `test_unseen`, or the seen/train split), and the output `.npy` name inside the script. The GAN stage expects the seen-class features at `npy_files/lstm_feats_51_classes_2048d.npy` (each row is 2048 feature dims + 1 label).

### 4. Train the conditional GAN

From `cgan_training/`:

```bash
cd cgan_training
python cgan_train_51.py
```

The generator maps a 300-d semantic vector + 100-d noise (plus a class-label embedding) to a 2048-d visual feature; training uses an LSGAN (MSE) adversarial loss. It loads `seen_semantic_51.npy` and `../npy_files/lstm_feats_51_classes_2048d.npy`, and saves the model to `saved_models/classes_51_epoch-<epoch>.pth.tar`.

### 5. Generate synthetic features for seen + unseen classes

```bash
python gen_cgan_51.py
```

This loads a trained generator checkpoint (`saved_models/classes_51_epoch-399.pth.tar` by default) plus `seen_semantic_51.npy` / `unseen_semantic_50.npy`, and writes synthesized visual features (2048-d + label) to `classes_51_add_10_generated.npy`.

### 6. Train the zero-shot fusion classifier

From `zsl/`:

```bash
cd zsl
python gcn_vis_train.py --gpu 0 --logfile_name gcn_vis_10
```

This trains the fusion model: a visual FC branch, a semantic FC branch, and a GCN branch (whose adjacency is the cosine similarity between class semantics) are combined and passed to a classifier over the 51 seen + 10 unseen classes, optimized with cross-entropy plus a triplet loss. It loads the generated features from `../cgan_training/gn_feats/classes_51_add_10_generated.npy` and the semantic `.npy` files from `../cgan_training/`, so place the file generated in step 5 at that path.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).