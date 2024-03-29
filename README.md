# PD-Flow: A Point Cloud Denoising Framework with Normalizing Flows

by Aihua Mao, Zihui Du, Yu-Hui Wen, Jun Xuan, and Yong-Jin Liu.

## Introduction

Official code of our ECCV 2022 paper. [[Paper & Supplement]](https://arxiv.org/abs/2203.05940)

This work introduces a framework for point cloud denoising by incorporating normalizing flows.

## Environment

First clone the code of this repo:

```bash
git clone --recursive https://github.com/unknownue/pdflow
```

Then other settings can be either configured manually or set up with docker.

### Manual configuration

The code is implemented with CUDA 11.5, Python 3.8, PyTorch 1.11.
Other require libraries:

- pytorch-lightning==1.5.3 (for training)
- [knn_cuda](https://github.com/unlimblue/KNN_CUDA)
- [point-cloud-utils](https://github.com/fwilliams/point-cloud-utils) (for evaluation)
- [torch-cluster](https://github.com/rusty1s/pytorch_cluster) (for denoising)
- [pytorch3d](https://github.com/facebookresearch/pytorch3d) (for denoising)
- [kaolin](https://github.com/NVIDIAGameWorks/kaolin) (for training)

### Docker configuration

If you are familiar with Docker, you can use provided [Dockerfile](docker/Dockerfile) to configure all setting automatically.

### Additional configuration

If you want to train the network, you also need to build the kernel of PytorchEMD like followings:

```bash
cd metric/PytorchEMD/
python setup.py install --user
cp build/lib.linux-x86_64-3.8/emd_cuda.cpython-38m-x86_64-linux-gnu.so .
```

## Datasets

All training and evaluation data can be downloaded from repo of [score-denoise](https://github.com/luost26/score-denoise) and [DMRDenoise](https://github.com/luost26/DMRDenoise/).
After downloading, place the extracted files into `data` directory as list in [here](data/.gitkeep).

We include a [pretrained model](pretrain/pdflow-score-LCC.pt) in this repo.

## Training & Denosing & Evaluation

Train the model as followings:

```bash
# train on PUSet, see train_deflow_score.py for tuning parameters
python models/deflow/train_deflow_score.py

# train on DMRSet, see train_deflow_dmr.py for tuning parameters
python models/deflow/train_deflow_dmr.py
```

Denoising a single point cloud as followings:

```bash
python models/deflow/denoise.py \
    --input=path/to/input.xyz \
    --output=path/to/output.xyz \
    --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
```

Denoising point clouds in directory as followings:

```bash
python models/deflow/denoise.py \
    --input=path/to/input_directory \
    --output=path/to/output_directory \
    --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
```

Evaluation:

```bash
# build executable to evaluate uniform, see build.sh for detail
bash eval/uniformity/build.sh
# evaluate all xyz files in a directory
python eval/eval2.py \
    --pred_dir=path/to/evaluation/directory \
    --off_dir=path/to/off_mesh/directory \
    --gt_dir=path/to/ground_truth/directory \
    --csv=path/to/evaluation/directory/result.csv
```

Reproduce Paper Results:

```bash
# PUSet dataset, 10K Points
python models/deflow/denoise.py --input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.01 --output=evaluation/PU_10000_n0.01_i1 --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
python models/deflow/denoise.py --input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.02 --output=evaluation/PU_10000_n0.02_i1 --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
python models/deflow/denoise.py --input=data/ScoreDenoise/examples/PUNet_10000_poisson_0.03 --output=evaluation/PU_10000_n0.03_i1 --patch_size=1024 --niters=2 --ckpt=pretrain/pdflow-score-LCC.pt
# PUSet dataset, 50K Points
python models/deflow/denoise.py --input=data/ScoreDenoise/examples/PUNet_50000_poisson_0.01 --output=evaluation/PU_50000_n0.01_i1 --patch_size=1024 --niters=1 --ckpt=pretrain/pdflow-score-LCC.pt
python models/deflow/denoise.py --input=data/ScoreDenoise/examples/PUNet_50000_poisson_0.02 --output=evaluation/PU_50000_n0.02_i1 --patch_size=1024 --niters=2 --first_iter_partition --ckpt=pretrain/pdflow-score-LCC.pt
python models/deflow/denoise.py --input=data/ScoreDenoise/examples/PUNet_50000_poisson_0.03 --output=evaluation/PU_50000_n0.03_i1 --patch_size=1024 --niters=2 --first_iter_partition --ckpt=pretrain/pdflow-score-LCC.pt
```

## Citation

If this work is useful for your research, please consider citing:

```bibtex
@inproceedings{unknownue2022pdflow,
  title={PD-Flow: A Point Cloud Denoising Framework with Normalizing Flows},
  author={Mao, Aihua and Du, Zihui and Wen, Yu-Hui and Xuan, Jun and Liu, Yong-Jin},
  booktitle={The European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
