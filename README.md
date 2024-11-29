# FaceCure-Project

## FaceCure Github

https://github.com/ftramer/FaceCure

## Introduction

This repository contains the code and resources for reproducing the experiments from the research paper "DATA POISONING WON'T SAVE YOU FROM FACIAL RECOGNITION". The project is part of the AI and Cybersecurity course, from the Master of Cybersecurity and Cyberdefence.
(link to paper https://openreview.net/pdf?id=B5XahNLmna)

## Setup

The experiments were performed on the HPC of the University of Luxembourg.

### Model files

GitHub is limited to 100MB Files, that's why you need to download some files manually.
https://drive.google.com/drive/folders/1D-EMXDRcx57aOUGSJSHWm-RLtn1bMK7W

### Dataset of People

https://vintage.winklerbros.net/facescrub.html

H.-W. Ng, S. Winkler.
A data-driven approach to cleaning large face datasets.
Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014

Script to download the files:
https://github.com/faceteam/facescrub

### Conda Env

conda create -n facecure python 3.7
conda activate facecure

### Required packages

download the requirements.txt from the different projects
pip install .

pip install fawkes==0.3.2

install keras,tensorflow,pillow,scikit-learn

### Required Extractors

https://github.com/IrvingMeng/MagFace/
https://github.com/openai/CLIP

##### install the correct packages for your GPU (depending on CUDA version)

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

## Commands

### NN classifier

#### Fawkes

python ./FaceCure-main/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match cloaked.png --classifier NN --names-list Aaron_Eckhart

#### Lowkey

python ./FaceCure-main/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier NN --names-list Aaron_Eckhart

### Oblivious NN classifier

#### Fawkes attack & Fawkes v1.0 extractor

python ./FaceCure-main/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match cloaked.png --classifier NN --names-list Aaron_Eckhart --model fawkesv10

#### Fawkes attack & MagFace extractor

python ./FaceCure-main/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match cloaked.png --classifier NN --names-list Aaron_Eckhart --model magface --resume path/to/magface_model.pth

#### LowKey attack & MagFace extractor

python ./FaceCure-main/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier NN --names-list Aaron_Eckhart --model magface --resume path/to/magface_model.pth

#### LowKey attack & CLIP extractor

python ./FaceCure-main/eval_oblivious.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier NN --names-list Aaron_Eckhart --model clip

### Adaptive NN classifier

#### fawkes

python ./FaceCure-main/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_fawkes_attack/download/ --unprotected-file-match .jpg --protected-file-match cloaked.png --classifier NN --names-list Aaron_Eckhart --robust-weights cp-robust-10.ckpt

#### lowkey

python ./FaceCure-main/eval.py --facescrub-dir facescrub/download/ --attack-dir facescrub_lowkey_attack/download/ --unprotected-file-match small.png --protected-file-match attacked.png --classifier NN --names-list Aaron_Eckhart --robust-weights cp-robust-10.ckpt

### Baseline evaluation with end-to-end training

#### fawkes

python ./FaceCure-main/eval_e2e.py --gpu 0 --attack-dir facescrub_fawkes_attack/download/Aaron_Eckhart/face --facescrub-dir facescrub/download/ --unprotected-file-match .jpg --protected-file-match cloaked.png

#### lowkey

python ./FaceCure-main/eval_e2e.py --gpu 0 --attack-dir facescrub_lowkey_attack/download/Aaron_Eckhart/face --facescrub-dir facescrub/download/ --unprotected-file-match small.png --protected-file-match attacked.png

### Adaptive end-to-end

#### fawkes

python ./FaceCure-main/eval_e2e.py --gpu 0 --attack-dir facescrub_fawkes_attack/download/Aaron_Eckhart/face --facescrub-dir facescrub/download/ --unprotected-file-match .jpg --protected-file-match cloaked.png --robust --public-attack-dirs facescrub_fawkes_attack/download facescrub_lowkey_attack/download facescrub_fawkesv03_attack/download

#### lowkey

python ./FaceCure-main/eval_e2e.py --gpu 0 --attack-dir facescrub_lowkey_attack/download/Aaron_Eckhart/face --facescrub-dir facescrub/download/ --unprotected-file-match small.png --protected-file-match attacked.png --robust --public-attack-dirs facescrub_fawkes_attack/download facescrub_lowkey_attack/download

# Sources

https://vintage.winklerbros.net/facescrub.html
https://github.com/ftramer/FaceCure
https://openreview.net/forum?id=hJmtwocEqzc
https://sandlab.cs.uchicago.edu/fawkes/
https://github.com/Shawn-Shan/fawkes
https://github.com/Shawn-Shan/fawkes/releases/tag/v0.3
https://github.com/IrvingMeng/MagFace/
https://github.com/openai/CLIP
