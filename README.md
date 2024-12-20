# CVDL_Practical

Group Project of the Winter 2024 Computer Vision and Deep Learning Practical at Ommer Lab. Group Members: 

Zharfan Nugroho, Sairam Yadla, Wu Yuanluo and Nikolai Röhrich

# Setup

To install all dependencies we suggest using a virtual environment (venv), and running:

```
pip install -r requirements.txt
```

Models can be trained by running `main.py`, where hyperparameters can be passed via command-line using `hydra` (see https://hydra.cc/docs/intro/). Possible hyperparameters are specified in `configs/config.yaml`. Examples:

- Train a CNN for 10 Epochs:
```
python main.py model=cnn epochs=10
```
- Usa cuda and specify device on which the model should be trained:
```
python main.py model=cnn cuda=True device=cuda:0
```

# Repository Overview 

- Datasets, as well as corresponding util files are saved in `data`.
- Models, with CNN, ViT and hybrid architectures, as well as underlying Modules such as Transformer Blocks, are located in `models`.
- The base training class, as well as saved models are located in `training`. 
- Dependencies to be installed via pip are saved in `requirements.txt`