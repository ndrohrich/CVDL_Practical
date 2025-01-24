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

# RUNNING THE COMPLETE PROCESS

Please follow these steps to complete the process:

Info:
For model 'lenet', 'vit'
For mode 'train' 'process_csv'. By Default it is set to train.
For dataset_mode 'ck-plus' 'affectnet' 'FER2013'
No spaces within parameters for example model = vit doesn't work, it must be model=vit.

Step 1: (Dataset and Training)
Train the model using the following hydra command and parametrs. (example)

```
python main.py model=lenet dataset_mode=ck_plus epochs=10
```

For Torch Resnet:

```
python main.py model=torch_resnet dataset_mode=ck_plus epochs=10
```

For changing the type of torch resnet for example resnet50. Refer to configs for more info and models.
```
python main.py model=torch_resnet torch_resnet.model_type=resent50 epochs 10
```



Step2: Test Images Predictions.

Please make sure that the model is trained and your validation test images are uploaded/daved in folder "RealTimeDemoandImageCSV\TestImagesFolder" for evaluation. The test image path and CSV File saving is automatically set in code. So once the below is executed you will find the emotion_prediciton.csv in your directory

After this, run the following hydra command (example)

```
python main.py model=vit mode=process_csv 
```
For Torch Resnet, Example Resnet50:
```
python main.py model=torch_resnet torch_resnet.model_type=resent50 mode=process_csv
```

If you want to change the validation image folder or output folder. Feel free to do so with below command format
```
python main.py mode=process_csv model=vit image_folder=./RealTimeDemoandImageCSV/TestImagesFolder output_csv=predictions.csv

```
Step 3: (Realtime Face Demo)

Same as Image Processing, but mode=realtime_detection (for normal emotion detection), mode=realtime_gradient (for activation maps)

```
python main.py model=vit mode=realtime_detection

python main.py model=vit mode=realtime_gradient
```

Step 4:(Feature maps and Highlights)

Still have to be figured out.

# DATASET
we using serveral dataset for this project, the dataset is:
- CK+ (Cohn-Kanade Extended) dataset
    - Input size: 48x48
    - Number of classes: 6
    - Number of samples: 327   
- FER2013 dataset
    - Input size: 48x48
    - Number of classes: 7
    - Number of samples: 35887

# MODELS
We have implemented several models for this project, the models are:
- CNN
- ViT
- Hybrid (CNN + ViT)
- FCN (Feature Clustering Network)

# Repository Overview 

- Datasets, as well as corresponding util files are saved in `data`.
- Models, with CNN, ViT and hybrid architectures, as well as underlying Modules such as Transformer Blocks, are located in `models`.
- The base training class, as well as saved models are located in `training`. 
- Dependencies to be installed via pip are saved in `requirements.txt`
