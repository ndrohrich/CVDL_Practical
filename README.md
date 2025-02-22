# CVDL_Practical

Group Project of the Winter 2024 Computer Vision and Deep Learning Practical at Ommer Lab. 

**SWAN Group Members:**
Sairam Yadla, Wu Yuanluo, Zharfan Nugroho, and Nikolai Röhrich

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

# RUNNING THE COMPLETE PROCESS (FROM TRAINING TO DEMO)

Please follow these steps to complete the process:

Info:
- **Models (model)**: lenet, vgg, vit, resnet, torch_resnet, hybrid
- **Datasets (dataset_mode)**: all, ck-plus, affectnet, FER2013
- **Modes (mode)**: train, realtime_detection, realtime_attention_map, process_video, process_csv


More individual parameters can be found in **config.yaml** file.
No spaces within hydra parameters for example **model = vit** doesn't work, it must be **model=vit**.

**Step 1: (Dataset and Training)**
Train the model using the following hydra command and parametrs. (example)
Choose the preferred dataset and model to train. We can also set parameters like epochs, cuda, and individual parameters for each model like depth, number of workers etc.

Example for lenet:
```
python main.py model=lenet dataset_mode=ck_plus epochs=10
```

Example for Torch Resnet:

```
python main.py model=torch_resnet dataset_mode=ck_plus epochs=10
```

For changing the type of torch resnet for example resnet50. Refer to configs for more info and models.
```
python main.py model=torch_resnet torch_resnet.model_type=resent50 epochs 10
```

Once the model is trained the model.pth and logs will be saved in the training folder. We can then run the trained model for realtime detection etc.
We also have the tensorboard integration, which we can use to analyse the training and testing accuracy. Run the following command in terminal. It will then give link to view the graphs and performace.

```
tensorboard --logdir PATH_TO_LOG_FILE
```


**Step 2: Using the trained model for emotion detection in realtime, video and image processing.**

**Step 2.1**: Realtime detection (Emotion Detection and GradCAM View)

INFO: As we decided to go with the hybrid architecture for our project, the GradCAM was implemented only for the hybrid model so it works only when model=hybrid.

We can now use the trained model for realtime evaluation. To do so please use the below hydra command:

```
python main.py model=hybrid mode=realtime_detection
```
This command will load the recently trained model in the training folder. For example, here we chose hybrid model so it will load the recently trained model in hybrid section from training folder. We can specify our desired model then it will take from that corresponding section from trainig folder.

We can also pass our custom model path stored in some other directory to use the model by using the following command. 

```
python main.py custom_model_path=MODEL_PATH_FILE.pth mode=realtime_detection
```


For GradCAM view we choose mode as realtime_attention_map:

```
python main.py model=hybrid mode=realtime_attention_map

```
So in this way we can load the trained model and choose the desired mode. When we run the demo these following features exists:

- T - Switch between Normal Emotion Detection and GradCAM view
- C - Switch camera (if any multiple cameras exist)
- Y- Switch the displayed probabilites (either top left of screen or under the detected face)

And in the RealTimeEmotionDetector.py code we can also set the 'threshold' for neutral emotion and also 'processing_interval=n' which basically processes the every n-th frame.

**Step 2.2: Video Output with Emotions and GradCAM**

Like realtime detection, we can also give a certain video with facial expressions to get the output video file with emotions and GradCAM overlayed.
To do so make sure you have the original/input video path and then run the following command:

```
python main.py model=hybrid mode=process_video input_video_path=ORIGINAL_VIDEO_PATH.mp4 
```

This will process the given video and gives the desired output with emotions and GradCAM overlays for detected faces.


**Step 2.3: Image Classification.**

We can also give set of images to classify the emotions and outout the CSV file with detected values for different emotions.

```
python main.py model=hybrid mode=process_csv
```
This will take all the images in the RealTimeDemoandImageCSV\TestImagesFolder and classify them. But we can also give our own Image Folder path with the following command:

```
python main.py model=hybrid mode=process_csv image_folder=IMAGE_FOLDER_PATH
```

Example for Torch Resnet:

```
python main.py model=torch_resnet torch_resnet.model_type=resent50 mode=process_csv
```



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
