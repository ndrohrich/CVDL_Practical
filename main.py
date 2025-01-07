import hydra
from training import train
from torchvision import transforms
from RealTimeDemoandImageCSV.ImageCSVProcessor import ImageCSVProcessor
from RealTimeDemoandImageCSV.RealTimeEmotionDetector import RealTimeEmotionDetector
import os
import torch


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    print(f"Running in {cfg.mode} mode...")
    if cfg.mode == "train":
        # Training mode
        trainer = train.Trainer(cfg)
        print("NUMBER OF PARAMETERS: ", trainer.num_parameters)
        trainer.train()

    elif cfg.mode == "process_csv":
        # Image processing mode
        process_images(cfg)

    elif cfg.mode == "realtime_detection":
        realtime_emotion_detection(cfg)
        
    else:
        print(f"Unknown mode: {cfg.mode}")


def get_model_path(cfg):
    """
    Dynamically determine the path to the trained model based on cfg.model.
    """
    
    base_folder = f"./training/trained_models/{cfg.model}"
    
    if not os.path.exists(base_folder):
        raise FileNotFoundError(f"No trained models found for {cfg.model} at {base_folder}")
    
    # Get all subdirectories within the model folder
    subfolders = [os.path.join(base_folder, d) for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    
    if not subfolders:
        raise FileNotFoundError(f"No trained model subfolders found for {cfg.model} in {base_folder}")
    
    # Sort subfolders by modification time (most recent first)
    subfolders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # find model.path
    for folder in subfolders:
        model_file = os.path.join(folder, "model", "model.pth")
        if os.path.exists(model_file):
            return model_file
    
    # If no model is found
    raise FileNotFoundError(f"No model.pth file found for {cfg.model} in {base_folder}")

def process_images(cfg):
    # Define the transform 
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load the trained model dynamically based on the selected model
    model_path = get_model_path(cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first!")
    model = torch.load(model_path, map_location=cfg.device)
    model.eval()

    # Emotion labels for predictions
    emotion_labels = ['happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # Initialize ImageCSVProcessor
    processor = ImageCSVProcessor(model, transform, emotion_labels)

    # Process the folder and save predictions to CSV
    processor.process_folder(cfg.image_folder, output_csv=cfg.output_csv)
    

def realtime_emotion_detection(cfg):
    
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    # Load the trained model dynamically based on the selected model
    model_path = get_model_path(cfg)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Train the model first!")
    model = torch.load(model_path, map_location=cfg.device)
    model.eval()

    # Emotion labels for predictions
    emotion_labels = ['happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # Initialize RealTimeEmotionDetector
    detector = RealTimeEmotionDetector(model, transform, emotion_labels)

    # Start the real-time detection
    detector.detect_emotion()

if __name__ == '__main__': 
    main()
