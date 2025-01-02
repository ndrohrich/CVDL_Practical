import os
import pandas as pd
import torch
from PIL import Image

#TODO: Needs to be integrated with other models and also make sure it takes random images and transforms to 64x64

emotion_labels_def = ['happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

class ImageCSVProcessor:
    def __init__(self, model, transform, emotion_labels): #we need to pass emotion labesl here when instantiatind demo?
        self.model = model
        self.transform = transform
        self.emotion_labels = emotion_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_folder(self, image_folder, output_csv='emotion_predictions.csv'):
        predictions = []
        for img_file in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_file)
            if os.path.isfile(img_path):
                img = Image.open(img_path).convert('RGB')
                img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    prediction = self.model(img_tensor)
                    prob = prediction.cpu().numpy().flatten()
                    label = self.emotion_labels[prediction.argmax().item()]
                    predictions.append([img_file, *prob, label])

        df = pd.DataFrame(predictions, columns=['File', *self.emotion_labels, 'Predicted'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
