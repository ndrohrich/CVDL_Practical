import os
import pandas as pd
import torch
from PIL import Image
from PIL import UnidentifiedImageError



class ImageCSVProcessor:
    def __init__(self, model, transform, emotion_labels): #emotion labels are passed in main.py
        self.model = model
        self.transform = transform
        self.emotion_labels = emotion_labels
        self.neutral_label = "neutral"
        self.threshold= 0.6
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_folder(self, image_folder, output_csv='emotion_predictions.csv'):
        predictions = []
        for img_file in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_file)
            if os.path.isfile(img_path):
                try:
                    img = Image.open(img_path).convert('L')
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        prediction = self.model(img_tensor, apply_softmax = True)
                        prob = prediction.cpu().numpy().flatten()
                        max_prob = prob.max()
                        max_index = prediction.argmax().item()

                        if max_prob >= self.threshold:
                            label = self.emotion_labels[max_index]
                        else:
                            label = self.neutral_label

                        predictions.append([img_file, *prob, max_prob, label])
                except UnidentifiedImageError:
                    print(f"Skipping non-image file: {img_file}")
                continue

        df = pd.DataFrame(predictions, columns=['File', *self.emotion_labels, 'Max Probability', 'Predicted Emotion (Threshold)'])
        df.to_csv(output_csv, index=False)
        print(f"Predictions saved to {output_csv}")
