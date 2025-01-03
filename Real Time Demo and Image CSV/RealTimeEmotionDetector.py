import cv2
import torch
from PIL import Image

#TODO: still working on feature maps integration and also needs this to be integrated and instatiated with other models



emotion_labels_def = ['happy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

class RealTimeEmotionDetector:
    def __init__(self, model, transform, emotion_labels):   #we need to pass emotion labesl here when instantiatind demo?
        self.model = model
        self.transform = transform
        self.emotion_labels = emotion_labels
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect_emotion(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if face_cascade.empty():
            raise ValueError("Error loading Haar cascade file.")

        cap = cv2.VideoCapture(0) #webcam

        if not cap.isOpened():
            print("Error: Unable to access the video source/ Webcam.")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in face_rects:
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face)
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    prediction = self.model(face_tensor)
                    label = self.emotion_labels[prediction.argmax().item()]

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow('Real-Time Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:  #Exit on pressing exc key
                break

        cap.release()
        cv2.destroyAllWindows()

