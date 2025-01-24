import cv2
import torch
from PIL import Image
from models.utils.visualization import visualize_gradients




class RealTimeEmotionDetector:
    def __init__(self, model, transform, emotion_labels,mode="realtime_detection", cfg=None):   
        self.model = model
        self.transform = transform
        self.emotion_labels = emotion_labels
        self.mode = mode
        self.cfg = cfg
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
            face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            probabilities = None #reset for each frame

            for (x, y, w, h) in face_rects:
                face = frame[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face).convert("L")
                face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)


                with torch.no_grad():
                        prediction = self.model(face_tensor)
                        probabilities = prediction.cpu().numpy().flatten()
                        label = self.emotion_labels[prediction.argmax().item()]


                if self.mode == "realtime_detection":
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                
                elif self.mode == "realtime_gradient":

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Gradient visualization
                    gradient_map = visualize_gradients(self.model, face_tensor, torch.tensor([0]), self.cfg)
                    grad_overlay = gradient_map[0].transpose(1, 2, 0)  
                    grad_overlay = cv2.cvtColor(grad_overlay, cv2.COLOR_RGB2BGR)

                    grad_overlay = cv2.convertScaleAbs(grad_overlay, alpha=1.5, beta=100) #alpha=contrast beta=brightness

                    grad_overlay = cv2.resize(grad_overlay, (w, h))

                    
                    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, grad_overlay, 0.4, 0)

            
            if probabilities is not None:
                        start_x, start_y = 10, 30  
                        for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
                            text = f"{emotion}: {prob:.2f}"
                            cv2.putText(
                                frame,
                                text,
                                (start_x, start_y + i * 20), 
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 0, 255),  #Text displayu color
                                1
                            )      
            
            cv2.imshow('Real-Time Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:  #Exit on pressing exc key
                break

        cap.release()
        cv2.destroyAllWindows()

