import cv2
import torch
from PIL import Image
#from models.utils.visualization import visualize_gradients
from models.utils.attention_map import GradCAMHandler




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

        self.gradcam_handler = None
        if cfg.model == "hybrid": #self.mode == "realtime_attention_map" and 
            self.gradcam_handler = GradCAMHandler(self.model, device=self.device)

    def detect_emotion(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if face_cascade.empty():
            raise ValueError("Error loading Haar cascade file.")

        cap = cv2.VideoCapture(0) #webcam

        if not cap.isOpened():
            print("Error: Unable to access the video source/ Webcam.")
            return
        
        threshold = 0.45 #tested various values this seems to be workig fine.
        neutral_label = "neutral"

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
                        prediction = self.model(face_tensor, apply_softmax = True)
                        probabilities = prediction.cpu().numpy().flatten()
                        max_prob = probabilities.max()
                        max_index = probabilities.argmax()

                        if max_prob >= threshold:
                            label = self.emotion_labels[max_index]
                        else:
                             label = neutral_label


                if self.mode == "realtime_detection":
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                
                elif self.mode == "realtime_attention_map" and self.cfg.model == "hybrid":

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Grad-CAM visualization
                    heatmap_colored = self.gradcam_handler.compute_gradcam(face_tensor, target_class=torch.argmax(prediction))
                    heatmap_colored = cv2.resize(heatmap_colored, (w, h))

                    # Overlay Grad-CAM on face
                    frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, heatmap_colored, 0.4, 0)

                    
                    #frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, heatmap_colored, 0.4, 0)

            
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
                                (255, 255, 255),  #Text displayu color
                                1
                            )      
                            cv2.putText(
                                frame,
                                f"{neutral_label}: {1 - max_prob:.2f}",  # Confidence for neutral
                                (start_x, start_y + len(self.emotion_labels) * 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 255),  # Text display color
                                1
                            )
            
                


            cv2.imshow('Real-Time Emotion Detection', frame)

            # Handle keypress for toggling modes
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Exit on pressing ESC key
                break
            elif key == ord('t') or key == ord('T'):  # Press T to toggling between modes
                self.mode = "realtime_attention_map" if self.mode == "realtime_detection" else "realtime_detection"
                print(f"Switched to mode: {self.mode}")

        cap.release()
        cv2.destroyAllWindows()
