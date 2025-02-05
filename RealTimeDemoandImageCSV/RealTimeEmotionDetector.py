import cv2
import torch
from PIL import Image
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
        if cfg.model == "hybrid":  #as our final model is Hybrid we implemented it only for hybrid
            self.gradcam_handler = GradCAMHandler(self.model, device=self.device)

    def detect_emotion(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        if face_cascade.empty():
            raise ValueError("Error loading Haar cascade file.")

        camera_index = 0
        max_cameras = 2 #here we might use 2 cams webcam and external cam.
        cap = cv2.VideoCapture(camera_index) 

        if not cap.isOpened():
            print("Error: Unable to access the video source/ Webcam.")
            return
        
        threshold = 0.6 #tested various values this seems to be workig fine.
        neutral_label = "neutral"

        frame_count = 0
        process_interval = 15 #process every n frames, n is values we give here.
        start_time = cv2.getTickCount()
        last_processed_frame = None

        display_mode = "under_head"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            if frame_count % process_interval == 0:
                probabilities = None #reset for each frame

                start_x, start_y = 10, 30

                for face_idx, (x, y, w, h) in enumerate(face_rects):
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
                                label = f"{self.emotion_labels[max_index]} ({max_prob:.2f})"
                            else:
                                label = f"{neutral_label} ({1 - max_prob:.2f})"

                    #We can start any mode, as they are switchable using T during Runtime
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


                
                if probabilities is not None:
                            current_y = start_y + face_idx * (20 * (len(self.emotion_labels) + 2))
                            if display_mode == "under_head":
                                for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
                                    text = f"{emotion}: {prob:.2f}"
                                    cv2.putText(
                                        frame,
                                        text,
                                        (x, y + h + 20 + i * 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),  #Text displayu color
                                        1
                                    )      
                                    cv2.putText(
                                        frame,
                                        f"{neutral_label}: {1 - max_prob:.2f}",  # Confidence for neutral
                                        (x, y + h + 20 + len(self.emotion_labels) * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),  # Text display color
                                        1
                                    )
                            elif display_mode == "top_left":
                                 for i, (emotion, prob) in enumerate(zip(self.emotion_labels, probabilities)):
                                    text = f"{emotion}: {prob:.2f}"
                                    cv2.putText(
                                        frame,
                                        text,
                                        (start_x, current_y + i * 20), 
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),  #Text displayu color
                                        1
                                    )      
                                    cv2.putText(
                                        frame,
                                        f"{neutral_label}: {1 - max_prob:.2f}",  # Confidence for neutral
                                        (start_x, current_y + len(self.emotion_labels) * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5,
                                        (0, 255, 0),  # Text display color
                                        1
                                    )


                last_processed_frame = frame.copy()

            display_frame = last_processed_frame if last_processed_frame is not None else frame

            
            frame_count += 1
            
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
            start_time = cv2.getTickCount()  # Reset start time
            
            height, width, _ = display_frame.shape
            bottom_left_x = 10
            bottom_left_y = height - 20

            cv2.putText(display_frame, f"Processing every {process_interval} frames.",
                        (bottom_left_x, bottom_left_y -30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0), 1)
            cv2.putText(display_frame, "T = Normal/GradCAM View, Y = change Probabilities position, C = Switch Cameras (if any)",
                        (bottom_left_x, bottom_left_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0), 1)
            
            #cv2.namedWindow('Real-Time Emotion Detection', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow('Real-Time Emotion Detection', 800, 800)
            cv2.imshow('Real-Time Emotion Detection', display_frame)


            # Handle keypress for toggling modes
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Exit on pressing ESC key
                break
            elif key == ord('t') or key == ord('T'):  # Press T to toggling between modes
                self.mode = "realtime_attention_map" if self.mode == "realtime_detection" else "realtime_detection"
                print(f"Switched to mode: {self.mode}")
            elif key == ord('y') or key == ord('Y'):  # Press Y to toggle probabiliies display mode
                display_mode = "top_left" if display_mode == "under_head" else "under_head"
                print(f"Switched to probabilities display mode: {display_mode}")
            elif key == ord('c') or key == ord('C'): #Press C to switch cameras
                 cap.release()
                 camera_index = (camera_index + 1) % max_cameras
                 cap = cv2.VideoCapture(camera_index)
                 if not cap.isOpened():
                      print(f"Error accesing the camera {camera_index}. Switching to Default camera 0")
                      camera_index = 0
                      cap = cv2.VideoCapture(camera_index)

        cap.release()
        cv2.destroyAllWindows()
