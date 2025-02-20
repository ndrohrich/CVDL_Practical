import cv2
import torch
from PIL import Image
from models.utils.attention_map import GradCAMHandler
from tqdm import tqdm


class EmotionVideoProcessor:
    def __init__(self, model, transform, emotion_labels, cfg=None):   
        self.model = model
        self.transform = transform
        self.emotion_labels = emotion_labels
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()

        self.gradcam_handler = None
        if cfg.model == "hybrid":  # Implemented only for hybrid model
            self.gradcam_handler = GradCAMHandler(self.model, device=self.device)

    def process_video(self, input_video_path, output_video_path):
        # Load the video
        cap = cv2.VideoCapture(input_video_path)
    
        #input_video_path = ""
        cap = cv2.VideoCapture(input_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


        frame_count = 0
        pbar= tqdm(total= total_frames, desc="Processing video")

        if not cap.isOpened():
            print("Error: Unable to open video file.")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        threshold = 0.6  # Adjust for neutral detection
        neutral_label = "neutral"

        frame_count = 0
        process_interval = 1  # Process every nth frame for videos

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            face_rects = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

            if frame_count % process_interval == 0:
                for (x, y, w, h) in face_rects:
                    face = frame[y:y+h, x:x+w]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face).convert("L")
                    face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        prediction = self.model(face_tensor, apply_softmax=True)
                        probabilities = prediction.cpu().numpy().flatten()
                        max_prob = probabilities.max()
                        max_index = probabilities.argmax()

                        if max_prob >= threshold:
                            label = f"{self.emotion_labels[max_index]} ({max_prob:.2f})"
                        else:
                            label = f"{neutral_label} ({1 - max_prob:.2f})"

                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    
                    if self.cfg.mode == "process_video" and self.cfg.model == "hybrid":
                        heatmap_colored = self.gradcam_handler.compute_gradcam(face_tensor, target_class=torch.argmax(prediction))
                        heatmap_colored = cv2.resize(heatmap_colored, (w, h))

                        
                        frame[y:y+h, x:x+w] = cv2.addWeighted(frame[y:y+h, x:x+w], 0.6, heatmap_colored, 0.4, 0)

            # Write the processed frame to output video
            out.write(frame)

            frame_count += 1
            pbar.update(1)
            progress = (frame_count / total_frames) * 100
            cv2.putText(frame, f"Progress: {progress:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow("Processing Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video processing done and saved Output at: {output_video_path}")
