import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import csv
from collections import deque

class YOLOInferenceLogger:
    def __init__(self, model_path, video_path, log_interval=30):
        print(f"[INFO] Loading TensorRT model from: {model_path}")
        self.model = YOLO(model_path, task='detect')
        
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.log_interval = log_interval
        self.frame_count = 0
        self.detections_log = []
        
        # FPS tracking (rolling average of last 30 frames)
        self.inference_times = deque(maxlen=30)
        self.pipeline_times = deque(maxlen=30)
        
        # NEW: Track total accumulated time for final report
        self.total_inference_time = 0.0
        
        # Get class names from model
        self.class_names = self.model.names
        
        # CSV logging setup (lightweight) - now with class names
        self.csv_file = 'detections_log.csv'
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'num_detections', 'class_names', 'confidences'])
        
        print(f"[INFO] Video loaded: {video_path}")
        print(f"[INFO] Class names loaded: {list(self.class_names.values())}")
        print(f"[INFO] Logging every {log_interval} frames to {self.csv_file}")

    def preprocess_frame(self, frame):
        """Downscale 4K to 640x640 for inference (major FPS boost)"""
        h, w = frame.shape[:2]
        if max(h, w) > 640:
            scale = 640 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return frame

    def get_class_names_list(self, class_ids):
        """Convert class IDs to human-readable names"""
        if not class_ids:
            return []
        return [self.class_names[int(cls_id)] for cls_id in class_ids]

    def run_inference(self):
        total_start = time.perf_counter()
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            self.frame_count += 1
            frame_start = time.perf_counter()
            
            # Preprocess (downscale)
            input_frame = self.preprocess_frame(frame)
            
            # Inference only (no visualize/save)
            inf_start = time.perf_counter()
            results = self.model.predict(input_frame, conf=0.3, verbose=False, device=0)
            inf_time = time.perf_counter() - inf_start
            
            self.inference_times.append(inf_time)
            self.total_inference_time += inf_time  # <--- Accumulate total time here
            
            # Extract detections (no drawing)
            num_dets = len(results[0].boxes) if results[0].boxes is not None else 0
            class_ids = results[0].boxes.cls.cpu().numpy().tolist() if num_dets > 0 else []
            confs = results[0].boxes.conf.cpu().numpy().tolist() if num_dets > 0 else []
            
            # Convert class IDs to names
            class_names_list = self.get_class_names_list(class_ids)
            
            pipeline_time = time.perf_counter() - frame_start
            self.pipeline_times.append(pipeline_time)
            
            # Sparse logging (avoid spam) - now shows class names
            if self.frame_count % self.log_interval == 0:
                # This FPS is "How fast the GPU is working"
                inf_fps = len(self.inference_times) / sum(self.inference_times) if self.inference_times else 0
                # This FPS is "How fast the CPU+GPU processes a frame ONCE DECODED"
                pipe_fps = len(self.pipeline_times) / sum(self.pipeline_times) if self.pipeline_times else 0
                
                class_str = ', '.join(class_names_list) if class_names_list else 'None'
                print(f"Frame {self.frame_count}: Inference FPS={inf_fps:.1f}, Pipeline FPS={pipe_fps:.1f}, Dets={num_dets} ({class_str})")
                
                # Log to CSV (class names instead of IDs)
                with open(self.csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.frame_count, num_dets, class_names_list, confs])
            
            # Optional: Save every 100th frame's detections to memory
            if self.frame_count % 100 == 0:
                self.detections_log.append({
                    'frame': self.frame_count,
                    'num_dets': num_dets,
                    'class_names': class_names_list,
                    'confs': confs
                })
        
        total_time = time.perf_counter() - total_start
        
        # CORRECTED CALCULATIONS
        # Use the accumulated total time, NOT the deque sum
        avg_inf_fps = self.frame_count / self.total_inference_time if self.total_inference_time > 0 else 0
        
        real_world_fps = self.frame_count / total_time
        
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Total frames: {self.frame_count}")
        print(f"Total runtime: {total_time:.1f}s")
        print("-" * 30)
        print(f"Avg Inference FPS: {avg_inf_fps:.1f} (GPU Only)")
        print(f"Real World FPS:    {real_world_fps:.1f} (Includes 4K Decoding + Resize + Inference)")
        print("-" * 30)
        print(f"Detections logged to: {self.csv_file}")
        print("Recent detections sample:")
        for entry in self.detections_log[-3:]:
            class_str = ', '.join(entry['class_names']) if entry['class_names'] else 'None'
            print(f"  Frame {entry['frame']}: {entry['num_dets']} dets ({class_str})")

    def __del__(self):
        self.cap.release()

def main():
    MODEL_PATH = "/workspace/models/runs/yolo12m_165784/train/weights/best.engine"
    VIDEO_PATH = "/workspace/data/DJI_0017.MP4"
    
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not Path(VIDEO_PATH).exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    
    logger = YOLOInferenceLogger(MODEL_PATH, VIDEO_PATH)
    logger.run_inference()

if __name__ == "__main__":
    main()
