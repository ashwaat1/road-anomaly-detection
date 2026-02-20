import numpy as np
import time
import cv2
import os
from collections import deque
from multiprocessing import shared_memory, Semaphore
from ultralytics import YOLO

# --- CONFIG ---
SHM_NAME = "camera_frame"
W, H = 640, 640
SAVE_DIR = "detections_log"
os.makedirs(SAVE_DIR, exist_ok=True)

# LOGIC PARAMS
BUFFER_SIZE = 3      # Must be detected in 5/5 frames
COOLDOWN_SEC = 1.0   # Wait 3 seconds before saving the same "event" again
CONF_LEVEL = 0.2    # Sensitive enough for INT8

# 1. Attach to Shared RAM & Semaphore
shm = shared_memory.SharedMemory(name=SHM_NAME)
sem = Semaphore(1)
shared_img = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm.buf)

model = YOLO('int8_best3.onnx', task='detect')

# 2. Tracking Variables
detection_buffer = deque(maxlen=BUFFER_SIZE)
last_save_time = 0

print(f"[*] Smart Logger Active. Saving detections to: {SAVE_DIR}")

try:
    while True:
        t0 = time.perf_counter()
        
        # --- STEP 1: IPC READ ---
        sem.acquire()
        local_frame = shared_img.copy() 
        sem.release()
        
        # --- STEP 2: INFERENCE ---
        results = model.predict(local_frame, verbose=False, imgsz=640, conf=CONF_LEVEL)
        t1 = time.perf_counter()
        
        # --- STEP 3: TEMPORAL LOGIC ---
        # Check if current frame has ANY detections
        has_detection = len(results[0].boxes) > 0
        detection_buffer.append(has_detection)
        
        # Only valid if the entire buffer is True (Consensus)
        is_consensus_met = all(detection_buffer) and len(detection_buffer) == BUFFER_SIZE
        
        current_time = time.time()
        on_cooldown = (current_time - last_save_time) < COOLDOWN_SEC
        
        save_triggered = False
        if is_consensus_met and not on_cooldown:
            # SAVE SNAPSHOT
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"detection_{timestamp}.jpg")
            
            # Use the annotated frame for the save
            annotated_save = results[0].plot()
            cv2.imwrite(save_path, annotated_save)
            
            last_save_time = current_time
            save_triggered = True
            # Clear buffer after saving to prevent immediate re-triggering
            detection_buffer.clear() 

        # --- STEP 4: VISUALIZATION & PROFILING ---
        display_frame = results[0].plot()
        
        inf_time = (t1 - t0) * 1000
        fps = 1 / (time.perf_counter() - t0)
        
        # Status Bar
        status = f"FPS: {fps:.1f} | INF: {inf_time:.1f}ms"
        if on_cooldown: status += " | COOLDOWN"
        if save_triggered: status += " | [SAVED!]"
        
        # Draw Buffer Health (Visual Debugging)
        buffer_viz = "".join(["X" if b else "_" for b in detection_buffer])
        cv2.putText(display_frame, f"Buf: [{buffer_viz}]", (10, 60), 1, 1, (255, 255, 0), 1)
        cv2.putText(display_frame, status, (10, 30), 1, 1.2, (0, 255, 0), 2)
        
        cv2.imshow("Smart Dashcam", display_frame)
        if cv2.waitKey(1) == ord('q'):
            break
finally:
    shm.close()
