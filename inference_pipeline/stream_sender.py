import numpy as np
import os
from multiprocessing import shared_memory, Semaphore
from picamera2 import Picamera2
from libcamera import controls

SHM_NAME = "camera_frame"
SEM_NAME = "/camera_sem" # Named semaphore for cross-process sync
W, H = 640, 640
FRAME_SIZE = W * H * 3

# 1. Setup Shared Memory & Semaphore
try:
    shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=FRAME_SIZE)
except FileExistsError:
    shm = shared_memory.SharedMemory(name=SHM_NAME)

# Semaphore initialized to 1 (Unlocked)
sem = Semaphore(1) 
shared_frame = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm.buf)

# 2. Setup Camera
device = Picamera2()
config = device.create_video_configuration(main={"format": 'RGB888', "size": (W, H)}) # Fixed to RGB
device.configure(config)
device.start()

device.set_controls({"AfMode":controls.AfModeEnum.Continuous})

print(f"[*] Streamer Live (RGB 640p) via RAM and Semaphore")

try:
    while True:
        request = device.capture_request()
        frame = request.make_array("main") # BGR to RGB via hardware ISP
        
        # 3. Synchronized Write
        sem.acquire()
        shared_frame[:] = frame[:]
        sem.release()
        
        request.release()
finally:
    device.stop()
    shm.close()
    shm.unlink()
