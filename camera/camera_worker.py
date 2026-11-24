from PySide6.QtCore import QObject, Signal, Slot
import numpy as np
import importlib
import time
import cv2 
from collections import deque
from utils.split_view import compose_split_view

_CV_MODULE_CACHE = {}

def get_cv_module(script_name):
    mod = _CV_MODULE_CACHE.get(script_name)
    if mod is None:
        mod = importlib.import_module(f"computer_vision.{script_name}")
        _CV_MODULE_CACHE[script_name] = mod
    return mod

class CameraWorker(QObject):
    image_ready = Signal(np.ndarray, int, int)
    finished = Signal()
    error = Signal(str)
    fps_ready = Signal(float)

    def __init__(self, camera_object, camera_instance):
        super().__init__()
        '''Get respective camera class and device instance'''
        self.camera_object = camera_object # Class
        self.camera_instance = camera_instance # Device Info
        self.pipeline = []
        self.pipeline_enabled = True
        self.split_view = False
        self.split_view_mode = "side_by_side"
        self._running = True
        self.frame_times = deque(maxlen=30)
        self._last_fps_emit_time = 0.0

    @Slot()
    def process(self):
        try:
            self.camera_object.setup(self.camera_instance) # Prepare the camera

            last_time = time.perf_counter()
            self._last_fps_emit_time = last_time
            while self._running:
                img = self.camera_object.image()
                if img is not None: # Only emit if actually got an image
                    raw_img = img.copy() if getattr(self, 'split_view', False) else None
                    if getattr(self, 'pipeline_enabled', True) and self.pipeline:
                        for step in list(self.pipeline):
                            module_name = step.get('name', '')
                            params = step.get('params', {})
                            try:
                                script_name = module_name.lower().replace(" ", "_")
                                m = get_cv_module(script_name)
                                img = m.process(img, params)
                            except Exception as e:
                                print(f"Pipeline Error ({module_name}): {e}")
                    else:
                        # Decouple from camera converter internal buffer if no filter produced a new array
                        img = raw_img if raw_img is not None else img.copy()

                    if getattr(self, 'split_view', False) and raw_img is not None:
                        mode = getattr(self, 'split_view_mode', 'side_by_side')
                        is_enabled = getattr(self, 'pipeline_enabled', True)
                        img = compose_split_view(raw_img, img, mode=mode, pipeline_enabled=is_enabled)

                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    last_time = current_time
                    
                    self.frame_times.append(dt)
                    
                    # Throttle FPS emission to ~4 Hz (every 250ms) to reduce cross-thread GUI events
                    if current_time - self._last_fps_emit_time >= 0.25 and len(self.frame_times) > 0:
                        self._last_fps_emit_time = current_time
                        avg_dt = sum(self.frame_times) / len(self.frame_times)
                        if avg_dt > 0:
                            fps = 1.0 / avg_dt
                            self.fps_ready.emit(round(fps, 1))

                    orig_h, orig_w = img.shape[:2]

                    # Emit image along with original full dimensions (no redundant copy)
                    self.image_ready.emit(img, orig_w, orig_h)
                else:
                    # Prevent tight loop when camera is waiting or idle
                    time.sleep(0.005)

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False