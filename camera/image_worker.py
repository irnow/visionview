import numpy as np
import importlib
from PySide6.QtCore import QObject, Signal, Slot, QThread
from utils.split_view import compose_split_view

_CV_MODULE_CACHE = {}

def get_cv_module(script_name):
    mod = _CV_MODULE_CACHE.get(script_name)
    if mod is None:
        mod = importlib.import_module(f"computer_vision.{script_name}")
        _CV_MODULE_CACHE[script_name] = mod
    return mod

class ImageWorker(QObject):
    image_ready = Signal(np.ndarray, int, int)
    finished = Signal()

    def __init__(self, image_array):
        super().__init__()
        self.image = image_array
        self.pipeline = []
        self.pipeline_enabled = True
        self.split_view = False
        self.split_view_mode = "side_by_side"
        self._running = True

    @Slot()
    def process(self):
        try:
            orig_h, orig_w = self.image.shape[:2]
            while self._running:
                raw_img = self.image.copy()
                img = raw_img.copy()
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

                if getattr(self, 'split_view', False):
                    mode = getattr(self, 'split_view_mode', 'side_by_side')
                    is_enabled = getattr(self, 'pipeline_enabled', True)
                    img = compose_split_view(raw_img, img, mode=mode, pipeline_enabled=is_enabled)
                
                curr_h, curr_w = img.shape[:2]
                self.image_ready.emit(img, curr_w, curr_h)
                
                # Control the frame rate (approx 33ms = 30fps)
                QThread.msleep(33)
        except Exception as e:
            print(f"ImageWorker error: {e}")
        finally:
            self.finished.emit()

    def stop(self):
        self._running = False