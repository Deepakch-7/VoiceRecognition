import numpy as np

class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision
        import mediapipe as mp
        
        # Use mediapipe's built-in tflite interpreter
        from mediapipe.python._framework_bindings import packet_getter
        import mediapipe.python.solutions.hands as _  # ensure mediapipe loaded

        # Direct tflite via mediapipe internal
        self._run_inference = self._build_interpreter(model_path, num_threads)

    def _build_interpreter(self, model_path, num_threads):
        try:
            # Try mediapipe's internal tflite
            from mediapipe.python._framework_bindings import image as mp_image
        except:
            pass
        
        # Use numpy-based direct tflite loading via flatbuffers (no tf needed)
        import struct, os
        
        # Store path for lazy loading
        self.model_path = model_path
        self.interpreter = None
        return None

    def _load_interpreter(self):
        if self.interpreter is not None:
            return
        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
        except ImportError:
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            except ImportError:
                # Fallback: use ai_edge_litert (new name for tflite in Python 3.12+)
                from ai_edge_litert.interpreter import Interpreter
                self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        self._load_interpreter()
        input_data = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        return np.argmax(np.squeeze(result))
