import numpy as np

class PointHistoryClassifier:
    def __init__(self, model_path='model/point_history_classifier/point_history_classifier.tflite', score_th=0.5, invalid_value=0, num_threads=1):
        self.model_path = model_path
        self.score_th = score_th
        self.invalid_value = invalid_value
        self.interpreter = None

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
                from ai_edge_litert.interpreter import Interpreter
                self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, point_history_list):
        self._load_interpreter()
        input_data = np.array([point_history_list], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        result = self.interpreter.get_tensor(self.output_details[0]['index'])
        result = np.squeeze(result)
        if np.max(result) < self.score_th:
            return self.invalid_value
        return np.argmax(result)
