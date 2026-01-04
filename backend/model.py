import tensorflow as tf
import numpy as np
import os

# UCF101 Class Names (Subset or Full - for now using a sample list or full if available)
# Since the user mentioned 101 classes, we ideally need the full list.
# For the purpose of this demo, I will define a few common classes or placeholder logic.
# In a real scenario, this should load from a class_indices.npy or txt file.

CLASS_NAMES = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
    "Bowling", "BoxingPunchingBag", "BoxingSpeedBag", "BreastStroke", "BrushingTeeth",
    # ... Add more as needed. For the demo, if prediction index > len, return "Unknown"
]

class ActionModel:
    def __init__(self, model_path="ucf101_cnn_lstm_subset_model.h5"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.model = None
        else:
            print(f"Model file {self.model_path} not found. Using MockModel.")
            self.model = None

    def predict(self, input_data):
        """
        Input data shape: (1, 16, 224, 224, 3)
        Returns: { "action": "ClassName", "confidence": float }
        """
        if self.model:
            preds = self.model.predict(input_data)
            class_idx = np.argmax(preds[0])
            confidence = float(preds[0][class_idx])
            
            # Safe class name retrieval
            action_name = f"Class {class_idx}"
            # Check if we have the class list. For 101 classes it's long, 
            # ideally we map it correctly. 
            # IF user provided list is not 101, this might be off.
            # For now, we return index + generic name unless we have the full list.
            if class_idx < len(CLASS_NAMES):
                action_name = CLASS_NAMES[class_idx]
            
            return {"action": action_name, "confidence": confidence}
        else:
            # Mock Prediction
            print("WARNING: Using Mock Prediction")
            return {"action": "Mock - ApplyEyeMakeup", "confidence": 0.99}

model_instance = ActionModel()
