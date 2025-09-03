import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def make_prediction(model_name, features):
    """
    Make a prediction using the selected model.
    In a real implementation, this would use actual trained models.
    
    Args:
        model_name: Name of the model to use
        features: Feature matrix for inference
        
    Returns:
        Prediction, confidence score, and anomaly flag
    """
    
    # Mean feature for each class (mock data for demonstration)
    class_means = {
        "NB": np.random.normal(0, 1, size=features.shape[1]),
        "IR7": np.random.normal(2, 1, size=features.shape[1]),
        "IR21": np.random.normal(4, 1, size=features.shape[1]),
        "OR7": np.random.normal(-2, 1, size=features.shape[1]),
        "OR21": np.random.normal(-4, 1, size=features.shape[1])
    }
    
    # Standardize features (assume scaler is fitted in a real scenario)
    scaler = StandardScaler()
    
    # Prediction based on model type
    if model_name in ["Artificial Neural Network", "Random Forest"]:
        # Classification models
        classes = ["NB", "IR7", "IR21", "OR7", "OR21"]
        
        # Calculate distance to each class mean and use as a proxy for class probability
        
        distances = {}
        for cls in classes:
            distances[cls] = np.mean(np.abs(features - class_means[cls]))
        
        # Normalize distances to get a pseudo-probability (lower distance = higher probability)
        
        total_distance = sum(1.0 / (d + 1e-10) for d in distances.values())
        probs = {cls: 1.0 / (d + 1e-10) / total_distance for cls, d in distances.items()}
        
        # Get most likely class
        prediction = max(probs, key=probs.get)
        confidence = probs[prediction]

        
        # For classification models, anomaly is False unless it's a fault
        is_anomaly = prediction != "NB"
        
        return prediction, confidence, is_anomaly
    
    elif model_name in ["Isolation Forest", "One-Class SVM", "Autoencoder", "LSTM Autoencoder", "Elliptic Envelope"]:
        # Anomaly detection models
        
        # Calculate mock anomaly score based on distance to normal class mean
        distance_to_normal = np.mean(np.abs(features - class_means["NB"]))
        
        # Normalize to 0-1 range (sigmoid)
        anomaly_score = 1 / (1 + np.exp(-2 * (distance_to_normal - 2)))
        
        # Determine if it's an anomaly based on threshold
        threshold = 0.5
        is_anomaly = anomaly_score > threshold
        
        # For anomaly detection, there's no specific class prediction
        prediction = "Anomaly" if is_anomaly else "Normal"
        
        return prediction, anomaly_score, is_anomaly
    
    elif model_name == "K-Means Clustering":
        # Clustering model
        
        # Calculate distance to each class mean
        distances = {cls: np.mean(np.abs(features - mean)) for cls, mean in class_means.items()}
        
        # Assign to closest cluster
        prediction = min(distances, key=distances.get)
        
        # Normalize distances for confidence
        total_distance = sum(1.0 / (d + 1e-10) for d in distances.values())
        confidence = (1.0 / (distances[prediction] + 1e-10)) / total_distance
        
        # For clustering, anomaly is determined by distance to assigned cluster
        is_anomaly = distances[prediction] > 3.0  # Arbitrary threshold
        
        return prediction, confidence, is_anomaly
    
    else:
        # Default case
        return "Unknown", 0.0, False



