import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Preprocess the raw vibration data.
    
    Args:
        data: DataFrame with DE and FE columns
        
    Returns:
        Preprocessed data
    """
    # Copy data to avoid modifying the original
    processed_data = data.copy()
    
    # Handle missing values if any
    processed_data = processed_data.fillna(method='ffill')
    
    # Remove outliers (optional)
    # Assuming DE and FE are the columns to preprocess
    for col in ['DE', 'FE']:
        if col in processed_data.columns:
            mean = processed_data[col].mean()
            std = processed_data[col].std()
            
            # Replace outliers beyond 3 standard deviations with the mean
            processed_data.loc[np.abs(processed_data[col] - mean) > 3 * std, col] = mean
    
    # Normalize data
    for col in ['DE', 'FE']:
        if col in processed_data.columns:
            processed_data[col] = (processed_data[col] - processed_data[col].mean()) / processed_data[col].std()
    
    return processed_data

def extract_features(data, window_size=200, overlap=0.5):
    """
    Extract features from time series data.
    
    Args:
        data: DataFrame with DE and FE columns
        window_size: Size of the window for feature extraction
        overlap: Overlap between consecutive windows
        
    Returns:
        Feature matrix
    """
    # Initialize list to store features
    features_list = []
    
    # Convert to numpy arrays for faster processing
    de_signal = data['DE'].values if 'DE' in data.columns else np.zeros(len(data))
    fe_signal = data['FE'].values if 'FE' in data.columns else np.zeros(len(data))
    
    # Calculate step size based on window size and overlap
    step_size = int(window_size * (1 - overlap))
    
    # Extract features from windows
    for i in range(0, len(data) - window_size, step_size):
        window_de = de_signal[i:i+window_size]
        window_fe = fe_signal[i:i+window_size]
        
        # Time domain features
        features = []
        
        # Basic statistics - DE
        features.append(np.mean(window_de))  # Mean
        features.append(np.std(window_de))   # Standard deviation
        features.append(stats.skew(window_de))  # Skewness
        features.append(stats.kurtosis(window_de))  # Kurtosis
        features.append(np.max(window_de))  # Peak value
        features.append(np.min(window_de))  # Minimum value
        features.append(np.max(window_de) - np.min(window_de))  # Range
        features.append(np.sqrt(np.mean(np.square(window_de))))  # RMS
        
        # Basic statistics - FE
        features.append(np.mean(window_fe))
        features.append(np.std(window_fe))
        features.append(stats.skew(window_fe))
        features.append(stats.kurtosis(window_fe))
        features.append(np.max(window_fe))
        features.append(np.min(window_fe))
        features.append(np.max(window_fe) - np.min(window_fe))
        features.append(np.sqrt(np.mean(np.square(window_fe))))
        
        # Frequency domain features
        # Compute FFT of the signals
        fft_de = np.abs(np.fft.fft(window_de))
        fft_fe = np.abs(np.fft.fft(window_fe))
        
        # Keep only the first half of the FFT (symmetry)
        fft_de = fft_de[:len(fft_de)//2]
        fft_fe = fft_fe[:len(fft_fe)//2]
        
        # Compute frequency features
        features.append(np.max(fft_de))  # Maximum amplitude
        features.append(np.mean(fft_de))  # Mean amplitude
        features.append(np.std(fft_de))   # Standard deviation of amplitude
        
        features.append(np.max(fft_fe))
        features.append(np.mean(fft_fe))
        features.append(np.std(fft_fe))
        
        # Statistical features on frequency domain
        features.append(stats.skew(fft_de))
        features.append(stats.kurtosis(fft_de))
        features.append(stats.skew(fft_fe))
        features.append(stats.kurtosis(fft_fe))
        
        # Cross-correlation between DE and FE
        corr = np.correlate(window_de, window_fe, mode='valid')[0] / (np.std(window_de) * np.std(window_fe) * len(window_de))
        features.append(corr)
        
        # Add the feature vector to the list
        features_list.append(features)
    
    # Convert list of features to numpy array
    feature_matrix = np.array(features_list)
    
    return feature_matrix

def apply_pca(features, n_components=2):
    """
    Apply PCA to reduce feature dimensionality.
    
    Args:
        features: Feature matrix
        n_components: Number of principal components to keep
        
    Returns:
        PCA result and explained variance ratio
    """
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(features_scaled)
    
    return pca_result, pca.explained_variance_ratio_




