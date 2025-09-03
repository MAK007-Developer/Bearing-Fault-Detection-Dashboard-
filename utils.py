import pandas as pd
import numpy as np
from scipy import stats
import os
import json
import datetime
import uuid

# File-based functions to replace database functions
def get_all_datasets():
    """Get metadata for all datasets in the datasets directory"""
    datasets = []
    if os.path.exists("datasets"):
        for file in os.listdir("datasets"):
            if file.lower().endswith('.csv'):
                # Determine fault type from filename
                fault_type = "Unknown"
                if "nb" in file.lower():
                    fault_type = "NB"
                elif "ir-7" in file.lower() or "ir - 7" in file.lower() or "ir7" in file.lower():
                    fault_type = "IR7"
                elif "ir-21" in file.lower() or "ir - 21" in file.lower() or "ir21" in file.lower():
                    fault_type = "IR21"
                elif "or-7" in file.lower() or "or - 7" in file.lower() or "or7" in file.lower():
                    fault_type = "OR7"
                elif "or-21" in file.lower() or "or - 21" in file.lower() or "or21" in file.lower():
                    fault_type = "OR21"
                
                # Add dataset metadata
                datasets.append({
                    "id": file,
                    "name": file,
                    "fault_type": fault_type,
                    "source": "File Upload",
                    "created_at": datetime.datetime.fromtimestamp(
                        os.path.getctime(os.path.join("datasets", file))
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })
    return datasets

def get_dataset(dataset_id):
    """Get dataset by ID (filename)"""
    if os.path.exists(os.path.join("datasets", dataset_id)):
        try:
            df = pd.read_csv(os.path.join("datasets", dataset_id))
            
            # Check if it has the required columns
            if 'DE' not in df.columns or 'FE' not in df.columns:
                if len(df.columns) >= 2:
                    # Rename the first two columns to DE and FE
                    col_names = list(df.columns)
                    df = df.rename(columns={col_names[0]: 'DE', col_names[1]: 'FE'})
            
            return df
        except Exception as e:
            print(f"Error loading dataset {dataset_id}: {e}")
            return None
    return None

def save_dataset(df, name, fault_type="Unknown", description=""):
    """Save dataset to file"""
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
        
    # Use the provided name for the filename (with some sanitization)
    # Replace spaces and special chars with underscores
    safe_name = name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_" or c == "-")
    
    # Add timestamp to avoid overwriting files with the same name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_name}_{timestamp}.csv"
    filepath = os.path.join("datasets", filename)
    
    # Save the DataFrame
    df.to_csv(filepath, index=False)
    
    # Update metadata
    metadata_path = os.path.join("datasets", "metadata.json")
    metadata = []
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except:
            metadata = []
    
    # Add new dataset metadata
    metadata.append({
        "id": filename,
        "name": name,
        "fault_type": fault_type,
        "description": description,
        "source": "User Upload",
        "created_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return filename

def delete_dataset(dataset_id):
    """Delete dataset by ID (filename)"""
    if os.path.exists(os.path.join("datasets", dataset_id)):
        try:
            os.remove(os.path.join("datasets", dataset_id))
            
            # Update metadata
            metadata_path = os.path.join("datasets", "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Remove deleted dataset from metadata
                    metadata = [d for d in metadata if d.get("id") != dataset_id]
                    
                    # Save updated metadata
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                except:
                    pass
            
            return True
        except Exception as e:
            print(f"Error deleting dataset {dataset_id}: {e}")
            return False
    return False

def process_uploaded_file(uploaded_file, name, fault_type="Unknown", description=""):
    """Process an uploaded file and save it to the datasets directory"""
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            # Check if it has the required columns
            if 'DE' not in df.columns or 'FE' not in df.columns:
                if len(df.columns) >= 2:
                    # Rename the first two columns to DE and FE
                    col_names = list(df.columns)
                    df = df.rename(columns={col_names[0]: 'DE', col_names[1]: 'FE'})
            
            # Save the dataset
            return save_dataset(df, name, fault_type, description)
        except Exception as e:
            print(f"Error processing uploaded file: {e}")
            return None
    return None

def load_data(file_path):
    """
    Load data from a CSV file.
    In a real implementation, this would load actual data files.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the data
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def generate_stats(data):
    """
    Generate statistics for the given data.
    
    Args:
        data: DataFrame containing DE and FE columns
        
    Returns:
        DataFrame with statistics for each signal
    """
    stats_df = pd.DataFrame({
        "DE_Mean": [data['DE'].mean()],
        "DE_StdDev": [data['DE'].std()],
        "DE_Skewness": [stats.skew(data['DE'])],
        "DE_Kurtosis": [stats.kurtosis(data['DE'])],
        "FE_Mean": [data['FE'].mean()],
        "FE_StdDev": [data['FE'].std()],
        "FE_Skewness": [stats.skew(data['FE'])],
        "FE_Kurtosis": [stats.kurtosis(data['FE'])]
    })
    
    return stats_df

def create_sample_data(fault_type):
    """
    Create sample data based on the fault type.
    First checks if a CSV file with the fault type name exists in datasets directory.
    Otherwise, generates synthetic data.
    
    Args:
        fault_type: Type of fault (NB, IR7, IR21, OR7, OR21)
        
    Returns:
        DataFrame with DE and FE columns
    """
    # Check if we have real CSV files to use
    if os.path.exists("datasets"):
        # Map of file types to potential filenames
        filename_map = {
            "NB": ["NB.csv"],
            "IR7": ["IR-7.csv", "IR - 7.csv", "IR7.csv"],
            "IR21": ["IR-21.csv", "IR - 21.csv", "IR21.csv"],
            "OR7": ["OR-7.csv", "OR - 7.csv", "OR7.csv"],
            "OR21": ["OR-21.csv", "OR - 21.csv", "OR21.csv"]
        }
        
        # Try to load from a file
        if fault_type in filename_map:
            for filename in filename_map[fault_type]:
                file_path = os.path.join("datasets", filename)
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        
                        # Check if it has the required columns
                        if 'DE' not in df.columns or 'FE' not in df.columns:
                            if len(df.columns) >= 2:
                                # Rename the first two columns to DE and FE
                                col_names = list(df.columns)
                                df = df.rename(columns={col_names[0]: 'DE', col_names[1]: 'FE'})
                        
                        print(f"Loaded {fault_type} data from {file_path}")
                        return df
                    except Exception as e:
                        print(f"Error loading from file {file_path}: {e}")
    
    # If file loading failed, create synthetic data
    print(f"Generating synthetic data for {fault_type}")
    np.random.seed(42 + hash(fault_type) % 100)
    
    # Number of data points
    n_points = 1000
    
    # Time vector
    t = np.linspace(0, 1, n_points)
    
    # Base frequency and amplitude parameters
    base_freq = 50
    base_amp = 1.0
    noise_level = 0.1
    
    # Generate different signal patterns based on fault type
    if fault_type == "NB":  # Normal baseline
        de_signal = base_amp * np.sin(2 * np.pi * base_freq * t) + noise_level * np.random.randn(n_points)
        fe_signal = 0.8 * base_amp * np.sin(2 * np.pi * base_freq * t + np.pi/4) + noise_level * np.random.randn(n_points)
    
    elif fault_type == "IR7":  # Inner race fault (minor)
        # Add high frequency components for inner race fault
        ir_freq = 3.5 * base_freq
        ir_amp = 0.3
        de_signal = base_amp * np.sin(2 * np.pi * base_freq * t) + \
                  ir_amp * np.sin(2 * np.pi * ir_freq * t) * np.exp(-5 * (t % 0.2)) + \
                  noise_level * np.random.randn(n_points)
        fe_signal = 0.8 * base_amp * np.sin(2 * np.pi * base_freq * t + np.pi/4) + \
                  0.2 * ir_amp * np.sin(2 * np.pi * ir_freq * t) * np.exp(-5 * (t % 0.2)) + \
                  noise_level * np.random.randn(n_points)
    
    elif fault_type == "IR21":  # Inner race fault (severe)
        # Add stronger high frequency components for severe inner race fault
        ir_freq = 3.5 * base_freq
        ir_amp = 0.7
        de_signal = base_amp * np.sin(2 * np.pi * base_freq * t) + \
                  ir_amp * np.sin(2 * np.pi * ir_freq * t) * np.exp(-3 * (t % 0.1)) + \
                  noise_level * 1.5 * np.random.randn(n_points)
        fe_signal = 0.8 * base_amp * np.sin(2 * np.pi * base_freq * t + np.pi/4) + \
                  0.4 * ir_amp * np.sin(2 * np.pi * ir_freq * t) * np.exp(-3 * (t % 0.1)) + \
                  noise_level * 1.2 * np.random.randn(n_points)
    
    elif fault_type == "OR7":  # Outer race fault (minor)
        # Add periodic impacts for outer race fault
        or_freq = 2.2 * base_freq
        or_amp = 0.4
        impact_idx = (t * or_freq).astype(int) % 1 < 0.2
        de_signal = base_amp * np.sin(2 * np.pi * base_freq * t) + \
                  or_amp * impact_idx * np.sin(2 * np.pi * 4 * base_freq * t) + \
                  noise_level * np.random.randn(n_points)
        fe_signal = 0.8 * base_amp * np.sin(2 * np.pi * base_freq * t + np.pi/4) + \
                  0.3 * or_amp * impact_idx * np.sin(2 * np.pi * 4 * base_freq * t) + \
                  noise_level * np.random.randn(n_points)
    
    elif fault_type == "OR21":  # Outer race fault (severe)
        # Add stronger periodic impacts for severe outer race fault
        or_freq = 2.2 * base_freq
        or_amp = 0.9
        impact_idx = (t * or_freq).astype(int) % 1 < 0.3
        de_signal = base_amp * np.sin(2 * np.pi * base_freq * t) + \
                  or_amp * impact_idx * np.sin(2 * np.pi * 4 * base_freq * t) + \
                  noise_level * 1.5 * np.random.randn(n_points)
        fe_signal = 0.8 * base_amp * np.sin(2 * np.pi * base_freq * t + np.pi/4) + \
                  0.6 * or_amp * impact_idx * np.sin(2 * np.pi * 4 * base_freq * t) + \
                  noise_level * 1.2 * np.random.randn(n_points)
    
    else:
        # Default case - normal signal with more noise
        de_signal = base_amp * np.sin(2 * np.pi * base_freq * t) + 2 * noise_level * np.random.randn(n_points)
        fe_signal = 0.8 * base_amp * np.sin(2 * np.pi * base_freq * t + np.pi/4) + 2 * noise_level * np.random.randn(n_points)
    
    # Create DataFrame
    data = pd.DataFrame({
        'DE': de_signal,
        'FE': fe_signal
    })
    
    return data




