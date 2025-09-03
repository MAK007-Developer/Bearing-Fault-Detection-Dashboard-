# dashboard.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

# Define CNN Model (moved from skin_disease_cnn.py to resolve import error)
class SkinDiseaseCNN(torch.nn.Module):
    def __init__(self, num_classes=7):
        super(SkinDiseaseCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 512)  # Adjust based on image size
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 64 * 16 * 16)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define class names (update based on your dataset, e.g., HAM10000)
class_names = ['Melanoma', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Benign Keratosis', 'Melanocytic Nevi', 'Vascular Lesions', 'Dermatofibroma']

# Load the trained model
@st.cache_resource
def load_model():
    model = SkinDiseaseCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load('models/skin_disease_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to predict skin disease
def predict_image(image, model):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1).numpy().flatten()
        predicted_class = np.argmax(probabilities)
    return class_names[predicted_class], probabilities

# Streamlit Dashboard
st.title("Skin Disease Classification Dashboard")
st.markdown("This dashboard allows you to classify skin diseases using a CNN model trained with PyTorch. Upload an image or explore the dataset and model performance.")

# Sidebar for Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Home", "Upload Image", "Dataset Insights", "Model Performance"])

# Section 1: Home
if section == "Home":
    st.header("Welcome to the Skin Disease Classification Dashboard")
    st.write("""
    This application uses a Convolutional Neural Network (CNN) to classify skin diseases from images.
    - **Dataset**: HAM10000 (or specify your dataset).
    - **Model**: Custom CNN trained with PyTorch on CPU.
    - **Classes**: 7 skin disease categories (e.g., Melanoma, Basal Cell Carcinoma).
    Use the sidebar to:
    - Upload an image for classification.
    - Explore dataset insights.
    - View model performance metrics.
    """)

# Section 2: Upload Image
elif section == "Upload Image":
    st.header("Classify a Skin Image")
    uploaded_file = st.file_uploader("Upload a skin image (JPG/PNG)", type=["jpg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", width=300)
        
        model = load_model()
        predicted_class, probabilities = predict_image(image, model)
        
        st.subheader("Prediction Results")
        st.write(f"**Predicted Disease**: {predicted_class}")
        st.write("**Prediction Probabilities**:")
        prob_df = pd.DataFrame({
            'Class': class_names,
            'Probability': [f"{p*100:.2f}%" for p in probabilities]
        })
        st.dataframe(prob_df)
        
        # Plot probabilities
        fig, ax = plt.subplots()
        ax.bar(class_names, probabilities)
        plt.xticks(rotation=45)
        ax.set_ylabel("Probability")
        ax.set_title("Class Probabilities")
        st.pyplot(fig)

# Section 3: Dataset Insights
elif section == "Dataset Insights":
    st.header("Dataset Insights")
    st.write("Explore the dataset used for training the model.")
    
    # Dataset statistics (update with your dataset's actual stats)
    st.subheader("Dataset Statistics")
    dataset_stats = {
        'Total Images': 10015,  # Example for HAM10000
        'Classes': len(class_names),
        'Training Set': 7010,
        'Validation Set': 1502,
        'Test Set': 1503
    }
    st.table(dataset_stats)
    
    # Class distribution
    st.subheader("Class Distribution")
    class_counts = [1500, 1200, 1000, 1800, 3000, 500, 1015]  # Example counts, replace with actual
    fig, ax = plt.subplots()
    ax.bar(class_names, class_counts)
    plt.xticks(rotation=45)
    ax.set_ylabel("Number of Images")
    ax.set_title("Class Distribution")
    st.pyplot(fig)

# Section 4: Model Performance
elif section == "Model Performance":
    st.header("Model Performance")
    st.write("Evaluate the performance of the trained CNN model.")
    
    # Example metrics (replace with your actual metrics)
    st.subheader("Performance Metrics")
    metrics = {
        'Accuracy': '85.6%',
        'Precision (Macro)': '83.2%',
        'Recall (Macro)': '82.9%',
        'F1-Score (Macro)': '83.0%'
    }
    st.table(metrics)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    # Example confusion matrix (replace with actual)
    y_true = [0, 1, 2, 3, 4, 5, 6] * 90  # 700 samples
    y_pred = [0, 1, 2, 3, 4, 5, 6] * 90  # y_pred = [0, 1, 2, 3, 4, 5, 6] * 95 + [1, 2, 3, 4, 0] * 5   700 samples
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    
    # Training/Validation Loss Plot
    st.subheader("Training and Validation Loss")
    train_losses = [0.9, 0.7, 0.5, 0.4, 0.3]  # Example, replace with actual
    val_losses = [1.0, 0.8, 0.6, 0.5, 0.4]  # Example, replace with actual
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and PyTorch for Skin Disease Classification Project")