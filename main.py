import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from streamlit_drawable_canvas import st_canvas

# Define the CNN model for digit recognition
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 32 output channels, 5x5 kernel
        # Second convolutional layer: 32 input channels, 64 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        # Dropout layer for regularization
        self.conv2_drop = nn.Dropout2d()

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)  # First FC layer reduces to 512 neurons
        self.fc2 = nn.Linear(512, 10)    # Output layer with 10 classes (digits 0-9)
        self.dropout = nn.Dropout(p=0.5)  # Dropout with 50% probability

    def forward(self, x):
        # First conv layer -> ReLU activation -> 2x2 max pooling
        # Second conv layer -> Dropout -> ReLU activation -> 2x2 max pooling
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 1024)
        # First FC layer with ReLU activation
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)

        # Output layer
        x = self.fc2(x)

        # Apply log softmax for numerical stability
        return torch.log_softmax(x, dim=1)

# Load the pre-trained model using Streamlit's caching
@st.cache_resource
def load_model():
    model = CNN()
    # Load the pre-trained weights from file
    # Note that mnist_cnn.pth is already trained and hsa run 20 epochs, with ~99.60% accuracy
    # Model can be retrained with train_model.py
    # Can modify constraints to train for less/more epochs
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_image(image):
    """
    Preprocess the input image for the model:
    1. Convert to grayscale
    2. Resize to 28x28 pixels
    3. Convert to tensor and normalize
    """
    
    image = image.convert('L') # Convert to grayscale
    image = image.resize((28, 28)) # Resize to 28x28 (MNIST format)

    # Convert to tensor and normalize using MNIST mean (0.1307) and std (0.3081)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def main():
    # PAGE #
    st.title("MNIST Digit Recognition")
    st.write("Draw a digit (0-9) in the canvas below:")

    # Canvas for user to draw digits on
    canvas_result = st_canvas(
        stroke_width=20,          # Width of drawing brush
        stroke_color="#fff",      # White brush color
        background_color="#000",  # Black background
        height=280,              # Canvas height
        width=280,               # Canvas width
        drawing_mode="freedraw", # Free drawing mode
        key="canvas",           # Unique key for Streamlit
    )

    # Process the drawing when available
    if canvas_result.image_data is not None:
        # Convert the numpy array to PIL Image
        image = Image.fromarray(canvas_result.image_data.astype('uint8'))
        
        if st.button('Predict'):
            # Load the trained model
            model = load_model()
            
            # Preprocess the drawn image
            tensor = preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():  # Disable gradient calculation for inference
                output = model(tensor)
                # Get the predicted digit (class with highest probability)
                prediction = output.argmax(dim=1, keepdim=True).item()
            
            # Display the prediction
            st.markdown(f"<h2>Prediction: {prediction}</h2>", unsafe_allow_html=True)

# Exceution
if __name__ == "__main__":
    main()