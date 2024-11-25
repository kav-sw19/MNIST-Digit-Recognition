# MNIST Digit Recognition with PyTorch and Streamlit
An interactive web application that recognizes hand-drawn digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset. 
Users can draw digits directly in the browser and get real-time predictions.
## ⭐ Features
1. Interactive drawing canvas
2. Real-time digit recognition
3. Pre-trained CNN model (~99.60% accuracy)
4. Simple and intuitive user interface
## 🛠️ Technologies Used
1. PyTorch - Deep Learning framework
2. Streamlit - Web interface
3. torchvision - Image processing and datasets
4. streamlit-drawable-canvas - Drawing interface
## 📋 Prerequisites
- Python 3.8+
 pip package manager
If any errors do occur, use Python 3.9.13 (Version I used)
## 🔧 Installation
1. Clone the repository
2. Install required packages - Do so entering the following in bash:
    pip install -r requirements.txt
## 🎮 Usage
1. Run the Streamlit application entering the following in bash:
    streamlit run main.py
   
2. Open your browser and navigate to the provided local URL
3. Draw a digit (0-9) on the canvas
4. Click "Predict" to see the model's prediction

## 🏋️‍♂️ Training the Model
Note that the model has already been trained and stored on the .pth file.
Has execute 20 epochs and has ~99.60% training accuracy and ~99.20% validation accuracy
To train the model from scratch:
    python train_model.py
Feel free to change the constraints for training by amending the n_epochs and patience variables in train_model.py
