# Image Classification Web App

This is a web application that uses TensorFlow and EfficientNetB0 to classify images into four categories: water, green_area, desert, and cloudy.

## Features

- Modern, responsive UI with drag-and-drop functionality
- Real-time image preview
- High-accuracy classification using transfer learning
- Confidence score display
- Support for common image formats (JPG, PNG, JPEG)

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the web application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Click on the upload area or drag and drop an image
2. Wait for the model to process the image
3. View the prediction results and confidence score

## Model Architecture

The model uses EfficientNetB0 as a base model with the following modifications:
- Global Average Pooling
- Dropout layers for regularization
- Dense layers for classification
- Transfer learning from ImageNet weights

## Requirements

- Python 3.8+
- TensorFlow 2.15.0
- Flask 3.0.0
- Pillow 10.1.0
- NumPy 1.24.3
- scikit-learn 1.3.2 