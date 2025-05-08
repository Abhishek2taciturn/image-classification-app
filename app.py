from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json

app = Flask(__name__)

# Load model and class names
print("Loading model and class names...")
try:
    model = tf.keras.models.load_model('model.h5')
    class_names = np.load('class_names.npy', allow_pickle=True)
    
    # Load model metrics if available
    try:
        with open('model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
    except:
        model_metrics = {
            'accuracy': 'N/A',
            'val_accuracy': 'N/A',
            'loss': 'N/A',
            'val_loss': 'N/A'
        }
    
    print(f"Model loaded successfully. Classes: {class_names}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    # Resize image
    image = image.resize((224, 224))
    # Convert to numpy array and normalize
    image = np.array(image) / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def home():
    return render_template('index.html', metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read and preprocess image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)  # Disable prediction logging
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 