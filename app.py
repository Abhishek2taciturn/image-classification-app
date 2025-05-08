from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model and class names
logger.info("Loading model and class names...")
try:
    model = tf.keras.models.load_model('model.h5')
    class_names = np.load('class_names.npy', allow_pickle=True)
    
    # Load model metrics if available
    try:
        with open('model_metrics.json', 'r') as f:
            model_metrics = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load metrics file: {str(e)}")
        model_metrics = {
            'accuracy': 'N/A',
            'val_accuracy': 'N/A',
            'loss': 'N/A',
            'val_loss': 'N/A'
        }
    
    logger.info(f"Model loaded successfully. Classes: {class_names}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    try:
        # Resize image
        image = image.resize((224, 224))
        # Convert to numpy array and normalize
        image = np.array(image) / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html', metrics=model_metrics)

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'model_loaded': model is not None,
        'classes': class_names.tolist() if class_names is not None else []
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        logger.info("Received prediction request")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400
        
        logger.info(f"Processing file: {file.filename}")
        
        # Read and preprocess image
        try:
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            logger.info("Image loaded successfully")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return jsonify({'error': 'Invalid image file'}), 400
        
        try:
            processed_image = preprocess_image(image)
            logger.info("Image preprocessed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return jsonify({'error': 'Error preprocessing image'}), 500
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = float(np.max(predictions[0]))
            logger.info(f"Prediction successful: {predicted_class} ({confidence:.2f})")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({'error': 'Error making prediction'}), 500
        
        response = {
            'class': predicted_class,
            'confidence': confidence
        }
        logger.info(f"Sending response: {response}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Error processing image',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port) 