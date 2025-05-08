from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
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
CORS(app)

# Global variables
model = None
class_names = None
model_metrics = {
    'accuracy': 'N/A',
    'val_accuracy': 'N/A',
    'loss': 'N/A',
    'val_loss': 'N/A'
}

def load_model():
    global model, class_names, model_metrics
    try:
        logger.info("Loading model and class names...")
        model = tf.keras.models.load_model('model.h5')
        class_names = np.load('class_names.npy', allow_pickle=True)
        
        try:
            with open('model_metrics.json', 'r') as f:
                model_metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metrics file: {str(e)}")
        
        logger.info(f"Model loaded successfully. Classes: {class_names}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

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
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Read image
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Resize and preprocess
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        predictions = model.predict(image, verbose=0)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Error processing image',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Load model before starting the server
    if not load_model():
        logger.error("Failed to load model. Server will start but predictions will not work.")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port) 