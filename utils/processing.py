import numpy as np
from PIL import Image
import tensorflow as tf
from utils.logger import logger

def preprocess_image(image):
    """
    Preprocesses the image for the FibonacciNet model.
    Steps:
    1. Resize to target size (224, 224).
    2. Convert to numpy array.
    3. Ensure 3 channels (RGB).
    4. batch dimension expansion.
    5. Rescale pixel values to [0, 1] (Standard for custom trained models).
    """
    logger.info(f"Preprocessing image of size: {image.size} and mode: {image.mode}")
    # Resize to 224x224
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize((224, 224))
    img_array = np.array(image)
    
    # Expand dims to create batch (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # normalized inputs if trained with standard pipelines.
    img_array = img_array.astype("float32") / 255.0
    
    return img_array