import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# 1. Initialize the App (The Web Server)
app = FastAPI()

# 2. Load the Brain (The ONNX Model)
# We load this ONCE when the app starts, not every time a user requests.
print("Loading model...")
ort_session = ort.InferenceSession("../models/mobilenetv2.onnx")
print("Model loaded!")

def preprocess_image(image_data):
    """
    The 'Brain' only understands numbers, not JPEGs.
    We must resize the image to 224x224 and convert it to a specific number format.
    """
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to numpy array
    img_array = np.array(image).astype('float32')
    
    # Normalize (Standard MobileNet Math: (Image - Mean) / Std_Dev)
    # This brings all numbers between -1 and 1 roughly.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array / 255.0 - mean) / std
    
    # Move the "Color Channels" to the front (H, W, C) -> (C, H, W)
    # Because ONNX expects: [Batch_Size, Channels, Height, Width]
    img_array = img_array.transpose(2, 0, 1)
    
    # Add the "Batch" dimension (1 image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Safety Check: Force float32 one last time
    return img_array.astype(np.float32)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # A. Read the image uploaded by the user
    image_data = await file.read()
    
    # B. Preprocess it
    input_tensor = preprocess_image(image_data)
    
    # C. Run Inference (The actual AI thinking)
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_tensor})
    
    # D. Get the result (The highest probability)
    # The model returns 1000 scores. We want the index of the highest one.
    scores = outputs[0][0]
    predicted_index = int(np.argmax(scores))
    confidence = float(scores[predicted_index])
    
    return {
        "class_index": predicted_index,
        "confidence": confidence,
        "message": "Model ran successfully!"
    }