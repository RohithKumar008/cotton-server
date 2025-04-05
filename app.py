import os
import io
import psycopg2
import torch
import torchvision.transforms as transforms
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# Connect to PostgreSQL database on Render
DATABASE_URL = os.environ.get('DATABASE_URL')
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Create table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS image_results (
        id SERIAL PRIMARY KEY,
        image BYTEA,
        timestamp TIMESTAMP,
        prediction TEXT,
        confidence REAL
    )
''')
conn.commit()

# Load your model
model = torch.load('best_model_large.pth', map_location=torch.device('cpu'))
model.eval()

# Define your image transformation
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
@app.route('/')
def home():
    return "Welcome to the Cotton Disease Classification API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        file = request.files['image']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)
    
        with torch.no_grad():
            output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = probabilities[predicted_class].item()
    
        result = f"Predicted class: {predicted_class}"
        timestamp = datetime.utcnow()
    
        # Save image, timestamp and result to PostgreSQL
        cursor.execute(
            "INSERT INTO image_results (image, timestamp, prediction,confidence) VALUES (%s, %s, %s, %s)",
            (psycopg2.Binary(image_bytes), timestamp, result,confidence)
        )
        conn.commit()
    
        return jsonify({'prediction': result, 'timestamp': str(timestamp), 'confidence':confidence })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)
