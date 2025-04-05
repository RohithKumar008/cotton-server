import os
import io
import psycopg2
import torch
import torch.nn as nn
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

# Define your image transformation
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- PyTorch Model Setup ---
class CNNmodel(nn.Module):
  def __init__(self,classes):
    super().__init__()

    self.conv1=nn.Conv2d(3,32,kernel_size=4,stride=1,padding=0)
    self.bn1=nn.BatchNorm2d(32)

    self.conv2=nn.Conv2d(32,64,kernel_size=4,stride=1, padding=0)
    self.bn2=nn.BatchNorm2d(64)

    self.conv3=nn.Conv2d(64,128,kernel_size=4,stride=1,padding=0)
    self.bn3=nn.BatchNorm2d(128)

    self.conv4=nn.Conv2d(128,128,kernel_size=4,stride=1,padding=0)
    self.bn4=nn.BatchNorm2d(128)

    self.pool=nn.MaxPool2d(kernel_size=3, stride=3)
    self.pool2=nn.MaxPool2d(kernel_size=3,stride=2)

    self.fc1=nn.Linear(6*6*128,512)
    self.fc2=nn.Linear(512,classes)

    self.flatten=nn.Flatten()
    self.relu=nn.ReLU()
    self.dropout=nn.Dropout(0.5)

  def forward(self,x):
    x=self.conv1(x)
    x=self.bn1(x)
    x=self.relu(x)
    x=self.pool(x)

    x=self.conv2(x)
    x=self.bn2(x)
    x=self.relu(x)
    x=self.pool(x)

    x=self.conv3(x)
    x=self.bn3(x)
    x=self.relu(x)
    x=self.pool2(x)

    x=self.conv4(x)
    x=self.bn4(x)
    x=self.relu(x)
    x=self.flatten(x)

    x=self.fc1(x)
    x=self.relu(x)
    x=self.dropout(x)

    x=self.fc2(x)
    return x

# Load your trained PyTorch model
model = CNNmodel(7)  # Replace with your model class
model.load_state_dict(torch.load("best_model_large.pth", map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set to evaluation mode

@app.route('/')
def home():
    return "koushik erri***ka!!"

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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
