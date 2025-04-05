import os
from flask import Flask, request, jsonify
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import io
import uuid

# Initialize Flask
app = Flask(__name__)



# Configure SQLite Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance', 'images.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
db = SQLAlchemy(app)

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('instance', exist_ok=True)

# --- Database Model ---
class ImageRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    filepath = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)  # Missing column
    upload_time = db.Column(db.DateTime, default=datetime.utcnow)
    source = db.Column(db.String(20), default='mobile')  # Missing column
    def __repr__(self):
        return f"<Image {self.filename}>"

# --- Initialize Database ---
def init_db():
    with app.app_context():
        db.create_all()
        print("Database tables created successfully")

# Call this immediately
init_db()
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


# --- Image Processing ---
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# --- API Endpoint ---
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Get image file
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # Generate unique filename
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Save file
        file.save(filepath)
        file.seek(0)
        image_bytes = file.read()

        # Run prediction
        input_tensor = preprocess_image(image_bytes)
        with torch.no_grad():
            prediction = model(input_tensor)
        probabilities = torch.softmax(prediction, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

        # Store in database
        new_record = ImageRecord(
            filename=filename,
            filepath=filepath,
            prediction=str(predicted_class),
            confidence=float(confidence),
            source='mobile'
        )
        db.session.add(new_record)
        db.session.commit()
        print(f"Stored record: {new_record}")
        print(f"Prediction: {predicted_class}, Confidence: {confidence}")
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'prediction': predicted_class,
            'confidence': confidence,
            'database_id': new_record.id
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)