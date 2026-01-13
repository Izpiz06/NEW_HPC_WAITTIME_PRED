"""
HPC Wait Time Predictor - Simple KNN Flask Application

A straightforward K-Nearest Neighbors approach for predicting HPC job wait times.
Simple, interpretable, and fast.
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Model paths
MODEL_PATH = 'knn_model.pkl'
ENCODERS_PATH = 'knn_encoders.pkl'
SCALER_PATH = 'knn_scaler.pkl'

model = None
label_encoders = None
scaler = None

def time_to_seconds(time_str):
    """Convert time string (HHH:MM:SS or HH:MM:SS) to seconds."""
    try:
        parts = str(time_str).strip().split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0
    except:
        return 0

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def load_model():
    """Load the trained KNN model and encoders."""
    global model, label_encoders, scaler
    
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: KNN model not found at {MODEL_PATH}. Train it first.")
        return False
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    print("KNN model loaded successfully!")
    return True

def train_knn_model(data_path='jobsdata-6m-20251219.csv', n_neighbors=10):
    """Train a simple KNN model on the HPC data."""
    global model, label_encoders, scaler
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Convert time columns
    df['walltime_seconds'] = df['walltime'].apply(time_to_seconds)
    df['eligibletime_seconds'] = df['eligibletime'].apply(time_to_seconds)
    
    # Remove invalid entries
    df = df[df['eligibletime_seconds'] > 0]
    df = df[df['walltime_seconds'] > 0]
    
    print(f"Training on {len(df)} records...")
    
    # Encode categorical features
    label_encoders = {
        'queue': LabelEncoder(),
        'arch': LabelEncoder()
    }
    
    df['queue_encoded'] = label_encoders['queue'].fit_transform(df['queue'].fillna('unknown'))
    df['arch_encoded'] = label_encoders['arch'].fit_transform(df['arch'].fillna('unknown'))
    
    # Features
    feature_cols = ['ncpus', 'ngpus', 'walltime_seconds', 'queue_encoded', 'arch_encoded']
    X = df[feature_cols].values
    y = np.log1p(df['eligibletime_seconds'].values)  # Log transform for better scaling
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train KNN
    print(f"Training KNN with k={n_neighbors}...")
    model = KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights='distance',  # Weight by distance
        algorithm='ball_tree',
        leaf_size=30,
        n_jobs=-1
    )
    model.fit(X_scaled, y)
    
    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print("KNN model trained and saved!")
    
    # Quick evaluation
    y_pred = model.predict(X_scaled)
    mae_log = np.mean(np.abs(y - y_pred))
    mae_seconds = np.mean(np.abs(np.expm1(y) - np.expm1(y_pred)))
    print(f"Training MAE: {mae_seconds:.0f} seconds ({mae_seconds/60:.1f} minutes)")
    
    return model

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'KNN',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict wait time using KNN."""
    global model, label_encoders, scaler
    
    if model is None:
        return jsonify({'error': 'Model not loaded. Train or load model first.'}), 500
    
    try:
        data = request.get_json()
        
        # Extract parameters
        queue = data.get('queue', 'standard')
        ncpus = int(data.get('ncpus', 1))
        ngpus = int(data.get('ngpus', 0))
        arch = data.get('arch', 'any')
        walltime = data.get('walltime', '01:00:00')
        
        # Convert walltime
        walltime_seconds = time_to_seconds(walltime)
        
        # Handle unknown categories
        queue_val = queue if queue in label_encoders['queue'].classes_ else label_encoders['queue'].classes_[0]
        arch_val = arch if arch in label_encoders['arch'].classes_ else label_encoders['arch'].classes_[0]
        
        # Create feature vector
        features = np.array([[
            ncpus,
            ngpus,
            walltime_seconds,
            label_encoders['queue'].transform([queue_val])[0],
            label_encoders['arch'].transform([arch_val])[0]
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Predict (log space)
        pred_log = model.predict(features_scaled)[0]
        prediction_seconds = max(0, np.expm1(pred_log))
        
        # Get K nearest neighbors for confidence estimate
        distances, indices = model.kneighbors(features_scaled)
        avg_distance = np.mean(distances[0])
        
        # Simple confidence based on neighbor distance
        # Closer neighbors = higher confidence
        confidence = max(0.1, min(1.0, 1.0 / (1.0 + avg_distance)))
        
        return jsonify({
            'success': True,
            'prediction': {
                'seconds': float(prediction_seconds),
                'formatted': seconds_to_time(prediction_seconds),
                'minutes': float(prediction_seconds / 60),
                'hours': float(prediction_seconds / 3600)
            },
            'input': {
                'queue': queue,
                'ncpus': ncpus,
                'ngpus': ngpus,
                'arch': arch,
                'walltime': walltime
            },
            'confidence': float(confidence),
            'model': 'KNN',
            'k_neighbors': model.n_neighbors
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train():
    """Train the KNN model (admin endpoint)."""
    try:
        data = request.get_json() or {}
        n_neighbors = int(data.get('n_neighbors', 10))
        data_path = data.get('data_path', 'jobsdata-6m-20251219.csv')
        
        train_knn_model(data_path=data_path, n_neighbors=n_neighbors)
        
        return jsonify({
            'success': True,
            'message': 'KNN model trained successfully',
            'n_neighbors': n_neighbors
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    """Simple home page."""
    return '''
    <html>
    <head><title>HPC Wait Time Predictor (KNN)</title></head>
    <body>
        <h1>HPC Wait Time Predictor</h1>
        <h2>Simple KNN Model</h2>
        <p>Endpoints:</p>
        <ul>
            <li><code>POST /predict</code> - Predict wait time</li>
            <li><code>POST /train</code> - Train model</li>
            <li><code>GET /health</code> - Health check</li>
        </ul>
        <h3>Example:</h3>
        <pre>
curl -X POST http://localhost:5000/predict \\
  -H "Content-Type: application/json" \\
  -d '{"queue": "standard", "ncpus": 4, "ngpus": 0, "arch": "any", "walltime": "01:00:00"}'
        </pre>
    </body>
    </html>
    '''

if __name__ == '__main__':
    # Try to load existing model, or train new one
    if not load_model():
        print("Training new KNN model...")
        train_knn_model()
    
    print("\nStarting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
