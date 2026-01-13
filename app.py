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
        
        # Determine confidence level
        if confidence > 0.7:
            confidence_level = 'high'
        elif confidence > 0.4:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Determine class based on prediction
        SHORT_THRESHOLD = 60       # < 1 minute
        LONG_THRESHOLD = 10800     # > 3 hours
        
        if prediction_seconds < SHORT_THRESHOLD:
            final_class = 'Short'
        elif prediction_seconds <= LONG_THRESHOLD:
            final_class = 'Medium'
        else:
            final_class = 'Long'
        
        # Build confidence_info to match frontend expectations
        confidence_info = {
            'original_class': final_class,
            'final_class': final_class,
            'model_used': 'KNN',
            'confidence_level': confidence_level,
            'probabilities': {
                'short': 1.0 if final_class == 'Short' else 0.0,
                'medium': 1.0 if final_class == 'Medium' else 0.0,
                'long': 1.0 if final_class == 'Long' else 0.0
            },
            'model_predictions': {
                'short_seconds': float(prediction_seconds) if final_class == 'Short' else 0.0,
                'medium_seconds': float(prediction_seconds) if final_class == 'Medium' else 0.0,
                'long_seconds': float(prediction_seconds) if final_class == 'Long' else 0.0
            },
            'safety_net_triggered': False,
            'range': {
                'p10': float(prediction_seconds * 0.5),
                'p10_formatted': seconds_to_time(prediction_seconds * 0.5),
                'p50': float(prediction_seconds),
                'p50_formatted': seconds_to_time(prediction_seconds),
                'p90': float(prediction_seconds * 2.0),
                'p90_formatted': seconds_to_time(prediction_seconds * 2.0),
                'spread_seconds': float(prediction_seconds * 1.5),
                'spread_ratio': 1.5
            },
            'conformal_interval': False
        }
        
        return jsonify({
            'success': True,
            'prediction_seconds': float(prediction_seconds),
            'prediction_formatted': seconds_to_time(prediction_seconds),
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
            'confidence_info': confidence_info,
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

# Full HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPC Wait Time Predictor (KNN)</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container { max-width: 700px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.2em;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        .card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 30px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }
        .form-group { margin-bottom: 20px; }
        label { display: block; margin-bottom: 8px; font-weight: 500; color: #ccc; }
        input, select {
            width: 100%;
            padding: 12px 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            font-size: 16px;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            transition: all 0.3s ease;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #00d4ff;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        select option { background: #1a1a2e; color: #fff; }
        .row { display: flex; gap: 15px; }
        .row .form-group { flex: 1; }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #fff;
            border: none;
            border-radius: 8px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(0, 212, 255, 0.4);
        }
        .result {
            margin-top: 25px;
            padding: 25px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.3);
            display: none;
        }
        .result.show { display: block; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .result h2 { color: #00d4ff; margin-bottom: 15px; font-size: 1.3em; }
        .prediction-value {
            font-size: 3em;
            font-weight: bold;
            color: #fff;
            text-align: center;
            padding: 20px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            margin-bottom: 15px;
            font-family: 'Courier New', monospace;
        }
        .prediction-seconds { text-align: center; color: #888; font-size: 1.1em; }
        .model-info { margin-top: 15px; text-align: center; }
        .confidence-info { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; margin-bottom: 10px; }
        .model-badge, .class-badge, .confidence-badge {
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .model-badge { background: rgba(0, 212, 255, 0.2); color: #00d4ff; }
        .class-badge { background: rgba(255, 255, 255, 0.1); color: #fff; }
        .class-badge.short { background: rgba(46, 213, 115, 0.2); color: #2ed573; }
        .class-badge.medium { background: rgba(255, 165, 2, 0.2); color: #ffa502; }
        .class-badge.long { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
        .confidence-badge { background: rgba(255, 255, 255, 0.1); color: #888; }
        .confidence-badge.high { background: rgba(46, 213, 115, 0.2); color: #2ed573; }
        .confidence-badge.medium { background: rgba(255, 165, 2, 0.2); color: #ffa502; }
        .confidence-badge.low { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
        .probabilities { width: 100%; font-size: 0.8em; color: #666; margin-top: 5px; }
        .error { background: rgba(255, 71, 87, 0.1); border-color: rgba(255, 71, 87, 0.3); color: #ff4757; }
        .info-box {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            font-size: 0.9em;
            color: #888;
        }
        .info-box h3 { color: #00d4ff; margin-bottom: 10px; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>⏱️ HPC Wait Time Predictor</h1>
        <p class="subtitle">Simple KNN Model - Fast & Interpretable</p>
        
        <div class="card">
            <form id="predictForm">
                <div class="form-group">
                    <label for="queue">Queue</label>
                    <select id="queue" name="queue" required>
                        {{QUEUE_OPTIONS}}
                    </select>
                </div>
                
                <div class="row">
                    <div class="form-group">
                        <label for="ncpus">Number of CPUs</label>
                        <input type="number" id="ncpus" name="ncpus" value="1" min="1" required>
                    </div>
                    <div class="form-group">
                        <label for="ngpus">Number of GPUs</label>
                        <input type="number" id="ngpus" name="ngpus" value="0" min="0" required>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="arch">Architecture</label>
                    <select id="arch" name="arch" required>
                        {{ARCH_OPTIONS}}
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="walltime">Walltime (HHH:MM:SS)</label>
                    <input type="text" id="walltime" name="walltime" value="01:00:00" 
                           pattern="[0-9]+:[0-5][0-9]:[0-5][0-9]" 
                           placeholder="e.g., 24:00:00 or 168:00:00" required>
                </div>
                
                <button type="submit">Predict Wait Time</button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Calculating prediction...</p>
            </div>
            
            <div class="result" id="result">
                <h2>Predicted Wait Time</h2>
                <div class="prediction-value" id="predictionValue">--:--:--</div>
                <p class="prediction-seconds" id="predictionSeconds"></p>
                <div class="model-info" id="modelInfo"></div>
            </div>
            
            <div class="info-box">
                <h3>ℹ️ About this predictor</h3>
                <p><strong>Simple KNN Model:</strong><br>
                   🎯 K-Nearest Neighbors with distance weighting<br>
                   📊 Finds similar historical jobs and averages their wait times<br>
                   ⚡ Fast inference, interpretable results<br>
                   🔧 Features: queue, CPUs, GPUs, architecture, walltime</p>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            loading.style.display = 'block';
            result.classList.remove('show');
            
            const formData = {
                queue: document.getElementById('queue').value,
                ncpus: parseInt(document.getElementById('ncpus').value),
                ngpus: parseInt(document.getElementById('ngpus').value),
                arch: document.getElementById('arch').value,
                walltime: document.getElementById('walltime').value
            };
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.error) {
                    document.getElementById('predictionValue').textContent = 'Error';
                    document.getElementById('predictionSeconds').textContent = data.error;
                    document.getElementById('modelInfo').innerHTML = '';
                    result.classList.add('error');
                } else {
                    document.getElementById('predictionValue').textContent = data.prediction_formatted;
                    document.getElementById('predictionSeconds').textContent = 
                        `(${Math.round(data.prediction_seconds).toLocaleString()} seconds)`;
                    
                    const info = data.confidence_info;
                    let infoHtml = `<div class="confidence-info">`;
                    infoHtml += `<span class="model-badge">${info.model_used}</span>`;
                    infoHtml += `<span class="class-badge ${info.final_class.toLowerCase()}">${info.final_class}</span>`;
                    infoHtml += `<span class="confidence-badge ${info.confidence_level}">${info.confidence_level.toUpperCase()} confidence</span>`;
                    
                    if (info.range) {
                        infoHtml += `<div class="range-info" style="margin-top: 12px; padding: 12px; background: rgba(0,212,255,0.1); border-radius: 8px; border-left: 3px solid #00d4ff; width: 100%;">`;
                        infoHtml += `<div style="font-weight: 600; margin-bottom: 8px; color: #00d4ff;">📊 Estimated Range</div>`;
                        infoHtml += `<div style="display: flex; justify-content: space-between; align-items: center;">`;
                        infoHtml += `<span style="color: #2ed573;">Best: ${info.range.p10_formatted}</span>`;
                        infoHtml += `<span style="color: #ffa502; font-weight: 600;">Expected: ${info.range.p50_formatted}</span>`;
                        infoHtml += `<span style="color: #ff4757;">Worst: ${info.range.p90_formatted}</span>`;
                        infoHtml += `</div></div>`;
                    }
                    
                    infoHtml += `</div>`;
                    document.getElementById('modelInfo').innerHTML = infoHtml;
                    result.classList.remove('error');
                }
                result.classList.add('show');
            } catch (error) {
                loading.style.display = 'none';
                document.getElementById('predictionValue').textContent = 'Error';
                document.getElementById('predictionSeconds').textContent = error.message;
                result.classList.add('error', 'show');
            }
        });
    </script>
</body>
</html>
"""

def get_queue_arch_options():
    """Get available queues and architectures from the training data or encoders."""
    if label_encoders is not None:
        queues = list(label_encoders['queue'].classes_)
        archs = list(label_encoders['arch'].classes_)
    else:
        # Default values
        queues = ['standard', 'debug', 'gpu', 'preempt']
        archs = ['any', 'x86_64', 'gpu']
    return queues, archs

@app.route('/')
def home():
    """Render full HTML page."""
    queues, archs = get_queue_arch_options()
    
    queue_options = '\n'.join([
        f'<option value="{q}" {"selected" if q == "standard" else ""}>{q}</option>'
        for q in queues
    ])
    arch_options = '\n'.join([
        f'<option value="{a}" {"selected" if a == "any" else ""}>{a}</option>'
        for a in archs
    ])
    
    html = HTML_TEMPLATE.replace('{{QUEUE_OPTIONS}}', queue_options)
    html = html.replace('{{ARCH_OPTIONS}}', arch_options)
    return html

if __name__ == '__main__':
    # Try to load existing model, or train new one
    if not load_model():
        print("Training new KNN model...")
        train_knn_model()
    
    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host='127.0.0.1', port=5000, debug=True)
