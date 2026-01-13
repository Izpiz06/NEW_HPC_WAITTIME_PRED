"""
HPC Wait Time Predictor - Flask Web Application

A web interface for predicting HPC job queue wait times using 4-Model Pipeline
with Soft Voting and Conformal Prediction.

Architecture:
- Gatekeeper: GradientBoostingClassifier routes jobs to Short/Medium/Long
- Sprinter: XGBRegressor for Short jobs (<1 min)
- Runner: XGBRegressor for Medium jobs (1 min - 3 hrs)
- Marathoner: MAPIE QuantileRegressor for Long jobs (>3 hrs) with conformal intervals

Inference uses Soft Voting (weighted average) to eliminate cliff-edge errors.
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Duration class thresholds (in seconds)
SHORT_THRESHOLD = 60          # < 1 minute = Short
LONG_THRESHOLD = 10800        # > 3 hours = Long

# Load pipeline and metadata on startup
PIPELINE_PATH = 'pipeline_models.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
METADATA_PATH = 'metadata.pkl'

pipeline = None
label_encoders = None
metadata = None

def time_to_seconds_load(time_str):
    """Convert time string for data loading."""
    try:
        parts = str(time_str).split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    except:
        return 0

def load_pipeline():
    """Load the trained 4-model pipeline with MAPIE Marathoner."""
    global pipeline, label_encoders, metadata
    
    if not os.path.exists(PIPELINE_PATH):
        raise FileNotFoundError(f"Pipeline file not found: {PIPELINE_PATH}. Run hpc_eligibletime_predictor.py first.")
    
    with open(PIPELINE_PATH, 'rb') as f:
        pipeline = pickle.load(f)
    
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    
    print("Pipeline loaded successfully!")
    print("  - Gatekeeper: GradientBoostingClassifier")
    print("  - Sprinter: XGBRegressor (Short jobs)")
    print("  - Runner: XGBRegressor (Medium jobs)")
    print("  - Marathoner: MAPIE QuantileRegressor (Long jobs with conformal intervals)")
    print("  - Inference: Soft Voting (weighted average)")

def time_to_seconds(time_str):
    """Convert time string (HHH:MM:SS or HH:MM:SS) to seconds.
    Handles hours > 99 (e.g., 168:00:00)
    """
    try:
        parts = time_str.strip().split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        return 0
    except:
        return 0

def seconds_to_time(seconds):
    """Convert seconds to HHH:MM:SS format."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def get_class_name(class_idx):
    """Get human-readable class name."""
    names = {0: 'Short', 1: 'Medium', 2: 'Long'}
    return names.get(class_idx, 'Unknown')

def get_model_name(class_idx):
    """Get model name for the class."""
    names = {0: 'Sprinter', 1: 'Runner', 2: 'Marathoner'}
    return names.get(class_idx, 'Unknown')

def predict_wait_time(queue, ncpus, ngpus, arch, walltime_str):
    """
    Predict wait time using OPTIMIZED Soft Voting inference.
    
    This is the production inference logic:
    1. Get probabilities from Gatekeeper [P_short, P_medium, P_long]
    2. Run all relevant models (lazy execution for speed)
    3. Calculate weighted average: Wait = P_short*T_short + P_medium*T_medium + P_long*T_long
    4. For Long jobs (>50% probability), show conformal prediction interval
    
    Returns:
        tuple: (pred_seconds, pred_formatted, confidence_info)
    """
    walltime_seconds = time_to_seconds(walltime_str)
    
    # Handle unknown categories
    queue_val = queue if queue in label_encoders['queue'].classes_ else label_encoders['queue'].classes_[0]
    arch_val = arch if arch in label_encoders['arch'].classes_ else label_encoders['arch'].classes_[0]
    
    features = pd.DataFrame({
        'ncpus': [ncpus],
        'ngpus': [ngpus],
        'walltime_seconds': [walltime_seconds],
        'queue_encoded': [label_encoders['queue'].transform([queue_val])[0]],
        'arch_encoded': [label_encoders['arch'].transform([arch_val])[0]]
    })
    
    # Get models
    gatekeeper = pipeline['gatekeeper']
    sprinter = pipeline['sprinter']
    runner = pipeline['runner']
    marathoner = pipeline['marathoner']
    
    # Step 1: GATEKEEPER - Get Probabilities
    probs = gatekeeper.predict_proba(features)[0]  # [P_short, P_medium, P_long]
    prob_short, prob_medium, prob_long = probs
    
    # Step 2: RUN MODELS (Lazy Execution - only run if probability > 1%)
    preds = {'short': 0, 'med': 0, 'long': 0}
    intervals = None
    
    if prob_short > 0.01:
        pred_log = sprinter.predict(features)[0]
        preds['short'] = max(0, np.expm1(pred_log))
    
    if prob_medium > 0.01:
        pred_log = runner.predict(features)[0]
        preds['med'] = max(0, np.expm1(pred_log))
    
    if prob_long > 0.01:
        # For Long jobs, get MAPIE prediction with conformal intervals
        pred_log, intervals_log = marathoner.predict_interval(features)
        preds['long'] = max(0, np.expm1(pred_log[0]))
        
        # Get conformal interval bounds (90% coverage guaranteed)
        lower_log = intervals_log[0, 0, 0]
        upper_log = intervals_log[0, 1, 0]
        intervals = {
            'lower': max(0, np.expm1(lower_log)),
            'upper': max(0, np.expm1(upper_log))
        }
    
    # Step 3: SOFT VOTING - Weighted Average
    # This prevents the "15-second glitch" automatically by incorporating
    # probability of Long jobs even when classified as Short/Medium
    weighted_wait = (prob_short * preds['short']) + \
                   (prob_medium * preds['med']) + \
                   (prob_long * preds['long'])
    
    # Determine final class based on weighted prediction
    if weighted_wait < SHORT_THRESHOLD:
        final_class = 'Short'
    elif weighted_wait <= LONG_THRESHOLD:
        final_class = 'Medium'
    else:
        final_class = 'Long'
    
    # Determine confidence level based on probability distribution
    max_prob = max(probs)
    if max_prob > 0.7:
        confidence_level = 'high'
    elif max_prob > 0.5:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'
    
    # Determine primary model
    max_prob_idx = np.argmax(probs)
    model_names = ['Sprinter', 'Runner', 'Marathoner']
    primary_model = model_names[max_prob_idx]
    
    # Build confidence info
    confidence_info = {
        'original_class': final_class,
        'final_class': final_class,
        'model_used': f'Soft Voting ({primary_model})',
        'confidence_level': confidence_level,
        'probabilities': {
            'short': float(prob_short),
            'medium': float(prob_medium),
            'long': float(prob_long)
        },
        'model_predictions': {
            'short_seconds': float(preds['short']),
            'medium_seconds': float(preds['med']),
            'long_seconds': float(preds['long'])
        },
        'safety_net_triggered': False  # Soft voting replaces safety net
    }
    
    # Step 4: FORMAT OUTPUT based on risk profile
    
    # If job is definitively Long (>50% probability), show conformal interval
    if prob_long > 0.50 and intervals is not None:
        confidence_info['range'] = {
            'p10': float(intervals['lower']),
            'p10_formatted': seconds_to_time(intervals['lower']),
            'p50': float(preds['long']),
            'p50_formatted': seconds_to_time(preds['long']),
            'p90': float(intervals['upper']),
            'p90_formatted': seconds_to_time(intervals['upper']),
            'spread_seconds': float(intervals['upper'] - intervals['lower']),
            'spread_ratio': float((intervals['upper'] - intervals['lower']) / max(preds['long'], 1))
        }
        confidence_info['conformal_interval'] = True
        confidence_info['risk_message'] = f"Estimated {weighted_wait/3600:.1f}h (Range: {intervals['lower']/3600:.1f}h - {intervals['upper']/3600:.1f}h)"
    
    # If risk is split (Medium but significant Long risk)
    elif prob_long > 0.30 and intervals is not None:
        confidence_info['range'] = {
            'p10': float(intervals['lower']),
            'p10_formatted': seconds_to_time(intervals['lower']),
            'p50': float(preds['long']),
            'p50_formatted': seconds_to_time(preds['long']),
            'p90': float(intervals['upper']),
            'p90_formatted': seconds_to_time(intervals['upper']),
            'spread_seconds': float(intervals['upper'] - intervals['lower']),
            'spread_ratio': float((intervals['upper'] - intervals['lower']) / max(preds['long'], 1))
        }
        confidence_info['conformal_interval'] = True
        confidence_info['risk_message'] = f"⚠️ High Risk: {prob_long*100:.0f}% chance of >3h wait"
    
    # Standard case - no special range display needed
    else:
        confidence_info['range'] = {
            'p10': float(weighted_wait * 0.5),
            'p10_formatted': seconds_to_time(weighted_wait * 0.5),
            'p50': float(weighted_wait),
            'p50_formatted': seconds_to_time(weighted_wait),
            'p90': float(weighted_wait * 2.0),
            'p90_formatted': seconds_to_time(weighted_wait * 2.0),
            'spread_seconds': float(weighted_wait * 1.5),
            'spread_ratio': 1.5
        }
        confidence_info['conformal_interval'] = False
    
    return weighted_wait, seconds_to_time(weighted_wait), confidence_info

# HTML template embedded in Python
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HPC Wait Time Predictor</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        
        .container {
            max-width: 700px;
            margin: 0 auto;
        }
        
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
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #ccc;
        }
        
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
        
        select option {
            background: #1a1a2e;
            color: #fff;
        }
        
        .row {
            display: flex;
            gap: 15px;
        }
        
        .row .form-group {
            flex: 1;
        }
        
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
        
        button:active {
            transform: translateY(0);
        }
        
        .result {
            margin-top: 25px;
            padding: 25px;
            background: rgba(0, 212, 255, 0.1);
            border-radius: 10px;
            border: 1px solid rgba(0, 212, 255, 0.3);
            display: none;
        }
        
        .result.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result h2 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
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
        
        .prediction-seconds {
            text-align: center;
            color: #888;
            font-size: 1.1em;
        }
        
        .model-info {
            margin-top: 15px;
            text-align: center;
        }
        
        .confidence-info {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .model-badge, .class-badge, .confidence-badge {
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .model-badge {
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
        }
        
        .class-badge {
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
        }
        
        .class-badge.short { background: rgba(46, 213, 115, 0.2); color: #2ed573; }
        .class-badge.medium { background: rgba(255, 165, 2, 0.2); color: #ffa502; }
        .class-badge.long { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
        
        .confidence-badge {
            background: rgba(255, 255, 255, 0.1);
            color: #888;
        }
        
        .confidence-badge.high { background: rgba(46, 213, 115, 0.2); color: #2ed573; }
        .confidence-badge.medium { background: rgba(255, 165, 2, 0.2); color: #ffa502; }
        .confidence-badge.low { background: rgba(255, 71, 87, 0.2); color: #ff4757; }
        
        .safety-net-alert {
            width: 100%;
            padding: 8px;
            background: rgba(255, 165, 2, 0.1);
            border: 1px solid rgba(255, 165, 2, 0.3);
            border-radius: 5px;
            color: #ffa502;
            font-size: 0.85em;
            margin-top: 10px;
        }
        
        .probabilities {
            width: 100%;
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
        }
        
        .error {
            background: rgba(255, 71, 87, 0.1);
            border-color: rgba(255, 71, 87, 0.3);
            color: #ff4757;
        }
        
        .info-box {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            font-size: 0.9em;
            color: #888;
        }
        
        .info-box h3 {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 3px solid rgba(255, 255, 255, 0.1);
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⏱️ HPC Wait Time Predictor</h1>
        <p class="subtitle">Predict job queue wait time using Soft Voting + Conformal Prediction</p>
        
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
                <p><strong>Optimized 4-Model Pipeline with Soft Voting:</strong><br>
                   🚦 <strong>Gatekeeper</strong> classifies jobs as Short/Medium/Long with probabilities<br>
                   🏃 <strong>Sprinter</strong> predicts Short jobs (&lt;1 min) - MAE ~5 seconds<br>
                   🏃‍♂️ <strong>Runner</strong> predicts Medium jobs (1 min - 3 hrs)<br>
                   🏃‍♀️ <strong>Marathoner</strong> (MAPIE) predicts Long jobs (&gt;3 hrs) with 90% coverage intervals<br><br>
                   ✨ <strong>Soft Voting</strong> eliminates cliff-edge errors by weighting all model predictions.<br>
                   📊 <strong>Conformal Prediction</strong> provides guaranteed coverage for high-variance Long jobs.</p>
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
                    headers: {
                        'Content-Type': 'application/json'
                    },
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
                    
                    // Display model/confidence info
                    const info = data.confidence_info;
                    let infoHtml = `<div class="confidence-info">`;
                    infoHtml += `<span class="model-badge">${info.model_used}</span>`;
                    infoHtml += `<span class="class-badge ${info.final_class.toLowerCase()}">${info.final_class}</span>`;
                    infoHtml += `<span class="confidence-badge ${info.confidence_level}">${info.confidence_level.toUpperCase()} confidence</span>`;
                    
                    // Display risk message if present
                    if (info.risk_message) {
                        infoHtml += `<div class="safety-net-alert" style="background: rgba(255, 71, 87, 0.1); border-color: rgba(255, 71, 87, 0.3); color: #ff4757;">${info.risk_message}</div>`;
                    }
                    
                    // Display conformal prediction range if Long job probability is high
                    if (info.range && info.conformal_interval) {
                        infoHtml += `<div class="range-info" style="margin-top: 12px; padding: 12px; background: rgba(0,212,255,0.1); border-radius: 8px; border-left: 3px solid #00d4ff;">`;
                        infoHtml += `<div style="font-weight: 600; margin-bottom: 8px; color: #00d4ff;">📊 Conformal Prediction Interval (90% Coverage)</div>`;
                        infoHtml += `<div style="display: flex; justify-content: space-between; align-items: center;">`;
                        infoHtml += `<span style="color: #2ed573;">Lower: ${info.range.p10_formatted}</span>`;
                        infoHtml += `<span style="color: #ffa502; font-weight: 600;">Point Est: ${info.range.p50_formatted}</span>`;
                        infoHtml += `<span style="color: #ff4757;">Upper: ${info.range.p90_formatted}</span>`;
                        infoHtml += `</div>`;
                        infoHtml += `</div>`;
                    } else if (info.range) {
                        // Show estimated range for non-Long jobs
                        infoHtml += `<div class="range-info" style="margin-top: 12px; padding: 12px; background: rgba(0,212,255,0.1); border-radius: 8px; border-left: 3px solid #00d4ff;">`;
                        infoHtml += `<div style="font-weight: 600; margin-bottom: 8px; color: #00d4ff;">📊 Estimated Range</div>`;
                        infoHtml += `<div style="display: flex; justify-content: space-between; align-items: center;">`;
                        infoHtml += `<span style="color: #2ed573;">Best: ${info.range.p10_formatted}</span>`;
                        infoHtml += `<span style="color: #ffa502; font-weight: 600;">Expected: ${info.range.p50_formatted}</span>`;
                        infoHtml += `<span style="color: #ff4757;">Worst: ${info.range.p90_formatted}</span>`;
                        infoHtml += `</div>`;
                        infoHtml += `</div>`;
                    }
                    
                    infoHtml += `<div class="probabilities">`;
                    infoHtml += `Short: ${(info.probabilities.short * 100).toFixed(1)}% | `;
                    infoHtml += `Medium: ${(info.probabilities.medium * 100).toFixed(1)}% | `;
                    infoHtml += `Long: ${(info.probabilities.long * 100).toFixed(1)}%`;
                    infoHtml += `</div></div>`;
                    
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

def render_html():
    """Render HTML with dynamic options."""
    queue_options = '\n'.join([
        f'<option value="{q}" {"selected" if q == "standard" else ""}>{q}</option>'
        for q in metadata['queues']
    ])
    arch_options = '\n'.join([
        f'<option value="{a}" {"selected" if a == "any" else ""}>{a}</option>'
        for a in metadata['architectures']
    ])
    
    html = HTML_TEMPLATE.replace('{{QUEUE_OPTIONS}}', queue_options)
    html = html.replace('{{ARCH_OPTIONS}}', arch_options)
    return html

@app.route('/')
def index():
    """Render the main page."""
    return render_html()

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests."""
    try:
        data = request.json
        
        queue = data.get('queue', 'standard')
        ncpus = int(data.get('ncpus', 1))
        ngpus = int(data.get('ngpus', 0))
        arch = data.get('arch', 'any')
        walltime = data.get('walltime', '01:00:00')
        
        # Validate inputs
        if ncpus < 1:
            return jsonify({'error': 'Number of CPUs must be at least 1'}), 400
        if ngpus < 0:
            return jsonify({'error': 'Number of GPUs cannot be negative'}), 400
        
        # Make prediction using 4-Model Pipeline
        pred_seconds, pred_formatted, confidence_info = predict_wait_time(queue, ncpus, ngpus, arch, walltime)
        
        return jsonify({
            'prediction_seconds': float(pred_seconds),
            'prediction_formatted': pred_formatted,
            'confidence_info': confidence_info,
            'input': {
                'queue': queue,
                'ncpus': ncpus,
                'ngpus': ngpus,
                'arch': arch,
                'walltime': walltime
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/metadata')
def get_metadata():
    """Return available queues and architectures."""
    return jsonify(metadata)

if __name__ == '__main__':
    print("Loading Optimized 4-Model Pipeline...")
    load_pipeline()
    print(f"\nAvailable queues: {metadata['queues']}")
    print(f"Available architectures: {metadata['architectures']}")
    print(f"\nInference Strategy: Soft Voting (weighted average)")
    print(f"Long Job Coverage: Conformal Prediction (90% guaranteed)")
    print("\nStarting Flask server...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, host='127.0.0.1', port=5000)
