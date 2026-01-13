"""
HPC Eligible Time Predictor - 2-Stage Classification + Regression

Stage 1: XGBoost Classifier predicts wait-time category
Stage 2: Specialized regressors for each category
  - Class 0 (<1h): KNN (good for short patterns)
  - Class 1 (1-5h): XGBoost
  - Class 2 (5-24h): XGBoost

Includes confidence scoring based on classifier probabilities.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, classification_report, confusion_matrix
import xgboost as xgb
import pickle
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# WAIT-TIME CLASSES
# =============================================================================

CLASS_THRESHOLDS = {
    0: (0, 3600),        # Class 0: < 1 hour
    1: (3600, 18000),    # Class 1: 1-5 hours  
    2: (18000, 86400),   # Class 2: 5-24 hours
}

CLASS_NAMES = {
    0: '< 1 hour',
    1: '1-5 hours',
    2: '5-24 hours'
}

# =============================================================================
# USER-DEFINED FEATURE WEIGHTS
# =============================================================================

FEATURE_WEIGHTS = {
    'ncpus': 1.0,
    'ngpus': 3.0,
    'walltime_seconds': 0.8,
    'queue_encoded': 1.5,
    'arch_encoded': 2.0,
}

# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = 'hpc_model.pkl'
ENCODERS_PATH = 'hpc_encoders.pkl'
SCALER_PATH = 'hpc_scaler.pkl'
METADATA_PATH = 'hpc_metadata.pkl'

USE_GPU = True  # Set to False if no CUDA GPU available

# =============================================================================
# Utility Functions
# =============================================================================

def time_to_seconds(time_str):
    """Convert time string (HHH:MM:SS) to seconds."""
    try:
        parts = str(time_str).strip().split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    except:
        return 0

def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    seconds = max(0, int(seconds))
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def assign_class(eligible_seconds):
    """Assign wait-time class based on thresholds."""
    if eligible_seconds < CLASS_THRESHOLDS[0][1]:
        return 0
    elif eligible_seconds < CLASS_THRESHOLDS[1][1]:
        return 1
    else:
        return 2

def get_confidence_level(max_prob):
    """Convert probability to confidence level."""
    if max_prob >= 0.8:
        return 'High'
    elif max_prob >= 0.5:
        return 'Medium'
    else:
        return 'Low'

# =============================================================================
# Feature Engineering
# =============================================================================

def apply_feature_weights(X, weights=None):
    """Apply user-defined weights to features."""
    if weights is None:
        weights = FEATURE_WEIGHTS
    X_weighted = X.copy()
    for col in X.columns:
        if col in weights:
            X_weighted[col] = X_weighted[col] * weights[col]
    return X_weighted

def extract_features(df):
    """Extract and encode features from dataframe."""
    features = pd.DataFrame()
    label_encoders = {}
    
    # Numeric features
    features['ncpus'] = df['ncpus'].astype(float)
    features['ngpus'] = df['ngpus'].astype(float)
    features['walltime_seconds'] = df['walltime_seconds'].astype(float)
    
    # Encode categorical features
    le_queue = LabelEncoder()
    features['queue_encoded'] = le_queue.fit_transform(df['queue'].astype(str))
    label_encoders['queue'] = le_queue
    
    le_arch = LabelEncoder()
    features['arch_encoded'] = le_arch.fit_transform(df['arch'].astype(str))
    label_encoders['arch'] = le_arch
    
    return features, label_encoders

# =============================================================================
# Model Training
# =============================================================================

def train_models(data_path='jobsdata-6m-20251219.csv', n_neighbors=15):
    """Train 2-stage model: Classifier + Class-specific Regressors."""
    
    print("=" * 70)
    print("HPC WAIT TIME PREDICTOR - 2-STAGE TRAINING")
    print("=" * 70)
    
    if USE_GPU:
        print("🚀 GPU Acceleration: ENABLED")
    else:
        print("💻 CPU Mode")
    
    # -------------------------------------------------------------------------
    # Load and Prepare Data
    # -------------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    
    df['walltime_seconds'] = df['walltime'].apply(time_to_seconds)
    df['eligibletime_seconds'] = df['eligibletime'].apply(time_to_seconds)
    
    # Filter: valid entries, eligible time <= 24 hours
    MAX_ELIGIBLE = 24 * 3600
    df = df[(df['eligibletime_seconds'] > 0) & 
            (df['eligibletime_seconds'] <= MAX_ELIGIBLE) & 
            (df['walltime_seconds'] > 0)]
    print(f"Valid records (0 < eligible <= 24h): {len(df)}")
    
    # Assign classes
    df['wait_class'] = df['eligibletime_seconds'].apply(assign_class)
    
    # Print class distribution
    print("\nClass Distribution:")
    for cls in [0, 1, 2]:
        count = (df['wait_class'] == cls).sum()
        pct = 100 * count / len(df)
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {count} ({pct:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Extract Features
    # -------------------------------------------------------------------------
    print("\n[2/5] Extracting features...")
    X, label_encoders = extract_features(df)
    y_seconds = df['eligibletime_seconds'].values
    y_class = df['wait_class'].values
    
    # Apply feature weights
    X_weighted = apply_feature_weights(X)
    
    # Split data
    X_train, X_test, y_train_sec, y_test_sec, y_train_cls, y_test_cls = train_test_split(
        X_weighted, y_seconds, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Scale for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # -------------------------------------------------------------------------
    # Stage 1: Train Classifier
    # -------------------------------------------------------------------------
    print("\n[3/5] Training Stage 1: XGBoost Classifier...")
    
    clf_params = {
        'n_estimators': 150,
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'multi:softprob',
        'num_class': 3,
        'random_state': 42,
        'verbosity': 0
    }
    
    if USE_GPU:
        clf_params['tree_method'] = 'hist'
        clf_params['device'] = 'cuda'
    else:
        clf_params['n_jobs'] = -1
    
    classifier = xgb.XGBClassifier(**clf_params)
    classifier.fit(X_train, y_train_cls)
    
    # Evaluate classifier
    y_cls_pred = classifier.predict(X_test)
    print("\nClassifier Performance:")
    print(classification_report(y_test_cls, y_cls_pred, target_names=list(CLASS_NAMES.values())))
    
    # -------------------------------------------------------------------------
    # Stage 2: Train Class-Specific Regressors
    # -------------------------------------------------------------------------
    print("[4/5] Training Stage 2: Class-Specific Regressors...")
    
    regressors = {}
    
    # Class 0 (<1h): KNN - good for short, similar patterns
    print("\n  Training Class 0 (<1h) Regressor: KNN...")
    mask_0 = y_train_cls == 0
    if mask_0.sum() > 0:
        knn_short = KNeighborsRegressor(
            n_neighbors=min(n_neighbors, mask_0.sum()),
            weights='distance',
            algorithm='ball_tree',
            n_jobs=-1
        )
        knn_short.fit(X_train_scaled[mask_0], np.log1p(y_train_sec[mask_0]))
        regressors[0] = {'model': knn_short, 'type': 'knn', 'scaled': True}
        print(f"    Trained on {mask_0.sum()} samples")
    
    # Class 1 (1-5h): XGBoost
    print("\n  Training Class 1 (1-5h) Regressor: XGBoost...")
    mask_1 = y_train_cls == 1
    if mask_1.sum() > 0:
        xgb_params = {
            'n_estimators': 150,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbosity': 0
        }
        if USE_GPU:
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = 'cuda'
        else:
            xgb_params['n_jobs'] = -1
        
        xgb_medium = xgb.XGBRegressor(**xgb_params)
        xgb_medium.fit(X_train[mask_1], np.log1p(y_train_sec[mask_1]))
        regressors[1] = {'model': xgb_medium, 'type': 'xgboost', 'scaled': False}
        print(f"    Trained on {mask_1.sum()} samples")
    
    # Class 2 (5-24h): XGBoost
    print("\n  Training Class 2 (5-24h) Regressor: XGBoost...")
    mask_2 = y_train_cls == 2
    if mask_2.sum() > 0:
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.08,
            'random_state': 42,
            'verbosity': 0
        }
        if USE_GPU:
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = 'cuda'
        else:
            xgb_params['n_jobs'] = -1
        
        xgb_long = xgb.XGBRegressor(**xgb_params)
        xgb_long.fit(X_train[mask_2], np.log1p(y_train_sec[mask_2]))
        regressors[2] = {'model': xgb_long, 'type': 'xgboost', 'scaled': False}
        print(f"    Trained on {mask_2.sum()} samples")
    
    # -------------------------------------------------------------------------
    # Evaluate Full Pipeline
    # -------------------------------------------------------------------------
    print("\n[5/5] Evaluating Full Pipeline...")
    
    y_pred_seconds = np.zeros(len(X_test))
    y_pred_class = classifier.predict(X_test)
    
    for cls in [0, 1, 2]:
        mask = y_pred_class == cls
        if mask.sum() > 0 and cls in regressors:
            reg_info = regressors[cls]
            if reg_info['scaled']:
                X_input = X_test_scaled[mask]
            else:
                X_input = X_test[mask]
            
            pred_log = reg_info['model'].predict(X_input)
            y_pred_seconds[mask] = np.expm1(pred_log)
    
    # Clip predictions to valid range
    y_pred_seconds = np.clip(y_pred_seconds, 0, MAX_ELIGIBLE)
    
    # Overall metrics
    mae = mean_absolute_error(y_test_sec, y_pred_seconds)
    print(f"\n{'='*70}")
    print("OVERALL RESULTS")
    print(f"{'='*70}")
    print(f"MAE: {mae:.0f}s ({mae/60:.1f} min, {mae/3600:.2f} hours)")
    
    # Per-class metrics
    print("\nPer-Class MAE:")
    for cls in [0, 1, 2]:
        mask = y_test_cls == cls
        if mask.sum() > 0:
            cls_mae = mean_absolute_error(y_test_sec[mask], y_pred_seconds[mask])
            print(f"  Class {cls} ({CLASS_NAMES[cls]}): {cls_mae:.0f}s ({cls_mae/60:.1f} min)")
    
    # -------------------------------------------------------------------------
    # Save Models
    # -------------------------------------------------------------------------
    print(f"\n{'-'*40}")
    print("SAVING MODELS")
    print(f"{'-'*40}")
    
    models = {
        'classifier': classifier,
        'regressors': regressors,
        'feature_weights': FEATURE_WEIGHTS,
        'class_thresholds': CLASS_THRESHOLDS,
        'class_names': CLASS_NAMES
    }
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(models, f)
    print(f"Models saved to '{MODEL_PATH}'")
    
    with open(ENCODERS_PATH, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Encoders saved to '{ENCODERS_PATH}'")
    
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to '{SCALER_PATH}'")
    
    metadata = {
        'queues': sorted(df['queue'].unique().tolist()),
        'architectures': sorted(df['arch'].unique().tolist()),
        'feature_weights': FEATURE_WEIGHTS,
        'class_thresholds': CLASS_THRESHOLDS
    }
    
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to '{METADATA_PATH}'")
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    
    return models, label_encoders, scaler, metadata

# =============================================================================
# Prediction Function
# =============================================================================

def load_models():
    """Load trained models."""
    with open(MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
    with open(ENCODERS_PATH, 'rb') as f:
        label_encoders = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
    return models, label_encoders, scaler, metadata

def predict(models, label_encoders, scaler, queue, ncpus, ngpus, arch, walltime_str):
    """
    Predict wait time using 2-stage approach with confidence scoring.
    
    Returns:
        dict with prediction, class, confidence, and details
    """
    classifier = models['classifier']
    regressors = models['regressors']
    
    # Convert walltime
    walltime_seconds = time_to_seconds(walltime_str)
    
    # Handle unknown categories
    queue_val = queue if queue in label_encoders['queue'].classes_ else label_encoders['queue'].classes_[0]
    arch_val = arch if arch in label_encoders['arch'].classes_ else label_encoders['arch'].classes_[0]
    
    # Create feature vector
    features = pd.DataFrame({
        'ncpus': [float(ncpus)],
        'ngpus': [float(ngpus)],
        'walltime_seconds': [float(walltime_seconds)],
        'queue_encoded': [label_encoders['queue'].transform([queue_val])[0]],
        'arch_encoded': [label_encoders['arch'].transform([arch_val])[0]]
    })
    
    # Apply feature weights
    features_weighted = apply_feature_weights(features)
    features_scaled = scaler.transform(features_weighted)
    
    # Stage 1: Classify
    class_probs = classifier.predict_proba(features_weighted)[0]
    predicted_class = np.argmax(class_probs)
    max_prob = class_probs[predicted_class]
    confidence = get_confidence_level(max_prob)
    
    # Stage 2: Regress using appropriate model
    reg_info = regressors[predicted_class]
    if reg_info['scaled']:
        X_input = features_scaled
    else:
        X_input = features_weighted
    
    pred_log = reg_info['model'].predict(X_input)[0]
    pred_seconds = max(0, np.expm1(pred_log))
    
    # Clip to class bounds (soft constraint)
    lower, upper = CLASS_THRESHOLDS[predicted_class]
    pred_seconds_clipped = np.clip(pred_seconds, lower, upper)
    
    return {
        'prediction_seconds': pred_seconds_clipped,
        'prediction_formatted': seconds_to_time(pred_seconds_clipped),
        'prediction_hours': pred_seconds_clipped / 3600,
        'predicted_class': predicted_class,
        'predicted_class_name': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'confidence_score': float(max_prob),
        'class_probabilities': {
            CLASS_NAMES[i]: float(p) for i, p in enumerate(class_probs)
        },
        'model_used': reg_info['type'],
        'input': {
            'queue': queue,
            'ncpus': ncpus,
            'ngpus': ngpus,
            'arch': arch,
            'walltime': walltime_str
        }
    }

# =============================================================================
# Main
# =============================================================================

def main():
    """Train models and run example predictions."""
    
    # Train
    models, label_encoders, scaler, metadata = train_models()
    
    # Example predictions
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    test_cases = [
        ('standard', 4, 0, 'icelake', '00:30:00'),
        ('standard', 4, 1, 'icelake', '05:00:00'),
        ('standard', 32, 0, 'skylake', '24:00:00'),
        ('scai_q', 32, 8, 'amdepyc', '48:00:00'),
        ('standard', 128, 4, 'icelake', '168:00:00'),
        ('standard', 1, 0, 'any', '01:00:00'),
    ]
    
    for queue, ncpus, ngpus, arch, walltime in test_cases:
        result = predict(models, label_encoders, scaler, queue, ncpus, ngpus, arch, walltime)
        
        print(f"\nJob: queue={queue}, ncpus={ncpus}, ngpus={ngpus}, arch={arch}, walltime={walltime}")
        print(f"  Predicted Class: {result['predicted_class_name']}")
        print(f"  Estimated Wait:  {result['prediction_formatted']} ({result['prediction_hours']:.2f} hours)")
        print(f"  Confidence:      {result['confidence']} ({result['confidence_score']:.1%})")
        print(f"  Model Used:      {result['model_used']}")
        print(f"  Class Probs:     {result['class_probabilities']}")
    
    return models, label_encoders, scaler

if __name__ == "__main__":
    main()
