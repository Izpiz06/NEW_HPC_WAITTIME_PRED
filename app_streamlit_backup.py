"""
IITD HPC Job Wait Time Predictor
================================
Production-ready Hybrid RF-LSTM Prediction System.
Predicts (1) Actual Runtime -> (2) Wait Time
Supports: RF-only, LSTM-only, Weighted, and Stacked predictions.

Version: 5.2.0 (Schedule-Agnostic + No-RAM Model + Admin Panel)
Author: IIT Delhi HPC Facility

Changes in v5.2:
- Added Admin Panel with login/registration
- Admin can add new architectures and queues
- Admin can view usage logs and training logs

Changes in v5.1:
- REMOVED timestamp features (Hour, Day, Month, Is_Weekend, Is_Business_Hours)
- Model now relies on Queue_Depth for system state (schedule-agnostic)
- Works correctly on holidays, weekends, and irregular schedules

Changes in v5.0:
- RAM is now OPTIONAL in UI (defaults to off)
- Prediction engine ignores RAM for node calculations
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import logging
import os
import sys
import sqlite3
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any, List
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# =============================================================================
# 0. DATABASE & ADMIN MANAGEMENT
# =============================================================================

DB_PATH = Path("database/HPC_admindb")

def init_database():
    """Initialize the admin database with required tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Users table (with status for approval workflow)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            role TEXT DEFAULT 'admin',
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            approved_by TEXT,
            approved_at TIMESTAMP
        )
    """)
    
    # Migration: Add new columns to existing users table if they don't exist
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'status' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN status TEXT DEFAULT 'approved'")
    if 'approved_by' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN approved_by TEXT")
    if 'approved_at' not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN approved_at TIMESTAMP")
    
    # Email configuration table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_config (
            id INTEGER PRIMARY KEY,
            smtp_server TEXT DEFAULT 'smtp.gmail.com',
            smtp_port INTEGER DEFAULT 587,
            sender_email TEXT,
            sender_password TEXT,
            admin_notification_email TEXT,
            is_configured INTEGER DEFAULT 0
        )
    """)
    
    # Insert default email config row if not exists
    cursor.execute("INSERT OR IGNORE INTO email_config (id) VALUES (1)")
    
    # Architectures table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS architectures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            cores_per_node INTEGER DEFAULT 48,
            ram_per_node INTEGER DEFAULT 128,
            gpus_per_node INTEGER DEFAULT 2,
            description TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        )
    """)
    
    # Queues table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS queues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            soft_limit_hours INTEGER DEFAULT 168,
            max_cores INTEGER DEFAULT 256,
            max_gpus INTEGER DEFAULT 8,
            max_memory_gb INTEGER DEFAULT 1024,
            priority INTEGER DEFAULT 1,
            description TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        )
    """)
    
    # Usage logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            action TEXT NOT NULL,
            user TEXT,
            details TEXT,
            ip_address TEXT,
            queue_used TEXT,
            architecture_used TEXT,
            cores_requested INTEGER,
            gpus_requested INTEGER,
            predicted_wait_seconds REAL
        )
    """)
    
    # Training logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT NOT NULL,
            model_version TEXT,
            training_duration_seconds REAL,
            samples_used INTEGER,
            metrics TEXT,
            status TEXT,
            error_message TEXT,
            triggered_by TEXT
        )
    """)
    
    # Insert default architectures if not exist
    default_archs = [
        ('any', 48, 128, 2, 'Default architecture'),
        ('skylake', 40, 96, 2, 'Intel Skylake nodes'),
        ('icelake', 64, 256, 2, 'Intel Icelake nodes')
    ]
    for arch in default_archs:
        cursor.execute("""
            INSERT OR IGNORE INTO architectures (name, cores_per_node, ram_per_node, gpus_per_node, description)
            VALUES (?, ?, ?, ?, ?)
        """, arch)
    
    # Insert default queues if not exist
    default_queues = [
        ('standard', 168, 256, 8, 1024, 1, 'Standard priority queue'),
        ('high', 24, 128, 4, 512, 2, 'High priority queue')
    ]
    for q in default_queues:
        cursor.execute("""
            INSERT OR IGNORE INTO queues (name, soft_limit_hours, max_cores, max_gpus, max_memory_gb, priority, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, q)
    
    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def get_email_config() -> Dict:
    """Get email configuration."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM email_config WHERE id = 1")
    columns = [desc[0] for desc in cursor.description]
    row = cursor.fetchone()
    conn.close()
    return dict(zip(columns, row)) if row else {}

def save_email_config(smtp_server: str, smtp_port: int, sender_email: str, 
                      sender_password: str, admin_email: str) -> Tuple[bool, str]:
    """Save email configuration."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE email_config SET 
                smtp_server = ?, smtp_port = ?, sender_email = ?,
                sender_password = ?, admin_notification_email = ?, is_configured = 1
            WHERE id = 1
        """, (smtp_server, smtp_port, sender_email, sender_password, admin_email))
        conn.commit()
        conn.close()
        return True, "Email configuration saved!"
    except Exception as e:
        return False, f"Failed to save config: {str(e)}"

def send_email(to_email: str, subject: str, body: str) -> Tuple[bool, str]:
    """Send an email notification."""
    config = get_email_config()
    
    if not config.get('is_configured'):
        return False, "Email not configured. Please configure SMTP settings in Admin Panel."
    
    try:
        msg = MIMEMultipart()
        msg['From'] = config['sender_email']
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))
        
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['sender_email'], config['sender_password'])
        server.send_message(msg)
        server.quit()
        
        return True, "Email sent successfully!"
    except Exception as e:
        logging.error(f"Email send failed: {e}")
        return False, f"Email failed: {str(e)}"

def send_approval_request_email(username: str, user_email: str) -> Tuple[bool, str]:
    """Send email to admin about new registration request."""
    config = get_email_config()
    admin_email = config.get('admin_notification_email')
    
    if not admin_email:
        return False, "Admin notification email not configured."
    
    subject = f"[ALERT] New Admin Registration Request - {username}"
    body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; padding: 20px;">
        <h2 style="color: #E31E24;">New Admin Registration Request</h2>
        <p>A new user has requested admin access to the HPC Wait Time Predictor system.</p>
        <table style="border-collapse: collapse; margin: 20px 0;">
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Username:</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{username}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Email:</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{user_email or 'Not provided'}</td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Requested At:</td>
                <td style="padding: 10px; border: 1px solid #ddd;">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</td>
            </tr>
        </table>
        <p>Please login to the Admin Panel to approve or reject this request.</p>
        <p style="color: #666; font-size: 12px;">— IIT Delhi HPC Facility</p>
    </body>
    </html>
    """
    
    return send_email(admin_email, subject, body)

def send_approval_notification(username: str, user_email: str, approved: bool) -> Tuple[bool, str]:
    """Send email to user about their approval status."""
    if not user_email:
        return False, "User email not provided."
    
    if approved:
        subject = "Admin Access Approved - IITD HPC Predictor"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #22C55E;">Access Approved!</h2>
            <p>Dear {username},</p>
            <p>Your admin access request for the IIT Delhi HPC Wait Time Predictor has been <strong>approved</strong>.</p>
            <p>You can now login with your credentials to access the Admin Panel.</p>
            <p style="color: #666; font-size: 12px;">— IIT Delhi HPC Facility</p>
        </body>
        </html>
        """
    else:
        subject = "Admin Access Rejected - IITD HPC Predictor"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #EF4444;">Access Rejected</h2>
            <p>Dear {username},</p>
            <p>Your admin access request for the IIT Delhi HPC Wait Time Predictor has been <strong>rejected</strong>.</p>
            <p>If you believe this is an error, please contact the system administrator.</p>
            <p style="color: #666; font-size: 12px;">— IIT Delhi HPC Facility</p>
        </body>
        </html>
        """
    
    return send_email(user_email, subject, body)

def verify_user(username: str, password: str) -> Tuple[bool, str]:
    """Verify user credentials and check approval status."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash, status FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return False, "Invalid username or password"
    
    if result[0] != hash_password(password):
        return False, "Invalid username or password"
    
    if result[1] == 'pending':
        return False, "Your account is pending approval. Please wait for admin approval."
    
    if result[1] == 'rejected':
        return False, "Your account has been rejected. Please contact the administrator."
    
    # Update last login
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET last_login = ? WHERE username = ?", 
                  (datetime.now(), username))
    conn.commit()
    conn.close()
    return True, "Login successful"

def register_user(username: str, password: str, email: str = None) -> Tuple[bool, str]:
    """Register a new admin user (pending approval)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if this is the first user (auto-approve first admin)
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        
        status = 'approved' if user_count == 0 else 'pending'
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, email, status)
            VALUES (?, ?, ?, ?)
        """, (username, hash_password(password), email, status))
        conn.commit()
        conn.close()
        
        if status == 'approved':
            return True, "First admin registered and auto-approved! You can login now."
        
        # Send email notification to admin
        email_sent, email_msg = send_approval_request_email(username, email)
        
        if email_sent:
            return True, "Registration submitted! An approval request has been sent to the administrator. You'll receive an email once approved."
        else:
            return True, f"Registration submitted! Waiting for admin approval. (Note: Email notification failed - {email_msg})"
            
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def get_pending_users() -> List[Dict]:
    """Get all pending user registrations."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, created_at FROM users WHERE status = 'pending' ORDER BY created_at DESC")
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def approve_user(user_id: int, approved_by: str) -> Tuple[bool, str]:
    """Approve a pending user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get user info for email
        cursor.execute("SELECT username, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "User not found"
        
        username, user_email = user
        
        cursor.execute("""
            UPDATE users SET status = 'approved', approved_by = ?, approved_at = ?
            WHERE id = ?
        """, (approved_by, datetime.now(), user_id))
        conn.commit()
        conn.close()
        
        # Send approval notification email
        send_approval_notification(username, user_email, approved=True)
        
        log_usage("user_approved", approved_by, f"Approved user: {username}")
        return True, f"User '{username}' has been approved!"
    except Exception as e:
        return False, f"Failed to approve: {str(e)}"

def reject_user(user_id: int, rejected_by: str) -> Tuple[bool, str]:
    """Reject a pending user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get user info for email
        cursor.execute("SELECT username, email FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, "User not found"
        
        username, user_email = user
        
        cursor.execute("UPDATE users SET status = 'rejected' WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        
        # Send rejection notification email
        send_approval_notification(username, user_email, approved=False)
        
        log_usage("user_rejected", rejected_by, f"Rejected user: {username}")
        return True, f"User '{username}' has been rejected."
    except Exception as e:
        return False, f"Failed to reject: {str(e)}"

def get_all_users() -> List[Dict]:
    """Get all users."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, role, status, created_at, last_login, approved_by FROM users ORDER BY created_at DESC")
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_architectures() -> List[Dict]:
    """Get all architectures from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM architectures WHERE is_active = 1 ORDER BY name")
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_queues() -> List[Dict]:
    """Get all queues from database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM queues WHERE is_active = 1 ORDER BY priority DESC, name")
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def add_architecture(name: str, cores: int, ram: int, gpus: int, description: str, created_by: str) -> Tuple[bool, str]:
    """Add a new architecture."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO architectures (name, cores_per_node, ram_per_node, gpus_per_node, description, created_by)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (name.lower(), cores, ram, gpus, description, created_by))
        conn.commit()
        conn.close()
        log_usage("add_architecture", created_by, f"Added architecture: {name}")
        return True, f"Architecture '{name}' added successfully!"
    except sqlite3.IntegrityError:
        return False, f"Architecture '{name}' already exists!"
    except Exception as e:
        return False, f"Failed to add architecture: {str(e)}"

def add_queue(name: str, soft_limit: int, max_cores: int, max_gpus: int, 
              max_memory: int, priority: int, description: str, created_by: str) -> Tuple[bool, str]:
    """Add a new queue."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO queues (name, soft_limit_hours, max_cores, max_gpus, max_memory_gb, priority, description, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (name.lower(), soft_limit, max_cores, max_gpus, max_memory, priority, description, created_by))
        conn.commit()
        conn.close()
        log_usage("add_queue", created_by, f"Added queue: {name}")
        return True, f"Queue '{name}' added successfully!"
    except sqlite3.IntegrityError:
        return False, f"Queue '{name}' already exists!"
    except Exception as e:
        return False, f"Failed to add queue: {str(e)}"

def delete_architecture(arch_id: int, deleted_by: str) -> Tuple[bool, str]:
    """Soft delete an architecture."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE architectures SET is_active = 0 WHERE id = ?", (arch_id,))
        conn.commit()
        conn.close()
        log_usage("delete_architecture", deleted_by, f"Deleted architecture ID: {arch_id}")
        return True, "Architecture deleted successfully!"
    except Exception as e:
        return False, f"Failed to delete: {str(e)}"

def delete_queue(queue_id: int, deleted_by: str) -> Tuple[bool, str]:
    """Soft delete a queue."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("UPDATE queues SET is_active = 0 WHERE id = ?", (queue_id,))
        conn.commit()
        conn.close()
        log_usage("delete_queue", deleted_by, f"Deleted queue ID: {queue_id}")
        return True, "Queue deleted successfully!"
    except Exception as e:
        return False, f"Failed to delete: {str(e)}"

def log_usage(action: str, user: str = None, details: str = None, 
              queue: str = None, arch: str = None, cores: int = None, 
              gpus: int = None, predicted_wait: float = None):
    """Log a usage event."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO usage_logs (action, user, details, queue_used, architecture_used, 
                                   cores_requested, gpus_requested, predicted_wait_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (action, user, details, queue, arch, cores, gpus, predicted_wait))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to log usage: {e}")

def log_training(model_type: str, version: str, duration: float, samples: int, 
                 metrics: str, status: str, error: str = None, triggered_by: str = None):
    """Log a training event."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO training_logs (model_type, model_version, training_duration_seconds, 
                                       samples_used, metrics, status, error_message, triggered_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_type, version, duration, samples, metrics, status, error, triggered_by))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to log training: {e}")

def get_usage_logs(limit: int = 100) -> List[Dict]:
    """Get usage logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM usage_logs ORDER BY timestamp DESC LIMIT {limit}")
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_training_logs(limit: int = 50) -> List[Dict]:
    """Get training logs."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM training_logs ORDER BY timestamp DESC LIMIT {limit}")
    columns = [desc[0] for desc in cursor.description]
    results = [dict(zip(columns, row)) for row in cursor.fetchall()]
    conn.close()
    return results

def get_usage_stats() -> Dict:
    """Get usage statistics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {}
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM usage_logs WHERE action = 'prediction'")
    stats['total_predictions'] = cursor.fetchone()[0]
    
    # Predictions today
    cursor.execute("""
        SELECT COUNT(*) FROM usage_logs 
        WHERE action = 'prediction' AND DATE(timestamp) = DATE('now')
    """)
    stats['predictions_today'] = cursor.fetchone()[0]
    
    # Most used queue
    cursor.execute("""
        SELECT queue_used, COUNT(*) as cnt FROM usage_logs 
        WHERE queue_used IS NOT NULL GROUP BY queue_used ORDER BY cnt DESC LIMIT 1
    """)
    result = cursor.fetchone()
    stats['most_used_queue'] = result[0] if result else 'N/A'
    
    # Most used architecture
    cursor.execute("""
        SELECT architecture_used, COUNT(*) as cnt FROM usage_logs 
        WHERE architecture_used IS NOT NULL GROUP BY architecture_used ORDER BY cnt DESC LIMIT 1
    """)
    result = cursor.fetchone()
    stats['most_used_arch'] = result[0] if result else 'N/A'
    
    # Average predicted wait time
    cursor.execute("""
        SELECT AVG(predicted_wait_seconds) FROM usage_logs 
        WHERE predicted_wait_seconds IS NOT NULL
    """)
    result = cursor.fetchone()
    stats['avg_predicted_wait'] = result[0] if result and result[0] else 0
    
    conn.close()
    return stats

# Initialize database on import
init_database()

# TensorFlow import (optional for hybrid model)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# =============================================================================
# 1. MODEL CLASS DEFINITIONS (MUST MATCH TRAIN.PY EXACTLY)
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration used during training."""
    model_version: str = "3.2.1"
    data_path: str = 'clean_hpc_data.csv'
    wait_threshold: int = 60
    n_folds: int = 5
    random_state: int = 42
    max_wait_seconds: int = 604800 
    runtime_n_estimators: int = 100
    runtime_max_depth: int = 15
    clf_n_estimators: int = 150
    clf_max_depth: int = 12
    clf_min_samples_split: int = 10
    reg_n_estimators: int = 150
    reg_max_depth: int = 25
    reg_min_samples_split: int = 5
    runtime_model_path: str = 'runtime_predictor.pkl'
    wait_model_path: str = 'final_hurdle_model.pkl'
    metadata_path: str = 'model_metadata.pkl'
    metrics_path: str = 'model_metrics.csv'
    feature_importance_path: str = 'feature_importance.csv'

class RuntimeModel:
    """Predicts ACTUAL runtime (Model A)."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_names:
            for c in set(self.feature_names) - set(X.columns): X[c] = 0
            X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return np.expm1(self.model.predict(X_scaled))

class HurdleModel:
    """Predicts Wait Time (Model B)."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def predict(self, X: pd.DataFrame, decision_threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        if self.feature_names:
            for c in set(self.feature_names) - set(X.columns): X[c] = 0
            X = X[self.feature_names]
        
        X_scaled = self.scaler.transform(X)
        prob_waiting = self.classifier.predict_proba(X_scaled)[:, 1]
        will_wait = (prob_waiting > (1.0 - decision_threshold)).astype(int)
        
        y_pred = np.zeros(len(X))
        if self.regressor:
            y_log_pred = self.regressor.predict(X_scaled)
            y_time_pred = np.maximum(np.expm1(y_log_pred), 0.0)
            mask_wait = will_wait == 1
            y_pred[mask_wait] = y_time_pred[mask_wait]
            
        return y_pred, prob_waiting

# -----------------------------------------------------------------------------
# HYBRID MODEL CLASSES (must match retrain_hybrid.py for joblib loading)
# -----------------------------------------------------------------------------

@dataclass
class HybridModelConfig:
    """Configuration for Hybrid RF-LSTM Model - must match retrain_hybrid.py."""
    model_version: str = "5.1.0"  # Schedule-Agnostic + No-RAM Model
    data_path: str = 'clean_hpc_data.csv'
    wait_threshold: int = 60
    n_folds: int = 3
    random_state: int = 42
    max_wait_seconds: int = 604800  # 7 days
    
    # RF (Static Features) Hyperparameters
    rf_n_estimators: int = 150
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    
    # LSTM (Temporal) Hyperparameters
    lstm_sequence_length: int = 50
    lstm_units: int = 64
    lstm_dropout: float = 0.2
    lstm_epochs: int = 50
    lstm_batch_size: int = 512
    lstm_patience: int = 10
    
    # Combination Weights
    rf_weight: float = 0.7
    lstm_weight: float = 0.3
    use_stacking: bool = True
    
    # Output paths
    rf_model_path: str = 'hybrid_rf_model.pkl'
    lstm_model_path: str = 'hybrid_lstm_model.keras'
    stacked_model_path: str = 'hybrid_stacked_model.pkl'
    metadata_path: str = 'hybrid_model_metadata.pkl'
    scaler_path: str = 'hybrid_scalers.pkl'
    metrics_path: str = 'hybrid_rf_metrices.csv'
    log_file: str = 'hybrid_training.log'


class StaticRFModel:
    """
    Random Forest for Static Job Specifications - must match retrain_hybrid.py.
    Required for joblib to load the trained model.
    
    v5.1: Schedule-Agnostic - REMOVED timestamp features
    """
    
    STATIC_FEATURES = [
        'NCORES', 'NGPUS', 'Walltime_Seconds', 'Req_Mem_GB', 'Req_Nodes',
        'Total_CPUs', 'Resource_Intensity', 'Is_High_Priority', 'Log_Req_Walltime',
        'CPU_Memory_Ratio', 'Memory_CPU_Product', 'Nodes_Required',
        'RAM_Per_Node', 'Core_Packing_Difficulty', 'Fragmentation_Risk',
        # v5.1: REMOVED - Submission_Hour, Submission_Day, Is_Weekend, Is_Business_Hours
        # System state features (schedule-agnostic)
        'Queue_Depth', 'Queue_Pressure'
    ]
    
    def __init__(self, config: HybridModelConfig):
        self.config = config
        self.model = RandomForestRegressor(
            n_estimators=config.rf_n_estimators,
            max_depth=config.rf_max_depth,
            min_samples_split=config.rf_min_samples_split,
            random_state=config.random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract static features available in the data."""
        available = [f for f in self.STATIC_FEATURES if f in df.columns]
        # Add queue columns
        for col in df.columns:
            if col.startswith('Q_') or col.startswith('Arch_'):
                available.append(col)
        self.feature_names = list(set(available))
        return pd.DataFrame(df[self.feature_names]).copy()
    
    def fit(self, df: pd.DataFrame, y: np.ndarray):
        """Train on static job features."""
        X = self.get_features(df)
        X_scaled = self.scaler.fit_transform(X)
        y_log = np.log1p(y)
        self.model.fit(X_scaled, y_log)
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict wait time from static features."""
        X = self.get_features(df)
        X_scaled = self.scaler.transform(X)
        y_log_pred = self.model.predict(X_scaled)
        return np.expm1(y_log_pred)
    
    def get_feature_importance(self) -> pd.DataFrame:
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

# =============================================================================
# 2. APP CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class AppConfig:
    APP_NAME: str = "IITD HPC Predictor"
    APP_VERSION: str = "5.2.0"  # Schedule-Agnostic + No-RAM Model + Admin Panel
    DEBUG: bool = False
    RUNTIME_MODEL_FILE: Path = Path("runtime_predictor.pkl")
    WAIT_MODEL_FILE: Path = Path("final_hurdle_model.pkl")
    METADATA_FILE: Path = Path("model_metadata.pkl")
    # Hybrid Model Files
    HYBRID_RF_MODEL_FILE: Path = Path("hybrid_rf_model.pkl")
    HYBRID_LSTM_MODEL_FILE: Path = Path("hybrid_lstm_model.keras")
    HYBRID_STACKED_MODEL_FILE: Path = Path("hybrid_stacked_model.pkl")
    HYBRID_METADATA_FILE: Path = Path("hybrid_model_metadata.pkl")
    HYBRID_SCALER_FILE: Path = Path("hybrid_scalers.pkl")

@dataclass(frozen=True)
class HPCConfig:
    CORES_PER_NODE: int = 48
    MAX_CORES: int = 2000
    MAX_GPUS: int = 32
    MAX_MEMORY_GB: int = 1024


@dataclass(frozen=True)
class QueueConfig:
    name: str
    soft_limit_hours: int

QUEUE_CONFIGS = {
    "standard": QueueConfig("standard", 168), # 7 Days
    "high": QueueConfig("high", 24),
}

# =============================================================================
# 3. SERVICES (Feature Engineering & Prediction)
# =============================================================================

class FeatureEngineer:
    def __init__(self, hpc_config):
        self.hpc = hpc_config

    def create_features(self, job, traffic: float, use_ram: bool = False) -> pd.DataFrame:
        """
        Creates the raw feature set from user input.
        
        Args:
            job: JobRequest object
            traffic: Current traffic congestion level
            use_ram: Whether to include RAM in calculations (v5.0: defaults to False)
        """
        # 1. Base Calculations - v5.0: Ignore RAM for intensity
        intensity = ((job.n_cores / self.hpc.MAX_CORES) + (job.n_gpus / self.hpc.MAX_GPUS))
        
        # v5.0: Node calculation based on CPUs only (not RAM) unless use_ram=True
        if getattr(job, "manual_nodes", None):
            nodes = int(job.manual_nodes)
        else:
            nodes = max(1, int(np.ceil(job.n_cores / self.hpc.CORES_PER_NODE)))

        # 2. Basic Dict - v5.1: Schedule-agnostic (no timestamp features)
        # v5.0: RAM is optional (defaults to 0 if not used)
        ram_value = job.ram_gb if use_ram else 0
        data = {
            "NCORES": job.n_cores,
            "NGPUS": job.n_gpus,
            "Walltime_Seconds": job.walltime_seconds,
            "Req_Mem_GB": ram_value,  # v5.0: Only include if use_ram=True
            "Req_Nodes": nodes,
            "Total_CPUs": job.n_cores,
            "Resource_Intensity": intensity,
            "Traffic_Congestion": traffic,
            "Traffic_Congestion_EWM": traffic, # Simulating EWM as current for single prediction
            # v5.1: REMOVED timestamp features - model is now schedule-agnostic
            # Old: Submission_Hour, Submission_Day, Submission_Month, Is_Weekend, Is_Business_Hours
            # New: Relies entirely on Queue_Depth/Traffic for system state
        }

        # 3. One-Hot Encoding (Manual)
        data[f"Q_{job.queue}"] = 1
        if job.architecture != "any":
            data[f"Arch_{job.architecture}"] = 1
            
        # 4. Complex Interaction Features (Must match train.py)
        is_high = int(job.queue == "high")
        data["Is_High_Priority"] = is_high
        data["Effective_Traffic"] = traffic * (1 - is_high)
        data["Queue_Pressure"] = traffic * intensity
        data["Log_Req_Walltime"] = np.log1p(job.walltime_seconds)
        data["Memory_CPU_Product"] = ram_value * job.n_cores  # v5.0: Uses ram_value
        # ============================
        # MULTI-NODE / FRAGMENTATION FEATURES - v5.0: Based on CPUs only
        # ============================
        CORES_PER_NODE = 48
        RAM_PER_NODE = 128

        nodes_core = np.ceil(job.n_cores / CORES_PER_NODE)
        
        # v5.0: Node calculation ignores RAM for consistency with No-RAM model
        if use_ram and job.ram_gb > 0:
            nodes_ram = np.ceil(job.ram_gb / RAM_PER_NODE)
            nodes_required = max(nodes_core, nodes_ram)
            ram_per_node = job.ram_gb / max(nodes_required, 1)
            cpu_mem_ratio = job.n_cores / (job.ram_gb + 1)
        else:
            nodes_required = nodes_core
            ram_per_node = 0
            cpu_mem_ratio = job.n_cores  # No RAM division

        data["Nodes_Required"] = nodes_required
        data["RAM_Per_Node"] = ram_per_node

        data["Core_Packing_Difficulty"] = nodes_required * data["Resource_Intensity"]
        data["Fragmentation_Risk"] = nodes_required * np.log1p(job.walltime_seconds)

        # Safe Division - v5.0: Adjusted for No-RAM
        data["CPU_Memory_Ratio"] = cpu_mem_ratio
        data["Resource_Per_Traffic"] = intensity / (traffic + 1)

        return pd.DataFrame([data])

# =============================================================================
# 3. DATA STRUCTURES (must be defined before prediction services)
# =============================================================================

@dataclass
class JobRequest:
    queue: str
    n_cores: int
    n_gpus: int
    ram_gb: int
    walltime_seconds: int
    architecture: str
    submission_datetime: datetime
    manual_nodes: Optional[int] = None

@dataclass
class PredictionResult:
    expected_wait_seconds: float
    probability_immediate_start: float
    estimated_start_time: datetime
    predicted_runtime: float
    # Hybrid model outputs
    rf_prediction: Optional[float] = None
    lstm_prediction: Optional[float] = None
    confidence: Optional[float] = None
    prediction_method: str = "legacy"
    
    @property
    def status(self) -> str:
        if self.probability_immediate_start > 0.8: return "immediate"
        if self.expected_wait_seconds < 3600: return "short"
        if self.expected_wait_seconds < 86400: return "moderate"
        return "extended"
    
    @property
    def confidence_level(self) -> str:
        """Get confidence level description."""
        if self.confidence is None:
            return "N/A"
        if self.confidence > 0.8:
            return "High"
        elif self.confidence > 0.5:
            return "Medium"
        else:
            return "Low"
    
    @property
    def agreement_analysis(self) -> str:
        """Analyze RF vs LSTM agreement."""
        if self.rf_prediction is None or self.lstm_prediction is None:
            return ""
        
        rf_t = self.format_time(self.rf_prediction)
        lstm_t = self.format_time(self.lstm_prediction)
        
        diff = abs(self.rf_prediction - self.lstm_prediction)
        avg = (self.rf_prediction + self.lstm_prediction) / 2
        
        if avg > 0 and diff / avg < 0.2:  # Within 20%
            return f"High agreement: Both models predict ~{rf_t}"
        elif self.rf_prediction < self.lstm_prediction:
            return f"Note: Job specs suggests {rf_t}, but cluster is backlogged ({lstm_t})"
        else:
            return f"Note: Large job ({rf_t}), but cluster is less busy ({lstm_t})"

    def format_time(self, s):
        if s < 60: return f"{int(s)}s"
        elif s < 3600: return f"{int(s//60)}m"
        else: return f"{s/3600:.1f}h"

# =============================================================================
# 4. PREDICTION SERVICES
# =============================================================================

class PredictionService:
    def __init__(self, runtime_model, wait_model, metadata):
        self.runtime_model = runtime_model
        self.wait_model = wait_model
        self.engineer = FeatureEngineer(HPCConfig())
        self.has_runtime_model = metadata.get('has_runtime_model', False)
        
    def predict(self, job_request, traffic_congestion: float, use_ram: bool = False) -> Any:
        # 1. Generate Base Features - v5.0: Pass use_ram flag
        df_features = self.engineer.create_features(job_request, traffic_congestion, use_ram=use_ram)
        
        # 2. TANDEM STEP 1: Get Predicted Runtime
        if self.has_runtime_model and self.runtime_model:
            # Predict using Model A
            pred_runtime = self.runtime_model.predict(df_features)[0]
        else:
            # Smart Fallback: Use Requested Walltime
            pred_runtime = job_request.walltime_seconds
            
        # Add to features for Model B
        df_features['Predicted_Runtime'] = pred_runtime
        
        # 3. TANDEM STEP 2: Predict Wait Time
        # Model B expects 'Predicted_Runtime' to be present
        wait_seconds, prob_waiting = self.wait_model.predict(df_features, decision_threshold=0.75)
        
        return PredictionResult(
            expected_wait_seconds=float(wait_seconds[0]),
            probability_immediate_start=1.0 - float(prob_waiting[0]),
            estimated_start_time=job_request.submission_datetime + timedelta(seconds=float(wait_seconds[0])),
            predicted_runtime=pred_runtime,
            prediction_method="legacy"
        )


class HybridPredictionService:
    """
    Hybrid RF-LSTM Prediction Service (v5.1).
    Uses REAL historical data from clean_hpc_data1.csv for LSTM context.
    Provides both RF (static) and LSTM (temporal) predictions with comparison.
    """
    
    # Path to historical data for LSTM context
    LSTM_DATA_FILE = Path("clean_hpc_data1.csv")
    HISTORY_ROWS = 500  # Load last N rows for context
    
    def __init__(self, rf_model, lstm_model, stacked_model, scalers, metadata, config):
        self.rf_model = rf_model
        self.lstm_model = lstm_model
        self.stacked_model = stacked_model
        self.scalers = scalers
        self.metadata = metadata
        self.config = config
        self.engineer = FeatureEngineer(HPCConfig())
        
        # Hybrid weights from config
        self.rf_weight = metadata.get('config', {}).get('rf_weight', 0.4)
        self.lstm_weight = metadata.get('config', {}).get('lstm_weight', 0.6)
        
        # Pre-load and cache historical data for LSTM
        self._history_cache = None
        self._history_timestamp = None
    
    def _load_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Load the last N rows from clean_hpc_data1.csv for LSTM context.
        Caches data and refreshes every 5 minutes to capture new jobs.
        """
        import time
        current_time = time.time()
        
        # Use cached data if fresh (less than 5 minutes old)
        if (self._history_cache is not None and 
            self._history_timestamp is not None and 
            current_time - self._history_timestamp < 300):  # 5 min cache
            return self._history_cache
        
        try:
            if not self.LSTM_DATA_FILE.exists():
                logging.warning(f"LSTM data file not found: {self.LSTM_DATA_FILE}")
                return None
            
            # Read only the tail of the CSV for efficiency
            # First, count total lines
            with open(self.LSTM_DATA_FILE, 'r') as f:
                total_lines = sum(1 for _ in f)
            
            # Skip to get last HISTORY_ROWS
            skip_rows = max(0, total_lines - self.HISTORY_ROWS - 1)  # -1 for header
            
            df = pd.read_csv(self.LSTM_DATA_FILE, skiprows=range(1, skip_rows + 1))
            
            logging.info(f"Loaded {len(df)} historical rows from {self.LSTM_DATA_FILE}")
            
            # Cache the data
            self._history_cache = df
            self._history_timestamp = current_time
            
            return df
            
        except Exception as e:
            logging.error(f"Failed to load historical data: {e}")
            return None
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal/rolling features from historical data.
        Must match the features used during LSTM training.
        """
        df = df.copy()
        
        # Sort by JobID for proper temporal ordering
        if 'JobID' in df.columns:
            df['_sort_key'] = df['JobID'].astype(str).str.extract(r'(\d+)').astype(float)
            df = df.sort_values('_sort_key').reset_index(drop=True)
            df = df.drop(columns=['_sort_key'])
        
        # Calculate Queue_Depth if not present
        if 'Queue_Depth' not in df.columns:
            # Simple approximation: cumulative jobs in queue
            n_jobs = len(df)
            df['Queue_Depth'] = np.arange(1, n_jobs + 1) % 100  # Cyclic pattern
        
        # Rolling statistics (must match training)
        window_sizes = [5, 10, 20, 50]
        
        for w in window_sizes:
            # Rolling average wait time
            df[f'Rolling_Wait_Mean_{w}'] = df['WaitTime_Seconds'].shift(1).rolling(
                window=w, min_periods=1).mean()
            
            # Rolling std
            df[f'Rolling_Wait_Std_{w}'] = df['WaitTime_Seconds'].shift(1).rolling(
                window=w, min_periods=1).std().fillna(0)
            
            # Rolling max
            df[f'Rolling_Wait_Max_{w}'] = df['WaitTime_Seconds'].shift(1).rolling(
                window=w, min_periods=1).max()
            
            # Arrival rate
            df[f'Arrival_Rate_{w}'] = df['Queue_Depth'].rolling(
                window=w, min_periods=1).mean()
            
            # Resource demand trend
            df[f'Resource_Demand_{w}'] = df['Resource_Intensity'].rolling(
                window=w, min_periods=1).sum()
        
        # Exponential weighted moving averages
        df['EWMA_Wait'] = df['WaitTime_Seconds'].shift(1).ewm(span=20).mean()
        df['EWMA_Resource'] = df['Resource_Intensity'].ewm(span=20).mean()
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def _get_rf_prediction(self, df_features: pd.DataFrame) -> float:
        """Get RF prediction from static features."""
        if self.rf_model is None:
            return None
        
        # Get static features
        rf_features = self.scalers.get('rf_features', [])
        available = [f for f in rf_features if f in df_features.columns]
        
        # Add missing features
        for f in rf_features:
            if f not in df_features.columns:
                df_features[f] = 0
        
        X = df_features[rf_features].copy()
        X_scaled = self.scalers['rf_scaler'].transform(X)
        y_log = self.rf_model.model.predict(X_scaled)
        return float(np.expm1(y_log[0]))
    
    def _get_lstm_prediction(self, job_request, live_queue_depth: float) -> Optional[float]:
        """
        Get LSTM prediction using REAL historical data from clean_hpc_data1.csv.
        
        v5.1: Uses actual cluster history instead of synthetic data.
        The last row of the sequence is "stitched" with live queue depth.
        """
        if self.lstm_model is None or not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            # Load real historical data
            df_history = self._load_historical_data()
            if df_history is None or len(df_history) < 50:
                logging.warning("Insufficient historical data for LSTM")
                return None
            
            # Create temporal features from real data
            df_temporal = self._create_temporal_features(df_history)
            
            # Get LSTM feature columns
            lstm_features = self.scalers.get('lstm_features', [])
            if not lstm_features:
                logging.warning("No LSTM features defined in scaler")
                return None
            
            # Ensure all required features exist
            for feat in lstm_features:
                if feat not in df_temporal.columns:
                    df_temporal[feat] = 0
            
            # Get sequence length from config
            seq_len = self.metadata.get('config', {}).get('lstm_sequence_length', 50)
            
            # Take the last seq_len rows as the sequence
            if len(df_temporal) < seq_len:
                # Pad with zeros if not enough history
                pad_len = seq_len - len(df_temporal)
                df_padded = pd.DataFrame(0, index=range(pad_len), columns=lstm_features)
                df_seq = pd.concat([df_padded, df_temporal[lstm_features].tail(len(df_temporal))], ignore_index=True)
            else:
                df_seq = df_temporal[lstm_features].tail(seq_len).reset_index(drop=True)
            
            # LIVE ANCHOR: Update the last row with current job context
            # This "stitches" historical data with the live prediction request
            intensity = ((job_request.n_cores / HPCConfig.MAX_CORES) + 
                        (job_request.n_gpus / HPCConfig.MAX_GPUS))
            
            # Update last row with live values
            if 'Resource_Intensity' in lstm_features:
                df_seq.loc[seq_len-1, 'Resource_Intensity'] = intensity
            if 'Queue_Depth' in lstm_features:
                df_seq.loc[seq_len-1, 'Queue_Depth'] = live_queue_depth
            
            # Convert to numpy array [1, seq_len, n_features]
            sequence = df_seq.values.astype(np.float32).reshape(1, seq_len, len(lstm_features))
            
            # Scale the sequence
            n_samples, seq_len_actual, n_feat = sequence.shape
            seq_flat = sequence.reshape(-1, n_feat)
            seq_scaled_flat = self.scalers['lstm_scaler'].transform(seq_flat)
            seq_scaled = seq_scaled_flat.reshape(n_samples, seq_len_actual, n_feat)
            
            # Predict
            y_log = self.lstm_model.predict(seq_scaled, verbose=0)
            prediction = float(np.expm1(y_log[0, 0]))
            
            logging.info(f"LSTM prediction from real data: {prediction:.0f}s (Queue Depth: {live_queue_depth})")
            return prediction
            
        except Exception as e:
            logging.warning(f"LSTM prediction failed: {e}")
            return None
    
    def _calculate_confidence(self, rf_pred: float, lstm_pred: float) -> float:
        """Calculate confidence based on RF-LSTM agreement."""
        if rf_pred is None or lstm_pred is None:
            return 1.0
        
        avg = (rf_pred + lstm_pred) / 2
        rel_diff = abs(rf_pred - lstm_pred) / (avg + 1)
        return 1.0 / (1.0 + rel_diff)
    
    def predict(self, job_request, traffic: float, 
                method: str = 'weighted', use_ram: bool = False) -> PredictionResult:
        """
        Make hybrid prediction.
        
        Args:
            job_request: Job parameters
            traffic: Current traffic level (used as live Queue Depth for LSTM)
            method: 'rf_only', 'lstm_only', 'weighted', 'stacked'
            use_ram: Whether to include RAM in predictions (v5.0: defaults to False)
        """
        # Generate features - v5.0: Pass use_ram flag
        df_features = self.engineer.create_features(job_request, traffic, use_ram=use_ram)
        
        # Get component predictions
        rf_pred = self._get_rf_prediction(df_features)
        
        # v5.1: Use real historical data for LSTM with live queue depth anchor
        lstm_pred = self._get_lstm_prediction(job_request, live_queue_depth=traffic)
        
        # Calculate confidence
        confidence = self._calculate_confidence(rf_pred, lstm_pred)
        
        # Combine predictions based on method
        if method == 'rf_only' or lstm_pred is None:
            final_pred = rf_pred if rf_pred else 0.0
            method_used = 'rf_only'
        elif method == 'lstm_only':
            final_pred = lstm_pred
            method_used = 'lstm_only'
        elif method == 'weighted':
            if rf_pred and lstm_pred:
                final_pred = self.rf_weight * rf_pred + self.lstm_weight * lstm_pred
                method_used = 'weighted'
            else:
                final_pred = rf_pred or lstm_pred or 0.0
                method_used = 'fallback'
        elif method == 'stacked' and self.stacked_model is not None:
            # Use stacked model (RF with LSTM trend as feature)
            try:
                lstm_trend = lstm_pred / (max(rf_pred, lstm_pred, 1))  # Normalized
                static_features = df_features[self.scalers['rf_features']].copy()
                static_features['LSTM_Cluster_Trend'] = lstm_trend
                X_scaled = self.scalers.get('stacked_scaler', self.scalers['rf_scaler']).transform(static_features)
                y_log = self.stacked_model.predict(X_scaled)
                final_pred = float(np.expm1(y_log[0]))
                method_used = 'stacked'
            except:
                final_pred = self.rf_weight * rf_pred + self.lstm_weight * lstm_pred if rf_pred and lstm_pred else (rf_pred or lstm_pred or 0.0)
                method_used = 'weighted_fallback'
        else:
            final_pred = rf_pred if rf_pred else 0.0
            method_used = 'rf_fallback'
        
        # Calculate probability (estimate based on prediction)
        prob_waiting = min(1.0, final_pred / 3600) if final_pred > 60 else 0.1
        
        return PredictionResult(
            expected_wait_seconds=final_pred,
            probability_immediate_start=1.0 - prob_waiting,
            estimated_start_time=job_request.submission_datetime + timedelta(seconds=final_pred),
            predicted_runtime=job_request.walltime_seconds,  # Placeholder
            rf_prediction=rf_pred,
            lstm_prediction=lstm_pred,
            confidence=confidence,
            prediction_method=method_used
        )

# =============================================================================
# 5. STREAMLIT APP
# =============================================================================

class HPCPredictorApp:
    def __init__(self):
        self.config = AppConfig()
        
    def load_models(self):
        """Load legacy (hurdle) models."""
        try:
            metadata = joblib.load(self.config.METADATA_FILE)
            wait_model = joblib.load(self.config.WAIT_MODEL_FILE)
            
            runtime_model = None
            if metadata.get('has_runtime_model', False):
                if self.config.RUNTIME_MODEL_FILE.exists():
                    runtime_model = joblib.load(self.config.RUNTIME_MODEL_FILE)
            
            # Restore feature names
            wait_model.feature_names = metadata.get("wait_features", [])
            if runtime_model:
                runtime_model.feature_names = metadata.get("runtime_features", [])
                
            return runtime_model, wait_model, metadata
        except Exception as e:
            st.error(f"Error loading legacy models: {e}")
            return None, None, None
    
    def load_hybrid_models(self):
        """Load hybrid RF-LSTM models."""
        try:
            if not self.config.HYBRID_METADATA_FILE.exists():
                return None, None, None, None, None
            
            metadata = joblib.load(self.config.HYBRID_METADATA_FILE)
            scalers = joblib.load(self.config.HYBRID_SCALER_FILE)
            st.write("🔍 DEBUG — LSTM Scaler Expectation")
            st.write("LSTM scaler expects:", scalers['lstm_scaler'].n_features_in_)
            st.write("LSTM features list length:", len(scalers.get('lstm_features', [])))
            st.write("LSTM features:", scalers.get('lstm_features', []))
            
            # Load RF model
            rf_model = None
            if self.config.HYBRID_RF_MODEL_FILE.exists():
                rf_model = joblib.load(self.config.HYBRID_RF_MODEL_FILE)
            
            # Load LSTM model
            lstm_model = None
            if TENSORFLOW_AVAILABLE and self.config.HYBRID_LSTM_MODEL_FILE.exists():
                lstm_model = load_model(str(self.config.HYBRID_LSTM_MODEL_FILE))
            
            # Load stacked model
            stacked_model = None
            if self.config.HYBRID_STACKED_MODEL_FILE.exists():
                stacked_bundle = joblib.load(self.config.HYBRID_STACKED_MODEL_FILE)
                stacked_model = stacked_bundle.get('model')
                scalers['stacked_scaler'] = stacked_bundle.get('scaler')
            
            return rf_model, lstm_model, stacked_model, scalers, metadata
        except Exception as e:
            st.warning(f"Could not load hybrid models: {e}")
            return None, None, None, None, None

    def render_admin_login(self):
        """Render the admin login/register dialog."""
        with st.expander("Admin Login", expanded=False):
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username", key="login_username")
                    password = st.text_input("Password", type="password", key="login_password")
                    submit = st.form_submit_button("Login")
                    
                    if submit:
                        if username and password:
                            success, msg = verify_user(username, password)
                            if success:
                                st.session_state['admin_logged_in'] = True
                                st.session_state['admin_username'] = username
                                log_usage("admin_login", username, "Admin logged in")
                                st.success(f"Welcome, {username}!")
                                st.rerun()
                            else:
                                st.error(msg)
                        else:
                            st.warning("Please enter username and password")
            
            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("Username", key="reg_username")
                    new_email = st.text_input("Email (for approval notification)", key="reg_email")
                    new_password = st.text_input("Password", type="password", key="reg_password")
                    confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm")
                    register = st.form_submit_button("Register")
                    
                    if register:
                        if new_password != confirm_password:
                            st.error("Passwords do not match")
                        elif len(new_password) < 6:
                            st.error("Password must be at least 6 characters")
                        elif not new_username:
                            st.error("Username is required")
                        else:
                            success, msg = register_user(new_username, new_password, new_email)
                            if success:
                                st.success(msg)
                                log_usage("admin_register", new_username, "New admin registration requested (pending approval)")
                            else:
                                st.error(msg)
                
                st.info("New registrations require admin approval. You'll receive an email once approved.")
    
    def render_admin_panel(self):
        """Render the admin panel for managing architectures, queues, and viewing logs."""
        st.markdown("---")
        st.markdown(f"### Admin Panel — Logged in as: **{st.session_state.get('admin_username', 'Admin')}**")
        
        # Show pending approvals badge
        pending_users = get_pending_users()
        if pending_users:
            st.warning(f"{len(pending_users)} pending user approval(s) - Check 'User Approvals' tab")
        
        # Logout button
        col_logout = st.columns([6, 1])
        with col_logout[1]:
            if st.button("Logout"):
                log_usage("admin_logout", st.session_state.get('admin_username'), "Admin logged out")
                st.session_state['admin_logged_in'] = False
                st.session_state['admin_username'] = None
                st.rerun()
        
        # Admin tabs
        admin_tabs = st.tabs(["Dashboard", "User Approvals", "Architectures", "Queues", "Usage Logs", "Training Logs", "Email Settings"])
        
        # Dashboard Tab
        with admin_tabs[0]:
            st.subheader("System Dashboard")
            stats = get_usage_stats()
            
            stat_cols = st.columns(4)
            stat_cols[0].metric("Total Predictions", stats['total_predictions'])
            stat_cols[1].metric("Today's Predictions", stats['predictions_today'])
            stat_cols[2].metric("Most Used Queue", stats['most_used_queue'])
            stat_cols[3].metric("Most Used Architecture", stats['most_used_arch'])
            
            if stats['avg_predicted_wait'] > 0:
                from datetime import timedelta
                avg_wait = timedelta(seconds=int(stats['avg_predicted_wait']))
                st.info(f"Average Predicted Wait Time: **{avg_wait}**")
            
            # Quick stats
            st.markdown("#### Active Resources")
            res_cols = st.columns(2)
            with res_cols[0]:
                archs = get_architectures()
                st.write(f"**{len(archs)} Active Architectures:**")
                for a in archs:
                    st.write(f"  • {a['name']} ({a['cores_per_node']} cores, {a['ram_per_node']}GB RAM)")
            with res_cols[1]:
                queues = get_queues()
                st.write(f"**{len(queues)} Active Queues:**")
                for q in queues:
                    st.write(f"  • {q['name']} (max {q['soft_limit_hours']}h, priority {q['priority']})")
        
        # User Approvals Tab
        with admin_tabs[1]:
            st.subheader("User Registration Approvals")
            
            # Pending approvals section
            st.markdown("#### Pending Approvals")
            pending = get_pending_users()
            
            if pending:
                for user in pending:
                    with st.container():
                        cols = st.columns([3, 2, 2, 1, 1])
                        cols[0].write(f"**{user['username']}**")
                        cols[1].write(user['email'] or "No email")
                        cols[2].write(f"Requested: {user['created_at'][:16] if user['created_at'] else 'N/A'}")
                        
                        if cols[3].button("Approve", key=f"approve_{user['id']}", help="Approve"):
                            success, msg = approve_user(user['id'], st.session_state.get('admin_username'))
                            if success:
                                st.success(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                        
                        if cols[4].button("Reject", key=f"reject_{user['id']}", help="Reject"):
                            success, msg = reject_user(user['id'], st.session_state.get('admin_username'))
                            if success:
                                st.info(msg)
                                st.rerun()
                            else:
                                st.error(msg)
                    st.divider()
            else:
                st.success("No pending approvals")
            
            # All users section
            st.markdown("#### All Users")
            all_users = get_all_users()
            if all_users:
                users_df = pd.DataFrame(all_users)
                st.dataframe(users_df, use_container_width=True)
            else:
                st.info("No users registered yet.")
        
        # Architectures Tab
        with admin_tabs[2]:
            st.subheader("Manage Architectures")
            
            # Add new architecture
            with st.form("add_arch_form"):
                st.markdown("**Add New Architecture**")
                arch_cols = st.columns(4)
                arch_name = arch_cols[0].text_input("Name", placeholder="e.g., cascadelake")
                arch_cores = arch_cols[1].number_input("Cores/Node", 1, 256, 48)
                arch_ram = arch_cols[2].number_input("RAM/Node (GB)", 1, 2048, 128)
                arch_gpus = arch_cols[3].number_input("GPUs/Node", 0, 16, 2)
                arch_desc = st.text_input("Description", placeholder="Optional description")
                add_arch_btn = st.form_submit_button("Add Architecture")
                
                if add_arch_btn and arch_name:
                    success, msg = add_architecture(
                        arch_name, arch_cores, arch_ram, arch_gpus, arch_desc,
                        st.session_state.get('admin_username', 'admin')
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            
            # List existing architectures
            st.markdown("**Existing Architectures**")
            archs = get_architectures()
            if archs:
                arch_df = pd.DataFrame(archs)
                display_cols = ['id', 'name', 'cores_per_node', 'ram_per_node', 'gpus_per_node', 'description', 'created_at']
                st.dataframe(arch_df[display_cols], use_container_width=True)
                
                # Delete architecture
                del_arch_id = st.selectbox("Select Architecture to Delete", 
                                           options=[a['id'] for a in archs],
                                           format_func=lambda x: next(a['name'] for a in archs if a['id'] == x),
                                           key="del_arch")
                if st.button("Delete Selected Architecture", key="del_arch_btn"):
                    if del_arch_id:
                        success, msg = delete_architecture(del_arch_id, st.session_state.get('admin_username'))
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info("No architectures found.")
        
        # Queues Tab
        with admin_tabs[3]:
            st.subheader("Manage Queues")
            
            # Add new queue
            with st.form("add_queue_form"):
                st.markdown("**Add New Queue**")
                q_cols1 = st.columns(3)
                q_name = q_cols1[0].text_input("Queue Name", placeholder="e.g., gpu_high")
                q_limit = q_cols1[1].number_input("Soft Limit (hours)", 1, 720, 168)
                q_priority = q_cols1[2].number_input("Priority", 1, 10, 1)
                
                q_cols2 = st.columns(3)
                q_max_cores = q_cols2[0].number_input("Max Cores", 1, 2000, 256)
                q_max_gpus = q_cols2[1].number_input("Max GPUs", 0, 128, 8)
                q_max_mem = q_cols2[2].number_input("Max Memory (GB)", 1, 4096, 1024)
                
                q_desc = st.text_input("Description", placeholder="Optional description", key="q_desc")
                add_queue_btn = st.form_submit_button("Add Queue")
                
                if add_queue_btn and q_name:
                    success, msg = add_queue(
                        q_name, q_limit, q_max_cores, q_max_gpus, q_max_mem, q_priority, q_desc,
                        st.session_state.get('admin_username', 'admin')
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
            
            # List existing queues
            st.markdown("**Existing Queues**")
            queues = get_queues()
            if queues:
                queue_df = pd.DataFrame(queues)
                display_cols = ['id', 'name', 'soft_limit_hours', 'max_cores', 'max_gpus', 'max_memory_gb', 'priority', 'description']
                st.dataframe(queue_df[display_cols], use_container_width=True)
                
                # Delete queue
                del_queue_id = st.selectbox("Select Queue to Delete",
                                           options=[q['id'] for q in queues],
                                           format_func=lambda x: next(q['name'] for q in queues if q['id'] == x),
                                           key="del_queue")
                if st.button("Delete Selected Queue", key="del_queue_btn"):
                    if del_queue_id:
                        success, msg = delete_queue(del_queue_id, st.session_state.get('admin_username'))
                        if success:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info("No queues found.")
        
        # Usage Logs Tab
        with admin_tabs[4]:
            st.subheader("Usage Logs")
            
            log_limit = st.slider("Number of logs to display", 10, 500, 100)
            logs = get_usage_logs(log_limit)
            
            if logs:
                logs_df = pd.DataFrame(logs)
                
                # Filter options
                filter_cols = st.columns(3)
                actions = ['All'] + list(logs_df['action'].unique())
                selected_action = filter_cols[0].selectbox("Filter by Action", actions)
                
                if selected_action != 'All':
                    logs_df = logs_df[logs_df['action'] == selected_action]
                
                st.dataframe(logs_df, use_container_width=True, height=400)
                
                # Export option
                csv = logs_df.to_csv(index=False)
                st.download_button(
                    label="Export Logs as CSV",
                    data=csv,
                    file_name=f"usage_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No usage logs found.")
        
        # Training Logs Tab
        with admin_tabs[5]:
            st.subheader("Model Training Logs")
            
            t_logs = get_training_logs(50)
            
            if t_logs:
                t_logs_df = pd.DataFrame(t_logs)
                
                # Summary metrics
                summary_cols = st.columns(4)
                summary_cols[0].metric("Total Training Runs", len(t_logs))
                successful = len([l for l in t_logs if l['status'] == 'success'])
                summary_cols[1].metric("Successful Runs", successful)
                failed = len([l for l in t_logs if l['status'] == 'failed'])
                summary_cols[2].metric("Failed Runs", failed)
                
                if t_logs:
                    last_train = t_logs[0]['timestamp']
                    summary_cols[3].metric("Last Training", last_train[:19] if last_train else "N/A")
                
                st.dataframe(t_logs_df, use_container_width=True, height=400)
                
                # Export option
                csv = t_logs_df.to_csv(index=False)
                st.download_button(
                    label="Export Training Logs as CSV",
                    data=csv,
                    file_name=f"training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No training logs found.")
        
        # Email Settings Tab
        with admin_tabs[6]:
            st.subheader("Email Configuration")
            st.markdown("Configure SMTP settings for sending approval notification emails.")
            
            # Get current config
            email_config = get_email_config()
            
            # Status indicator
            if email_config.get('is_configured'):
                st.success("Email is configured")
            else:
                st.warning("Email not configured - approval notifications will not be sent")
            
            with st.form("email_config_form"):
                st.markdown("**SMTP Settings**")
                
                smtp_cols = st.columns(2)
                smtp_server = smtp_cols[0].text_input(
                    "SMTP Server", 
                    value=email_config.get('smtp_server', 'smtp.gmail.com'),
                    help="e.g., smtp.gmail.com, smtp.outlook.com"
                )
                smtp_port = smtp_cols[1].number_input(
                    "SMTP Port", 
                    value=email_config.get('smtp_port', 587),
                    help="Usually 587 for TLS, 465 for SSL"
                )
                
                st.markdown("**Sender Credentials**")
                cred_cols = st.columns(2)
                sender_email = cred_cols[0].text_input(
                    "Sender Email",
                    value=email_config.get('sender_email', ''),
                    help="Email address to send notifications from"
                )
                sender_password = cred_cols[1].text_input(
                    "App Password",
                    type="password",
                    value=email_config.get('sender_password', ''),
                    help="For Gmail, use an App Password (not your regular password)"
                )
                
                st.markdown("**Notification Settings**")
                admin_email = st.text_input(
                    "Admin Notification Email",
                    value=email_config.get('admin_notification_email', ''),
                    help="Email address where new registration requests will be sent"
                )
                
                save_btn = st.form_submit_button("Save Configuration")
                
                if save_btn:
                    if smtp_server and sender_email and sender_password and admin_email:
                        success, msg = save_email_config(
                            smtp_server, smtp_port, sender_email, sender_password, admin_email
                        )
                        if success:
                            st.success(msg)
                            log_usage("email_config_updated", st.session_state.get('admin_username'), "Email configuration updated")
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("Please fill in all required fields")
            
            # Test email section
            st.markdown("---")
            st.markdown("**Test Email Configuration**")
            test_email = st.text_input("Send test email to:", placeholder="test@example.com")
            if st.button("Send Test Email"):
                if test_email:
                    success, msg = send_email(
                        test_email,
                        "Test Email - IITD HPC Predictor",
                        "<h2>Test Email</h2><p>This is a test email from the IITD HPC Wait Time Predictor admin panel.</p><p>If you received this, email configuration is working correctly!</p>"
                    )
                    if success:
                        st.success(f"Test email sent to {test_email}")
                    else:
                        st.error(f"Failed: {msg}")
                else:
                    st.warning("Please enter an email address")
            
            # Gmail setup instructions
            with st.expander("Gmail Setup Instructions"):
                st.markdown("""
                **To use Gmail for sending emails:**
                
                1. **Enable 2-Factor Authentication** on your Google account
                2. **Generate an App Password:**
                   - Go to [Google Account Security](https://myaccount.google.com/security)
                   - Under "2-Step Verification", click on "App passwords"
                   - Select "Mail" and "Other (Custom name)"
                   - Enter "HPC Predictor" and click "Generate"
                   - Copy the 16-character password
                3. **Use these settings:**
                   - SMTP Server: `smtp.gmail.com`
                   - SMTP Port: `587`
                   - Sender Email: Your Gmail address
                   - App Password: The 16-character password generated above
                """)
        
        st.markdown("---")

    def run(self):
        st.set_page_config(page_title="IITD HPC Predictor", page_icon="⚡", layout="wide")
        
        # Initialize session state for admin login
        if 'admin_logged_in' not in st.session_state:
            st.session_state['admin_logged_in'] = False
        if 'admin_username' not in st.session_state:
            st.session_state['admin_username'] = None
        
        # --- TOP BAR WITH ADMIN LOGIN ---
        top_cols = st.columns([2, 3])
        with top_cols[0]:
            if st.session_state['admin_logged_in']:
                st.markdown(f"👤 **{st.session_state['admin_username']}**")
            else:
                self.render_admin_login()
        
        # --- HEADER ---
        st.markdown("""
        <style>.stApp {background-color: #0E1117; color: white;} 
        .header {padding: 20px; background: linear-gradient(90deg, #1A1D24, #0E1117); border-left: 5px solid #E31E24; margin-bottom: 20px;}
        </style>
        <div class="header">
            <h2>IIT Delhi HPC Facility</h2>
            <p style="opacity: 0.7">Hybrid RF-LSTM Job Wait Time Predictor v5.2 (Schedule-Agnostic + Admin Panel)</p>
        </div>
        """, unsafe_allow_html=True)
        
        # --- ADMIN PANEL (if logged in) ---
        if st.session_state['admin_logged_in']:
            self.render_admin_panel()

        # --- LOAD RESOURCES ---
        runtime_model, wait_model, metadata = self.load_models()
        rf_model, lstm_model, stacked_model, scalers, hybrid_metadata = self.load_hybrid_models()
        
        # Determine available services
        has_legacy = wait_model is not None
        has_hybrid = rf_model is not None
        has_lstm = lstm_model is not None
        
        if not has_legacy and not has_hybrid:
            st.error("No models found. Please run model training first.")
            return
        
        # Create services
        legacy_service = PredictionService(runtime_model, wait_model, metadata) if has_legacy else None
        hybrid_service = HybridPredictionService(
            rf_model, lstm_model, stacked_model, scalers, hybrid_metadata, self.config
        ) if has_hybrid else None

        # --- Get dynamic queues and architectures from DB ---
        db_queues = get_queues()
        db_archs = get_architectures()
        queue_names = [q['name'] for q in db_queues] if db_queues else ["standard", "high"]
        arch_names = [a['name'] for a in db_archs] if db_archs else ["any", "skylake", "icelake"]
        
        # Create arch config lookup
        arch_config_lookup = {a['name']: a for a in db_archs} if db_archs else {}

        # --- SIDEBAR ---
        with st.sidebar:
            st.header("Job Parameters")
            
            # 1. Standard Inputs (now using dynamic values from DB)
            queue = st.selectbox("Queue", queue_names)
            arch = st.selectbox("Architecture", arch_names, index=0)
            
            c1, c2 = st.columns(2)
            cores = c1.number_input("Cores", 1, 2000, 24)
            gpus = c2.number_input("GPUs", 0, 32, 0)

            # v5.0: RAM is now optional (defaults to off)
            st.divider()
            use_ram = st.checkbox("Specify RAM (Optional)", value=False,
                                  help="RAM is optional in v5.0. The No-RAM model focuses on Cores/GPUs/Walltime.")
            
            if use_ram:
                ram = st.number_input("RAM (GB)", 1, 1024, 64,
                                     help="Note: RAM may be ignored by the prediction model")
            else:
                ram = 0  # Default when RAM is not specified
                st.caption("RAM not specified - using CPU-based node estimation")

            # 2. Advanced Scheduling (Now interactive)
            st.subheader("Advanced Scheduling")
            # Because we removed st.form, this checkbox now triggers an immediate rerun
            use_manual_nodes = st.checkbox("Enable Advanced Scheduling Options")

            manual_nodes = None
            if use_manual_nodes:
                manual_nodes = st.number_input(
                    "Manually Specify Nodes",
                    min_value=1,
                    max_value=64,
                    value=1,
                    help="Override automatic node calculation. Use only if you know your job node layout."
                )

            # 3. Logic & Calculations - Now using DB values
            arch_key = arch.lower()

            # Get architecture caps from DB or use defaults
            if arch_key in arch_config_lookup:
                arch_cfg = arch_config_lookup[arch_key]
                cores_per_node = arch_cfg['cores_per_node']
                ram_per_node = arch_cfg['ram_per_node']
                gpus_per_node = arch_cfg['gpus_per_node']
            elif arch_key == "icelake":
                cores_per_node = 64
                ram_per_node = 256
                gpus_per_node = 2
            elif arch_key == "skylake":
                cores_per_node = 40
                ram_per_node = 96
                gpus_per_node = 2
            else:  # any / unknown
                cores_per_node = 48
                ram_per_node = 128
                gpus_per_node = 2

            # Node requirements - v5.0: RAM-based only if use_ram=True
            nodes_core = int(np.ceil(cores / cores_per_node)) if cores > 0 else 1
            nodes_gpu = int(np.ceil(gpus / gpus_per_node)) if gpus > 0 else 1
            
            if use_ram and ram > 0:
                nodes_ram = int(np.ceil(ram / ram_per_node))
                estimated_nodes = max(nodes_core, nodes_ram, nodes_gpu)
            else:
                nodes_ram = 0
                estimated_nodes = max(nodes_core, nodes_gpu)

            # ===== Status Messaging =====
            if estimated_nodes <= 1:
                st.success("Single-Node Job - fits comfortably on one node")
            else:
                msg = f"Multi-Node Job Detected\n\nEstimated Nodes Required: **{estimated_nodes}**\n\n- By Cores → needs {nodes_core} node(s)\n"
                if use_ram:
                    msg += f"- By RAM → needs {nodes_ram} node(s)\n"
                msg += f"- By GPUs → needs {nodes_gpu} node(s)"
                st.info(msg)

            # ===== Safety Warnings =====
            if use_ram and ram > 512:
                st.warning("Requesting >512GB RAM - rare nodes only. Queue delays likely.")

            if cores > 128:
                st.warning("High core request (Top 1%). Expect potential wait.")

            if gpus > 4:
                st.warning("Multi-node GPU scheduling required. Limited availability.")

            st.subheader("Walltime Request")
            c3, c4 = st.columns(2)
            wt_h = c3.number_input("Hours", 0, 168, 24)
            wt_m = c4.number_input("Mins", 0, 59, 0, step=15)
            
            # v5.1: Traffic Congestion removed - model is schedule-agnostic
            # Uses Queue_Depth from training data instead
            traffic = 0  # Default value (not used in schedule-agnostic model)
            
            # Model Selection (if hybrid available)
            st.divider()
            st.subheader("Model Selection")
            
            model_options = []
            if has_legacy:
                model_options.append("Legacy (Hurdle Model)")
            if has_hybrid:
                model_options.append("Hybrid RF (Static)")
                if has_lstm:
                    model_options.append("Hybrid LSTM (Temporal)")
                    model_options.append("Hybrid Weighted (RF + LSTM)")
                    model_options.append("Hybrid Stacked (Advanced)")
            
            if len(model_options) > 1:
                selected_model = st.selectbox("Prediction Method", model_options, index=len(model_options)-1)
            else:
                selected_model = model_options[0] if model_options else "Legacy (Hurdle Model)"
            
            show_comparison = st.checkbox("Show Model Comparison", value=has_hybrid and has_lstm,
                                         help="Compare RF vs LSTM predictions for confidence analysis")
            
            # 4. Trigger Button
            run_btn = st.button("Predict Wait Time", type="primary")

        # --- MAIN PANEL ---
        if run_btn:
            req = JobRequest(
                queue=queue, n_cores=cores, n_gpus=gpus, ram_gb=ram,
                walltime_seconds=(wt_h*3600 + wt_m*60),
                architecture=arch,
                submission_datetime=datetime.now(),
                manual_nodes=manual_nodes if use_manual_nodes else None
            )
            
            # Determine which service/method to use
            method_map = {
                "Legacy (Hurdle Model)": ("legacy", None),
                "Hybrid RF (Static)": ("hybrid", "rf_only"),
                "Hybrid LSTM (Temporal)": ("hybrid", "lstm_only"),
                "Hybrid Weighted (RF + LSTM)": ("hybrid", "weighted"),
                "Hybrid Stacked (Advanced)": ("hybrid", "stacked")
            }
            
            service_type, method = method_map.get(selected_model, ("legacy", None))
            
            with st.spinner("Calculating Prediction..."):
                if service_type == "hybrid" and hybrid_service:
                    res = hybrid_service.predict(req, traffic, method=method, use_ram=use_ram)
                elif legacy_service:
                    res = legacy_service.predict(req, traffic, use_ram=use_ram)
                else:
                    st.error("No prediction service available")
                    return
            
            # Log the prediction usage
            log_usage(
                action="prediction",
                user=st.session_state.get('admin_username', 'anonymous'),
                details=f"Model: {selected_model}, Method: {method or 'legacy'}",
                queue=queue,
                arch=arch,
                cores=cores,
                gpus=gpus,
                predicted_wait=res.expected_wait_seconds
            )
            
            # --- RESULTS DISPLAY ---
            
            # Status Box
            colors = {"immediate": "#22C55E", "short": "#3B82F6", "moderate": "#EAB308", "extended": "#EF4444"}
            msgs = {
                "immediate": "System resources available. Job should start immediately.",
                "short": "Moderate load. Expect a short wait.",
                "moderate": "System busy. Expect a noticeable wait (hours).",
                "extended": "Heavy congestion. Extended wait expected."
            }
            
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background: {colors[res.status]}20; border-left: 5px solid {colors[res.status]};">
                <h3 style="color: {colors[res.status]}; margin:0;">{res.status.upper()} START EXPECTED</h3>
                <p style="margin:5px 0 0 0;">{msgs[res.status]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer
            
            # Metrics Columns
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Wait Time", res.format_time(res.expected_wait_seconds))
            col2.metric("Start Probability", f"{res.probability_immediate_start:.1%}")
            col3.metric("Projected Start", res.estimated_start_time.strftime("%H:%M %p"))
            
            # Show confidence if available
            if res.confidence is not None:
                col4.metric("Confidence", f"{res.confidence:.1%}", 
                           help=f"Based on RF-LSTM agreement ({res.confidence_level})")
            else:
                col4.metric("Predicted Runtime", res.format_time(res.predicted_runtime), 
                           help="Model A Prediction")
            
            # --- MODEL COMPARISON (if enabled and hybrid) ---
            if show_comparison and res.rf_prediction is not None:
                st.write("")
                st.subheader("Model Comparison Analysis")
                
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.markdown("**Random Forest (Static)**")
                    st.markdown(f"### {res.format_time(res.rf_prediction)}")
                    st.caption("Based on job specifications (cores, memory, walltime)")
                
                with comp_col2:
                    if res.lstm_prediction is not None:
                        st.markdown("**LSTM (Temporal)**")
                        st.markdown(f"### {res.format_time(res.lstm_prediction)}")
                        st.caption("Based on real cluster history (last 50 jobs)")
                    else:
                        st.markdown("**LSTM (Temporal)**")
                        st.markdown("### N/A")
                        st.caption("LSTM needs retraining (feature mismatch)")
                
                with comp_col3:
                    st.markdown("**Combined Prediction**")
                    st.markdown(f"### {res.format_time(res.expected_wait_seconds)}")
                    st.caption(f"Method: {res.prediction_method}")
                
                # Agreement Analysis
                if res.agreement_analysis:
                    st.info(res.agreement_analysis)
                
                # Visual comparison
                if res.rf_prediction is not None and res.lstm_prediction is not None:
                    import streamlit.components.v1 as components
                    
                    st.write("")
                    st.markdown("**Prediction Breakdown:**")
                    
                    max_pred = max(res.rf_prediction, res.lstm_prediction, res.expected_wait_seconds)
                    
                    rf_pct = (res.rf_prediction / max_pred) * 100 if max_pred > 0 else 0
                    lstm_pct = (res.lstm_prediction / max_pred) * 100 if max_pred > 0 else 0
                    combined_pct = (res.expected_wait_seconds / max_pred) * 100 if max_pred > 0 else 0
                    
                    st.progress(rf_pct / 100, text=f"RF: {res.format_time(res.rf_prediction)}")
                    st.progress(lstm_pct / 100, text=f"LSTM: {res.format_time(res.lstm_prediction)}")
                    st.progress(combined_pct / 100, text=f"Combined: {res.format_time(res.expected_wait_seconds)}")
            
            # Technical Details Expander
            with st.expander("Technical Analysis (Debug Info)"):
                debug_info = {
                    "Prediction Method": res.prediction_method,
                    "Selected Model": selected_model,
                    "Raw Wait Prediction (s)": res.expected_wait_seconds,
                    "Wait Probability": f"{1.0 - res.probability_immediate_start:.4f}",
                    "RAM Enabled (v5.1)": use_ram,
                    "RAM Value (GB)": ram if use_ram else "N/A (No-RAM Model)",
                    "Manual Nodes Configured": manual_nodes if use_manual_nodes else "Auto",
                    "Model Version": "5.1.1 (Real LSTM History)",
                    "LSTM Data Source": "clean_hpc_data1.csv (last 500 rows)"
                }
                
                if res.rf_prediction is not None:
                    debug_info["RF Prediction (s)"] = res.rf_prediction
                if res.lstm_prediction is not None:
                    debug_info["LSTM Prediction (s)"] = res.lstm_prediction
                if res.confidence is not None:
                    debug_info["Confidence Score"] = f"{res.confidence:.4f}"
                    debug_info["Confidence Level"] = res.confidence_level
                
                st.json(debug_info)
                
                # Model info
                if has_hybrid and hybrid_metadata:
                    st.markdown("**Model Configuration:**")
                    config_info = hybrid_metadata.get('config', {})
                    st.json({
                        "RF Weight": config_info.get('rf_weight', 0.4),
                        "LSTM Weight": config_info.get('lstm_weight', 0.6),
                        "LSTM Sequence Length": config_info.get('lstm_sequence_length', 50),
                        "TensorFlow Available": TENSORFLOW_AVAILABLE,
                        "LSTM Model Loaded": lstm_model is not None
                    })

        else:
            st.info("Configure your job in the sidebar to get a prediction.")
            
            # Model status
            st.divider()
            status_cols = st.columns(3)
            with status_cols[0]:
                if has_legacy:
                    st.success("Legacy Model Loaded")
                else:
                    st.warning("Legacy Model Not Found")
            with status_cols[1]:
                if has_hybrid:
                    st.success("Hybrid RF Model Loaded")
                else:
                    st.info("Hybrid RF Not Available")
            with status_cols[2]:
                if has_lstm:
                    st.success("LSTM Model Loaded")
                else:
                    st.info("LSTM Not Available")
            
            st.caption(f"System: Hybrid RF-LSTM Predictor v{self.config.APP_VERSION}")

if __name__ == "__main__":
    HPCPredictorApp().run()