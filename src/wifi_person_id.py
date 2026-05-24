#!/usr/bin/env python3
"""
WiFi-Based Person Identification System
Based on research from Karlsruher Institut für Technologie (KIT)
Uses beamforming feedback information (BFI) to identify people via WiFi signals
"""

import numpy as np
import pandas as pd
import json
import time
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WiFiPersonIdentifier:
    """
    Main class for identifying people using WiFi beamforming feedback information
    """
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def extract_bfi_features(self, bfi_data: np.ndarray) -> np.ndarray:
        """
        Extract features from Beamforming Feedback Information (BFI)
        BFI contains information about how signals reflect off objects/people in the environment
        """
        # In a real implementation, this would process actual BFI data from WiFi chipsets
        # For demonstration, we'll simulate feature extraction
        
        features = []
        
        # Statistical features
        features.append(np.mean(bfi_data))  # Mean signal strength
        features.append(np.std(bfi_data))   # Signal variation
        features.append(np.max(bfi_data))   # Peak signal
        features.append(np.min(bfi_data))   # Minimum signal
        
        # Frequency domain features (simulated)
        fft_data = np.fft.fft(bfi_data)
        features.append(np.mean(np.abs(fft_data[:len(fft_data)//2])))  # Low frequency energy
        features.append(np.mean(np.abs(fft_data[len(fft_data)//2:])))  # High frequency energy
        
        # Signal pattern features
        features.append(np.sum(np.diff(bfi_data) > 0))  # Number of positive transitions
        features.append(np.sum(np.diff(bfi_data) < 0))  # Number of negative transitions
        
        # Statistical moments
        from scipy import stats
        if len(bfi_data) > 1:
            features.append(stats.skew(bfi_data))      # Skewness
            features.append(stats.kurtosis(bfi_data))  # Kurtosis
        else:
            features.extend([0, 0])
            
        return np.array(features)
    
    def collect_training_data(self, duration_seconds: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate collecting training data from WiFi signals
        In practice, this would interface with WiFi hardware to collect BFI data
        """
        logger.info(f"Collecting training data for {duration_seconds} seconds...")
        
        # Simulate data collection - in reality, this would read from WiFi interface
        X = []  # Features
        y = []  # Labels (person IDs)
        
        # Simulate multiple people
        n_people = 5
        samples_per_person = 20
        
        for person_id in range(n_people):
            logger.info(f"Collecting data for person {person_id}")
            
            for sample in range(samples_per_person):
                # Simulate BFI data for a person
                # Each person has a unique signal reflection pattern
                base_signal = np.random.normal(0, 1, 100)
                
                # Add person-specific signature
                person_signature = np.sin(np.linspace(0, person_id*np.pi, 100)) * 0.5
                noise = np.random.normal(0, 0.1, 100)
                
                bfi_sample = base_signal + person_signature + noise
                
                # Extract features
                features = self.extract_bfi_features(bfi_sample)
                X.append(features)
                y.append(person_id)
                
                # Small delay to simulate real-time collection
                time.sleep(0.01)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Collected {len(X)} samples from {n_people} people")
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the person identification model
        """
        logger.info("Training WiFi person identification model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model (Random Forest works well for this type of classification)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        logger.info(f"Model trained with accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'n_features': X.shape[1],
            'n_samples': len(X)
        }
    
    def predict(self, bfi_data: np.ndarray) -> Tuple[int, float]:
        """
        Predict person ID from BFI data
        Returns: (person_id, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_bfi_features(bfi_data).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return int(prediction), float(confidence)
    
    def save_model(self, path: str):
        """Save the trained model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model and scaler"""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {path}")

def simulate_wifi_interface():
    """
    Simulate a WiFi interface that provides BFI data
    In a real implementation, this would interface with actual WiFi hardware/drivers
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting simulated WiFi interface...")
    
    # In reality, you would:
    # 1. Put WiFi card in monitor mode
    # 2. Capture BFI frames from associated devices
    # 3. Extract the beamforming feedback information
    # 4. Process it in real-time
    
    # For simulation, we'll generate synthetic BFI-like data
    while True:
        # Simulate receiving BFI data from connected devices
        # BFI is typically a matrix of complex numbers representing channel state
        bfi_matrix = np.random.complex128((8, 8)) * (np.random.rand(8, 8) + 1j*np.random.rand(8, 8))
        
        # Convert to real-valued features for our model
        bfi_features = np.abs(bfi_matrix).flatten()
        
        yield bfi_features
        time.sleep(0.1)  # Simulate 10Hz sampling rate

def main():
    """
    Main function to demonstrate the WiFi person identification system
    """
    print("=" * 60)
    print("WiFi-Based Person Identification System")
    print("Based on KIT research: Ordinary WiFi can identify people")
    print("=" * 60)
    
    # Initialize the identifier
    identifier = WiFiPersonIdentifier()
    
    # Collect training data
    print("\n1. Collecting training data...")
    X, y = identifier.collect_training_data(duration_seconds=30)
    
    # Train the model
    print("\n2. Training model...")
    results = identifier.train(X, y)
    print(f"   Training accuracy: {results['accuracy']:.4f}")
    
    # Save the model
    model_path = '/home/linuxuser/wifi-person-id/models/wifi_person_id_model.pkl'
    identifier.save_model(model_path)
    
    # Demonstrate real-time identification
    print("\n3. Demonstrating real-time identification...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        wifi_interface = simulate_wifi_interface()
        
        for i, bfi_data in enumerate(wifi_interface):
            if i >= 10:  # Limit demo to 10 samples
                break
                
            person_id, confidence = identifier.predict(bfi_data)
            print(f"   Sample {i+1}: Person ID {person_id} (confidence: {confidence:.3f})")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n   Stopped by user")
    
    print("\n4. System ready for deployment!")
    print(f"   Model saved to: {model_path}")
    print("\nNote: This is a demonstration implementation.")
    print("For actual WiFi hardware integration, you would need:")
    print("- WiFi chipset that exposes BFI/CSI data")
    print("- Driver modifications to capture BFI frames")
    print("- Real-time signal processing pipeline")
    print("- Proper calibration for your specific environment")

if __name__ == "__main__":
    main()