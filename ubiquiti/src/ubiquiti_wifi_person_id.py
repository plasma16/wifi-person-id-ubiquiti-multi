#!/usr/bin/env python3
"""
Ubiquiti Multi-Floor Multi-AP WiFi-Based Person Identification System
Extends the base WiFi person identification system to support multiple
Ubiquiti access points across multiple floors.
"""

import numpy as np
import pandas as pd
import json
import time
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UbiquitiWiFiPersonIdentifier:
    """
    Main class for identifying people using WiFi beamforming feedback information
    from multiple Ubiquiti APs across multiple floors.
    """
    
    def __init__(self, config_path: str = None, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.aps_config = {}  # Will hold AP configuration
        self.ap_features = {}  # Per-AP feature scaler and model if needed
        
        # Load Ubiquiti AP configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'ubiquiti_aps.json')
        self.load_ap_config(config_path)
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_ap_config(self, config_path: str) -> Dict:
        """Load Ubiquiti AP configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.aps_config = config.get('floors', {})
            logger.info(f"Loaded AP configuration for {len(self.aps_config)} floors")
            total_aps = sum(len(floor.get('aps', [])) for floor in self.aps_config.values())
            logger.info(f"Total APs configured: {total_aps}")
            return self.aps_config
        except FileNotFoundError:
            logger.warning(f"AP config file not found: {config_path}. Using empty config.")
            self.aps_config = {}
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in AP config: {e}")
            self.aps_config = {}
            return {}
    
    def extract_bfi_features(self, bfi_data: np.ndarray, ap_id: str = None) -> np.ndarray:
        """
        Extract features from Beamforming Feedback Information (BFI)
        BFI contains information about how signals reflect off objects/people in the environment
        Includes AP-specific features if ap_id is provided.
        """
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
        
        # AP-specific features (if AP ID is provided)
        if ap_id is not None:
            ap_info = self.get_ap_info(ap_id)
            if ap_info:
                # Add AP location features (normalized)
                loc = ap_info.get('location', {'x': 0, 'y': 0, 'z': 0})
                # Normalize coordinates (assuming max 20m in x/y, 10m in z)
                features.append(loc.get('x', 0) / 20.0)
                features.append(loc.get('y', 0) / 20.0)
                features.append(loc.get('z', 0) / 10.0)
                
                # Add AP interface as a numeric hash (simple)
                interface = ap_info.get('interface', 'wlan0')
                # Simple hash of interface name to a number between 0 and 1
                interface_hash = sum(ord(c) for c in interface) % 100 / 100.0
                features.append(interface_hash)
            else:
                # If AP not found, add zeros for AP features
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features)
    
    def get_ap_info(self, ap_id: str) -> Optional[Dict]:
        """Get AP information by AP ID from the configuration."""
        for floor_num, floor_data in self.aps_config.items():
            for ap in floor_data.get('aps', []):
                if ap.get('id') == ap_id:
                    return ap
        return None
    
    def get_all_aps(self) -> List[Dict]:
        """Get a flat list of all APs with their floor information."""
        all_aps = []
        for floor_num, floor_data in self.aps_config.items():
            for ap in floor_data.get('aps', []):
                ap_copy = ap.copy()
                ap_copy['floor'] = int(floor_num)
                ap_copy['floor_name'] = floor_data.get('name', f'Floor {floor_num}')
                all_aps.append(ap_copy)
        return all_aps
    
    def collect_training_data(self, duration_seconds: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate collecting training data from WiFi signals across multiple APs.
        In practice, this would interface with WiFi hardware to collect BFI data.
        """
        logger.info(f"Collecting training data for {duration_seconds} seconds across {len(self.get_all_aps())} APs...")
        
        # Simulate data collection - in reality, this would read from WiFi interfaces
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
                
                # For each AP, extract features with AP-specific info
                for ap in self.get_all_aps():
                    ap_id = ap['id']
                    features = self.extract_bfi_features(bfi_sample, ap_id)
                    X.append(features)
                    y.append(person_id)
                
                # Small delay to simulate real-time collection
                time.sleep(0.01)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Collected {len(X)} samples from {n_people} people across {len(self.get_all_aps())} APs")
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train the person identification model using data from all APs.
        """
        logger.info("Training Ubiquiti multi-AP WiFi person identification model...")
        
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
            'n_samples': len(X),
            'n_aps': len(self.get_all_aps())
        }
    
    def predict(self, bfi_data: np.ndarray, ap_id: str = None) -> Tuple[int, float]:
        """
        Predict person ID from BFI data for a specific AP.
        Returns: (person_id, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features
        features = self.extract_bfi_features(bfi_data, ap_id).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(probabilities)
        
        return int(prediction), float(confidence)
    
    def predict_multi_ap(self, bfi_data_per_ap: Dict[str, np.ndarray]) -> Tuple[int, float]:
        """
        Predict person ID by aggregating predictions from multiple APs.
        bfi_data_per_ap: dict mapping ap_id to BFI data array
        Returns: (person_id, confidence) where confidence is average of AP confidences
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        confidences = []
        
        for ap_id, bfi_data in bfi_data_per_ap.items():
            try:
                person_id, confidence = self.predict(bfi_data, ap_id)
                predictions.append(person_id)
                confidences.append(confidence)
            except Exception as e:
                logger.warning(f"Failed to predict for AP {ap_id}: {e}")
                continue
        
        if not predictions:
            raise ValueError("No valid predictions from any AP")
        
        # Use majority vote for person ID
        from collections import Counter
        vote_counts = Counter(predictions)
        final_person_id = vote_counts.most_common(1)[0][0]
        
        # Average confidence (could be weighted by vote count)
        final_confidence = np.mean(confidences)
        
        return final_person_id, final_confidence
    
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

def simulate_wifi_interface(ap_id: str = None):
    """
    Simulate a WiFi interface that provides BFI data for a specific AP.
    In a real implementation, this would interface with actual WiFi hardware/drivers.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting simulated WiFi interface for AP {ap_id if ap_id else 'all'}...")
    
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

def simulate_multi_ap_interface():
    """
    Simulate receiving BFI data from multiple APs simultaneously.
    Yields a dict mapping ap_id to BFI features.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting simulated multi-AP WiFi interface...")
    
    # Create a simulator for each AP
    ap_simulators = {}
    for ap in []:  # We'll get APs from config when identifier is created
        pass
    
    # For simplicity, we'll just yield data for all known APs
    # In a real implementation, each AP would have its own interface
    while True:
        bfi_data_per_ap = {}
        # We'll need access to the identifier's AP list - this is a limitation of this approach
        # For demo, we'll simulate a few APs
        for i in range(3):  # Simulate 3 APs
            ap_id = f"ap{i+1}"
            bfi_matrix = np.random.complex128((8, 8)) * (np.random.rand(8, 8) + 1j*np.random.rand(8, 8))
            bfi_features = np.abs(bfi_matrix).flatten()
            bfi_data_per_ap[ap_id] = bfi_features
        
        yield bfi_data_per_ap
        time.sleep(0.1)

def main():
    """
    Main function to demonstrate the Ubiquiti multi-floor multi-AP WiFi person identification system
    """
    print("=" * 70)
    print("Ubiquiti Multi-Floor Multi-AP WiFi-Based Person Identification System")
    print("Based on KIT research: Ordinary WiFi can identify people")
    print("Extended for Ubiquiti AP deployment across multiple floors")
    print("=" * 70)
    
    # Initialize the identifier (will load AP config from default location)
    identifier = UbiquitiWiFiPersonIdentifier()
    
    # Show AP configuration
    print("\nAP Configuration:")
    for ap in identifier.get_all_aps():
        print(f"  AP {ap['id']} ({ap['floor_name']}): {ap['name']} at {ap['location']}")
    
    # Collect training data
    print("\n1. Collecting training data...")
    X, y = identifier.collect_training_data(duration_seconds=30)
    
    # Train the model
    print("\n2. Training model...")
    results = identifier.train(X, y)
    print(f"   Training accuracy: {results['accuracy']:.4f}")
    print(f"   Features per sample: {results['n_features']}")
    print(f"   Number of APs: {results['n_aps']}")
    
    # Save the model
    model_path = '/home/linuxuser/wifi-person-id-ubiquiti-multi/models/ubiquiti_wifi_person_id_model.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    identifier.save_model(model_path)
    print(f"\n3. Model saved to: {model_path}")
    
    # Demonstrate real-time identification
    print("\n4. Demonstrating real-time identification...")
    print("   (Press Ctrl+C to stop)")
    
    try:
        # Simulate data from multiple APs
        wifi_interface = simulate_multi_ap_interface()
        
        for i, bfi_data_per_ap in enumerate(wifi_interface):
            if i >= 5:  # Limit demo to 5 samples
                break
                
            person_id, confidence = identifier.predict_multi_ap(bfi_data_per_ap)
            print(f"   Sample {i+1}: Person ID {person_id} (confidence: {confidence:.3f})")
            print(f"     AP data keys: {list(bfi_data_per_ap.keys())}")
            
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n   Stopped by user")
    
    print("\n5. System ready for deployment!")
    print(f"   Model saved to: {model_path}")
    print("\nNote: This is a demonstration implementation.")
    print("\nFor actual Ubiquiti hardware integration, you would need:")
    print("- Ubiquiti APs that can provide BFI/CSI data (via custom firmware or telnet/ssh)")
    print("- A method to capture BFI frames from each AP (e.g., using tcpdump or custom agent)")
    print("- Real-time signal processing pipeline to extract features from BFI data")
    print("- Proper calibration for your specific multi-floor environment")
    print("- Consideration for AP handoff as people move between floors")

if __name__ == "__main__":
    main()