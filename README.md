# WiFi-Based Person Identification System

This repository contains a demonstration implementation of a system that identifies people using ordinary WiFi signals, based on research from the Karlsruher Institut für Technologie (KIT).

## Overview

The system uses Beamforming Feedback Information (BFI) from WiFi signals to create unique "views" of people based on how radio waves reflect off their bodies. By analyzing these signal patterns with machine learning, the system can identify individuals with high accuracy, even if they are not carrying an active device.

**Important Note**: This is a demonstration/simulation implementation. Actual WiFi hardware integration requires specialized drivers and access to low-level WiFi chipset data.

## Repository Structure

```
wifi-person-id/
├── src/                    # Source code
│   └── wifi_person_id.py   # Main implementation
├── models/                 # Trained models (generated after training)
├── data/                   # Data storage (for real implementations)
├── scripts/                # Utility scripts
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd wifi-person-id
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main demonstration script:
```bash
python src/wifi_person_id.py
```

The script will:
1. Generate simulated training data (representing different people)
2. Train a Random Forest classifier on the extracted features
3. Save the trained model to `models/wifi_person_id_model.pkl`
4. Demonstrate real-time identification using simulated WiFi data

## How It Works (Simulation)

In this demonstration:
1. **Data Collection**: We simulate collecting BFI-like data from multiple people
2. **Feature Extraction**: We extract statistical and frequency-domain features from the BFI data
3. **Model Training**: We train a Random Forest classifier to recognize individuals based on their feature patterns
4. **Prediction**: The model predicts person ID from new BFI samples with a confidence score

## Real-World Implementation Requirements

For an actual implementation using real WiFi hardware, you would need:

1. **Hardware**: WiFi chipsets that expose Beamforming Feedback Information (BFI) or Channel State Information (CSI)
2. **Driver Modifications**: Custom drivers or kernel modules to capture BFI/CSI frames
3. **Real-time Processing**: A pipeline to continuously extract features from incoming BFI data
4. **Calibration**: Environment-specific calibration to account for room geometry and static objects
5. **Privacy Considerations**: Appropriate notices and consent mechanisms given the surveillance implications

## Research Reference

Based on the research article:
"Ordinary WiFi can now identify people with near perfect accuracy" 
ScienceDaily, May 22, 2026
Source: Karlsruher Institut für Technologie (KIT)

## Ethical Notice

This technology raises significant privacy concerns. The ability to identify people without their knowledge or consent using everyday WiFi networks could enable covert surveillance. Any real-world implementation should:
- Include clear notification to individuals
- Obtain explicit consent where required by law
- Implement strong data protection measures
- Be subject to oversight and accountability mechanisms

## Disclaimer

This code is for educational and demonstration purposes only. 
The authors are not liable for any misuse of this technology.
Always comply with local laws and regulations regarding privacy and surveillance.