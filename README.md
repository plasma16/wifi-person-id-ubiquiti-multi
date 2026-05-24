# WiFi-Based Person Identification System (Ubiquiti Multi-Floor Multi-AP Edition)

This repository contains a demonstration implementation of a system that identifies people using ordinary WiFi signals, based on research from the Karlsruher Institut für Technologie (KIT). This edition extends the base system to support multiple Ubiquiti access points across multiple floors.

## Overview

The system uses Beamforming Feedback Information (BFI) from WiFi signals to create unique "views" of people based on how radio waves reflect off their bodies. By analyzing these signal patterns with machine learning, the system can identify individuals with high accuracy, even if they are not carrying an active device.

**Important Note**: This is a demonstration/simulation implementation. Actual WiFi hardware integration requires specialized drivers and access to low-level WiFi chipset data.

## Repository Structure

```
wifi-person-id-ubiquiti-multi/
├── src/                    # Original source code (single AP)
│   └── wifi_person_id.py   # Main implementation (single AP)
├── ubiquiti/               # Ubiquiti multi-AP extension
│   ├── config/             # Configuration files for Ubiquiti APs
│   │   └── ubiquiti_aps.json   # AP layout configuration
│   ├── src/                # Source code for Ubiquiti multi-AP system
│   │   └── ubiquiti_wifi_person_id.py   # Main implementation (multi-AP)
│   └── scripts/            # Utility scripts for Ubiquiti deployment
├── models/                 # Trained models (generated after training)
├── data/                   # Data storage (for real implementations)
├── scripts/                # Utility scripts (general)
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd wifi-person-id-ubiquiti-multi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Single AP (Original)
Run the original demonstration script:
```bash
python src/wifi_person_id.py
```

### Ubiquiti Multi-AP
Run the Ubiquiti multi-AP demonstration script:
```bash
python ubiquiti/src/ubiquiti_wifi_person_id.py
```

## How It Works (Simulation)

In this demonstration:

1. **Data Collection**: We simulate collecting BFI-like data from multiple people across multiple APs.
2. **Feature Extraction**: We extract statistical and frequency-domain features from the BFI data, including AP-specific location features.
3. **Model Training**: We train a Random Forest classifier to recognize individuals based on their feature patterns across all APs.
4. **Prediction**: The model predicts person ID from new BFI samples, with options for single-AP or multi-AP aggregation.

## Real-World Implementation Requirements

For an actual implementation using real Ubiquiti WiFi hardware, you would need:

1. **Hardware**: Ubiquiti access points that can provide Beamforming Feedback Information (BFI) or Channel State Information (CSI) (may require custom firmware or shell access).
2. **Data Collection**: A method to capture BFI frames from each AP (e.g., using shell scripts, custom agents, or integrating with Ubiquiti's API if available).
3. **Real-time Processing**: A pipeline to continuously extract features from incoming BFI data from multiple APs.
4. **Calibration**: Environment-specific calibration to account for multi-floor geometry, AP placement, and static objects.
5. **Privacy Considerations**: Appropriate notices and consent mechanisms given the surveillance implications.

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