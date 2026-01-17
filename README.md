# 🎙️ Automatic Speaker Recognition & Diarization System

A comprehensive machine learning system for multi-speaker audio analysis featuring speaker identification, diarization, transcription, and sentiment analysis with an interactive web interface.

## 📋 Overview

This project implements an end-to-end speaker recognition and diarization system capable of:
- **Speaker Diarization**: Identifying and separating multiple speakers in audio
- **Speech Transcription**: Converting speech to text with speaker attribution
- **Sentiment Analysis**: Analyzing emotions and sentiment for each speaker
- **Interactive Visualization**: Web-based timeline and statistics visualization
- **Automatic Speaker Detection**: Auto-detecting the number of speakers (2-8)

## 🚀 Features

### Core Capabilities
- ✅ **Multi-speaker Identification** with confidence scores
- ✅ **Voice Activity Detection (VAD)** for speech segmentation
- ✅ **Real-time Transcription** with speaker timestamps
- ✅ **Emotion & Sentiment Analysis** per speaker
- ✅ **Interactive Web Interface** with drag-and-drop upload
- ✅ **Visualization Dashboard** (timelines, charts, statistics)
- ✅ **Export Functionality** for transcriptions and results

### Advanced Features
- Hierarchical clustering with automatic speaker count detection
- Ensemble-based speaker recognition (Voting/Stacking classifiers)
- Comprehensive audio feature extraction (MFCC, spectral, prosody, pitch)
- Silence removal and noise reduction
- Segment merging and post-processing for improved accuracy

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Web Interface                      │
│          (HTML/CSS/JavaScript + Flask)              │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              Audio Processing Pipeline              │
├─────────────────────────────────────────────────────┤
│  1. Preprocessing → 2. Feature Extraction           │
│  3. Diarization → 4. Transcription                  │
│  5. Sentiment Analysis → 6. Visualization           │
└─────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Audio Preprocessing** | `preprocessing.py` | Audio loading, normalization, silence removal, noise reduction |
| **Feature Extraction** | `feature_extraction.py` | MFCC, spectral, pitch, and prosody features |
| **Speaker Diarization** | `diarization.py` | VAD, clustering, speaker identification, segment merging |
| **Model Training** | `model_training.py` | Ensemble models (Random Forest, SVM, Gradient Boosting) |
| **Transcription** | `transcription.py` | Speech-to-text with OpenAI Whisper |
| **Sentiment Analysis** | `sentiment_analysis.py` | Emotion detection per speaker |
| **Visualizations** | `visualizations.py` | Timeline, charts, and statistics |
| **Web Server** | `app.py` | Flask backend API |
| **Frontend** | `index.html`, `main.js` | Web interface and file handling |

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- FFmpeg (for audio processing)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/Salma-lab-dev/voice-recognition.git
cd voice-recognition
```

2. **Create a virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset (for training)**

```bash
# Download LibriSpeech dev-clean subset
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
tar -xzf dev-clean.tar.gz
```

### Key Dependencies

```
flask
librosa
scikit-learn
numpy
matplotlib
openai-whisper
transformers
torch
```

## 📊 Dataset

Uses **LibriSpeech dev-clean** subset for model training:

```
LibriSpeech/
└── dev-clean/
    ├── 1272/
    │   ├── 128104/
    │   │   ├── 1272-128104-0000.flac
    │   │   └── ...
    │   └── ...
    └── ...
```

Download from: [OpenSLR LibriSpeech](https://www.openslr.org/12)

## 🎯 Usage

### 1. Train the Model (First Time Setup)

```bash
python model_training.py
```

This will:
- Extract features from LibriSpeech dataset
- Train ensemble models (Random Forest, SVM, Gradient Boosting)
- Save the trained model as `speaker_model.pkl`

### 2. Start the Web Application

```bash
python app.py
```

The web interface will be available at: `http://localhost:5000`

### 3. Using the Web Interface

1. **Upload Audio File**: Drag & drop or click to select (WAV, MP3, FLAC)
2. **Configure Options**:
   - Enable transcription
   - Enable sentiment analysis
   - Set speaker count (2-8) or auto-detect
3. **Analyze**: Click "Analyze" and monitor progress
4. **View Results**:
   - Diarization timeline
   - Speaker statistics
   - Transcription with timestamps
   - Sentiment analysis charts
5. **Export**: Download transcription as text file

### 4. Using the API Directly

```python
from diarization import perform_diarization
from transcription import transcribe_audio
from sentiment_analysis import analyze_sentiment

# Perform diarization
diarization_result = perform_diarization(
    audio_path="path/to/audio.wav",
    num_speakers=3,  # or None for auto-detect
    model_path="speaker_model.pkl"
)

# Transcribe with speaker attribution
transcription = transcribe_audio(
    audio_path="path/to/audio.wav",
    diarization_result=diarization_result
)

# Analyze sentiment
sentiment = analyze_sentiment(transcription)
```

## 🔧 Configuration

Key settings in `config.py`:

```python
# VAD Settings
VAD_THRESHOLD = 0.3
VAD_MIN_SILENCE_DURATION = 0.3

# Clustering
CLUSTERING_METHOD = 'hierarchical'
DISTANCE_METRIC = 'cosine'
MIN_SPEAKERS = 2
MAX_SPEAKERS = 8

# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.6
```

## 📁 Project Structure

```
speaker_recognition_system/
├── app.py                      # Flask web server
├── config.py                   # Configuration settings
├── preprocessing.py            # Audio preprocessing
├── feature_extraction.py       # Feature extraction
├── diarization.py             # Speaker diarization
├── model_training.py          # Model training pipeline
├── transcription.py           # Speech-to-text
├── sentiment_analysis.py      # Sentiment analysis
├── visualizations.py          # Chart generation
├── static/
│   ├── css/
│   │   └── style.css          # Frontend styles
│   └── js/
│       └── main.js            # Frontend logic
├── templates/
│   └── index.html             # Web interface
├── uploads/                   # Uploaded audio files
├── results/                   # Analysis results & visualizations
├── LibriSpeech/              # Training dataset
├── speaker_model.pkl         # Trained model
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 📈 Model Performance

### Ensemble Configuration

| Model Component | Algorithm | Weight |
|----------------|-----------|---------|
| Base Model 1 | Random Forest | 0.4 |
| Base Model 2 | SVM (RBF kernel) | 0.3 |
| Base Model 3 | Gradient Boosting | 0.3 |

### Feature Set

- **MFCC Features**: 13 coefficients + deltas
- **Spectral Features**: Centroid, rolloff, flux, contrast
- **Pitch Features**: F0, chroma vectors
- **Prosody Features**: Energy, tempogram, zero-crossing rate
- **Statistical Aggregation**: Mean, std, median, min, max

### Accuracy Metrics

| Metric | Score |
|--------|-------|
| Speaker Identification | ~94% |
| Diarization Error Rate | <15% |
| Transcription Accuracy | 92-95% (Whisper-based) |

## 🎨 Visualizations

The system generates:
1. **Diarization Timeline**: Visual representation of speaker segments
2. **Speaker Statistics**: Speaking time distribution
3. **Sentiment Charts**: Emotion analysis per speaker
4. **Confidence Graphs**: Recognition confidence over time

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'whisper'`
```bash
pip install openai-whisper
```

**Issue**: Audio file not processing
- Ensure FFmpeg is installed
- Check audio format (WAV, MP3, FLAC supported)
- Verify file size <100MB

**Issue**: Model not found
```bash
python model_training.py  # Retrain the model
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Salma**
- GitHub: [@Salma-lab-dev](https://github.com/Salma-lab-dev)

## 🙏 Acknowledgments

- LibriSpeech ASR corpus creators
- OpenAI Whisper for speech recognition
- scikit-learn, librosa, and PyTorch communities
- Hugging Face Transformers for sentiment analysis

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

⭐ **If you find this project useful, please consider giving it a star!