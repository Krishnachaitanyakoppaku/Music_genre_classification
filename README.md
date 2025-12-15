# 🎵 Music Genre Classification

A complete machine learning web application that classifies music genres using audio features extracted from the GTZAN dataset. Features automatic model training, real-time audio classification, and interactive data visualization.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://krishnachaitanyakoppaku-music-genre-classification-app-mbpq83.streamlit.app/)

## 🚀 Live Demo

**[Try the Live App](https://krishnachaitanyakoppaku-music-genre-classification-app-mbpq83.streamlit.app/)** - Upload audio files and get instant genre predictions!

## ✨ Features

### 🎯 **Real-time Classification**
- Upload audio files (WAV, MP3, FLAC) for instant genre prediction
- Confidence scores and probability distributions
- Support for 10 music genres from the GTZAN dataset

### 🤖 **Advanced ML Pipeline**
- **Multiple algorithms**: Random Forest, SVM, KNN with hyperparameter optimization
- **Smart preprocessing**: Outlier handling, feature scaling, file-based splitting
- **High accuracy**: 72.71% achieved through cross-validation

### 📊 **Interactive Analytics**
- Dataset exploration with interactive visualizations
- Feature correlation analysis and statistics
- Genre distribution charts and insights

### 🎼 **Comprehensive Audio Features**
- **Spectral Features**: centroid, bandwidth, rolloff
- **Temporal Features**: RMS energy, zero-crossing rate
- **Harmonic Features**: chroma STFT
- **MFCC Features**: 20 Mel-frequency cepstral coefficients

## 🏗️ Project Structure

```
├── app.py                      # 🚀 Main application (training + Streamlit)
├── IDSE_Project.ipynb          # 📓 Original research notebook  
├── Features_3_sec.csv          # 📊 Dataset file (audio features)
├── requirements.txt            # 📦 Python dependencies
├── music_genre_model.pkl       # 🤖 Trained model (auto-generated)
└── README.md                   # 📖 This file
```

## 🎯 Model Performance

| Model | Accuracy | Key Features |
|-------|----------|--------------|
| **SVM** | **72.71%** | Best overall performance, kernel-based classification |
| **Random Forest** | 68.20% | Ensemble method, feature importance analysis |
| **KNN** | 64.60% | Instance-based learning, distance metrics |

### 🔬 Technical Approach

**1. Smart Preprocessing**
- IQR-based outlier detection and capping
- Group-wise missing value imputation
- Feature standardization with StandardScaler
- File-based train-test splitting (prevents data leakage)

**2. Feature Engineering**
- 58 comprehensive audio features
- Spectral, temporal, and MFCC characteristics
- Statistical measures (mean, variance) for each feature type

**3. Model Optimization**
- Grid search with cross-validation
- Hyperparameter tuning for each algorithm
- Automated best model selection

## 🎨 Web Application Features

### 🏠 **Home Dashboard**
- Project overview and key metrics
- Dataset statistics and information
- Performance highlights

### 🎯 **Audio Classifier**
- Drag-and-drop audio upload
- Real-time genre prediction
- Confidence scores and probability distributions
- Support for WAV, MP3, FLAC formats

### 📊 **Data Analytics**
- Interactive genre distribution charts
- Feature correlation heatmaps
- Statistical analysis and insights

### 🔧 **Model Information**
- Technical architecture details
- Performance metrics and methodology
- Model retraining capabilities

## Requirements

See `requirements.txt` for complete dependencies:
- streamlit
- pandas
- numpy  
- scikit-learn
- plotly
- librosa
- matplotlib
- seaborn

## 🚀 Quick Start

### Option 1: One-Command Launch (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification

# Install dependencies
pip install -r requirements.txt

# Launch the app (auto-trains models + starts web interface)
streamlit run app.py
```

### Option 2: Streamlit Cloud Deployment
1. **Fork this repository**
2. **Connect to Streamlit Cloud**
3. **Deploy with one click** - models train automatically!

### Option 3: Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the unified application
streamlit run app.py
```

## 📊 Dataset Information

**Source**: [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)

- **10 Music Genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- **9,990 Audio Samples**: 3-second segments with extracted features
- **58 Audio Features**: Comprehensive spectral, temporal, and MFCC characteristics

**Note**: The dataset `Features_3_sec.csv` is included in the repository for immediate use.

## Model Training

The project implements a comprehensive pipeline:

```python
# Data preprocessing
- Outlier detection and capping
- Missing value imputation
- Feature standardization
- File-based train-test split

# Model optimization
- Grid search with cross-validation
- Hyperparameter tuning for each algorithm
- Performance evaluation on test set
```

## Results

The SVM model achieved the best performance with optimized hyperparameters. Detailed results including:
- Classification reports
- Confusion matrices
- Feature importance analysis
- Cross-validation scores

Can be found in the Jupyter notebook and the accompanying report.

## Future Improvements

- **Deep Learning**: Implement CNN/RNN models for raw audio processing
- **Feature Engineering**: Explore additional audio features and transformations
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Real-time Classification**: Develop a system for live audio genre classification

## Contributing

Feel free to contribute to this project by:
- Improving model performance
- Adding new features or algorithms
- Enhancing data visualization
- Optimizing code efficiency

## License

This project is available under the MIT License.

## Acknowledgments

- GTZAN Dataset creators for providing the music genre classification dataset
- Scikit-learn community for machine learning tools
- Kaggle for hosting the dataset

## 🚀 Deployment

### 🌐 Live Application
The application is currently deployed and accessible at:
**[https://krishnachaitanyakoppaku-music-genre-classification-app-mbpq83.streamlit.app/](https://krishnachaitanyakoppaku-music-genre-classification-app-mbpq83.streamlit.app/)**

### Streamlit Cloud (Recommended)
1. **Fork this repository**
2. **Connect to [Streamlit Cloud](https://streamlit.io/cloud)**
3. **Deploy**: Models train automatically on first run
4. **Share**: Get a public URL instantly

### Local Deployment
```bash
# Clone and setup
git clone https://github.com/yourusername/music-genre-classification.git
cd music-genre-classification
pip install -r requirements.txt

# Launch application
streamlit run app.py
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## 🤝 Contributing

We welcome contributions! 

### Quick Contribution Steps
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

⭐ **Star this repository if you found it helpful!** ⭐