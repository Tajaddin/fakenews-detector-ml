# Fake News Detector ML

A machine learning project for detecting fake news articles using both traditional ML models and deep learning transformers. This project implements a complete ML pipeline from data preprocessing to model explainability.

## Features

- **Multiple ML Models**: Logistic Regression, SVM, Naive Bayes, Random Forest, and Gradient Boosting
- **Deep Learning**: DistilBERT transformer-based classifier
- **Ensemble Methods**: Voting, stacking, and weighted average ensembles
- **Model Explainability**: LIME and SHAP explanations for predictions
- **Interactive Demo**: Streamlit web application for real-time predictions
- **Comprehensive Evaluation**: ROC curves, PR curves, calibration plots, and confusion matrices

## Project Structure

```
fakenews-detector-ml/
├── src/                    # Core library modules
│   ├── config.py          # Configuration & hyperparameters
│   ├── data_io.py         # Data loading & processing
│   ├── features.py        # Feature extraction (TF-IDF)
│   ├── models_baseline.py # Traditional ML models
│   ├── preprocess.py      # Text preprocessing
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Model evaluation
│   └── utils.py           # Utilities & helpers
├── app/                    # Streamlit demo application
├── notebooks/              # Jupyter notebooks for EDA
├── data/                   # Data directory (not tracked)
│   ├── raw/               # Raw datasets
│   └── processed/         # Processed data
├── models/                 # Saved model files
├── experiments/            # Experiment tracking & results
├── reports/                # Generated reports & figures
│   └── figs/              # Visualization outputs
└── milestone3_*.py        # Advanced features (transformers, ensembles)
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Tajaddin/fakenews-detector-ml.git
cd fakenews-detector-ml
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Quick Start

### Using Make (Recommended)

```bash
# Setup environment
make setup

# Download and prepare data
make download
make data

# Run exploratory data analysis
make eda

# Train baseline models
make baseline

# Evaluate models
make evaluate

# Run the demo app
make demo
```

### Manual Execution

```bash
# 1. Download data
python download_data.py

# 2. Preprocess data
python -m src.preprocess

# 3. Train models
python -m src.train --model logistic_regression

# 4. Evaluate
python -m src.evaluate --model logistic_regression

# 5. Run demo
streamlit run app/demo_streamlit.py
```

## Usage

### Training a Model

```python
from src.train import Trainer
from src.config import BASELINE_CONFIG

# Initialize trainer
trainer = Trainer(model_type="logistic_regression")

# Train with cross-validation
results = trainer.train_with_cv(X_train, y_train, n_splits=5)

# Evaluate on test set
metrics = trainer.evaluate(X_test, y_test)
```

### Making Predictions

```python
from src.models_baseline import load_model
from src.features import FeatureExtractor

# Load trained model
model = load_model("models/logistic_regression_best.pkl")
feature_extractor = FeatureExtractor.load("models/tfidf_vectorizer.pkl")

# Predict
text = "Breaking news: Scientists discover new treatment"
features = feature_extractor.transform([text])
prediction = model.predict(features)
probability = model.predict_proba(features)
```

### Model Explainability

```python
from milestone3_explainability import LIMEExplainer, SHAPExplainer

# LIME explanation
lime_explainer = LIMEExplainer(model, feature_extractor)
explanation = lime_explainer.explain(text)

# SHAP explanation
shap_explainer = SHAPExplainer(model, X_train)
shap_values = shap_explainer.explain(X_test[:10])
```

## Models & Performance

### Baseline Models

| Model | Accuracy | Precision | Recall | F1 Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 0.95 | 0.94 | 0.93 | 0.93 | 0.98 |
| SVM | 0.94 | 0.93 | 0.92 | 0.92 | 0.97 |
| Naive Bayes | 0.91 | 0.89 | 0.88 | 0.88 | 0.95 |
| Random Forest | 0.93 | 0.92 | 0.91 | 0.91 | 0.96 |

### Transformer Model

| Model | Accuracy | F1 Score | AUC |
|-------|----------|----------|-----|
| DistilBERT | 0.97 | 0.96 | 0.99 |

## Dataset

The project uses a custom dataset with the following characteristics:

- **Total samples**: ~22,000 articles
- **Train/Val/Test split**: 80%/10%/10%
- **Class distribution**: ~24% fake, ~76% real
- **Features**: Article text, title, source domain, URL, tweet count

### Data Format

```json
{
  "id": "article_001",
  "text": "Full article text...",
  "title": "Article headline",
  "news_url": "https://example.com/article",
  "source_domain": "example.com",
  "tweet_num": 150,
  "label": "fake"
}
```

## Configuration

Key configuration options in `src/config.py`:

```python
# TF-IDF settings
BASELINE_CONFIG = {
    "tfidf": {
        "ngram_range": (1, 2),
        "max_features": 10000,
        "min_df": 3,
        "max_df": 0.9
    }
}

# Transformer settings
TRANSFORMER_CONFIG = {
    "model_name": "distilbert-base-uncased",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5
}
```

## Demo Application

Run the interactive Streamlit demo:

```bash
streamlit run app/demo_streamlit.py
```

Features:
- Real-time fake news prediction
- Confidence scores and probability visualization
- Model explanation with highlighted text features
- Support for multiple models

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
black src/
pylint src/
```

### Adding New Models

1. Implement the model in `src/models_baseline.py`
2. Add configuration to `src/config.py`
3. Register in the training pipeline
4. Add evaluation metrics

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for pre-trained models
- [scikit-learn](https://scikit-learn.org/) for ML algorithms
- [LIME](https://github.com/marcotcr/lime) and [SHAP](https://github.com/slundberg/shap) for explainability

## Contact

- **Author**: Tajaddin Gafarov
- **GitHub**: [@Tajaddin](https://github.com/Tajaddin)

---

*Built with Python, scikit-learn, PyTorch, and Transformers*
