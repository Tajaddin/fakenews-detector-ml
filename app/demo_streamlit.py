"""
Streamlit Demo Application for Fake News Detection
Interactive web interface for classifying news articles
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, Any
import re

# Import project modules
try:
    from src.features import FeatureExtractor, create_baseline_features
    from src.models_baseline import BaselineModel
    from src.preprocess import TextPreprocessor
    from src.config import MODELS_DIR, RANDOM_SEED
except ImportError:
    from features import FeatureExtractor, create_baseline_features
    from models_baseline import BaselineModel
    from preprocess import TextPreprocessor
    from config import MODELS_DIR, RANDOM_SEED


# Page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .real-news {
        background-color: #D1FAE5;
        border: 2px solid #10B981;
    }
    .fake-news {
        background-color: #FEE2E2;
        border: 2px solid #EF4444;
    }
    .confidence-meter {
        height: 20px;
        border-radius: 10px;
        background: linear-gradient(to right, #EF4444, #F59E0B, #10B981);
    }
</style>
""", unsafe_allow_html=True)


class FakeNewsDetector:
    """Fake News Detection Pipeline for Streamlit"""

    def __init__(self):
        self.model: Optional[BaselineModel] = None
        self.feature_extractor: Optional[FeatureExtractor] = None
        self.preprocessor: Optional[TextPreprocessor] = None
        self.is_loaded = False

    def load_models(self, model_type: str = "logistic_regression") -> bool:
        """Load trained model and feature extractor"""
        try:
            model_path = MODELS_DIR / f"{model_type}_model.pkl"
            feature_path = MODELS_DIR / "feature_extractor.pkl"

            if model_path.exists() and feature_path.exists():
                self.model = BaselineModel.load(model_path)
                self.feature_extractor = FeatureExtractor.load(feature_path)
                self.preprocessor = TextPreprocessor(
                    lowercase=True,
                    remove_urls=True,
                    remove_html=True,
                    normalize_whitespace=True
                )
                self.is_loaded = True
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False

    def preprocess_text(self, text: str) -> str:
        """Preprocess input text"""
        if self.preprocessor:
            return self.preprocessor.process(text)
        return text.lower().strip()

    def predict(self, text: str) -> Tuple[str, float, np.ndarray]:
        """
        Make prediction on input text

        Returns:
            Tuple of (label, confidence, probabilities)
        """
        if not self.is_loaded:
            raise RuntimeError("Models not loaded")

        # Preprocess
        processed_text = self.preprocess_text(text)

        # Create DataFrame for feature extraction
        df = pd.DataFrame({
            'text': [text],
            'text_processed': [processed_text]
        })

        # Extract features
        features = self.feature_extractor.transform(df)

        # Predict
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]

        label = "Real" if prediction == 0 else "Fake"
        confidence = probabilities[prediction]

        return label, confidence, probabilities

    def get_important_features(self, text: str, top_k: int = 10) -> pd.DataFrame:
        """Get important features for prediction explanation"""
        if not self.is_loaded:
            return pd.DataFrame()

        processed_text = self.preprocess_text(text)
        df = pd.DataFrame({
            'text': [text],
            'text_processed': [processed_text]
        })

        features = self.feature_extractor.transform(df)
        feature_names = self.feature_extractor.get_feature_names()

        # Get non-zero features
        non_zero_indices = features.nonzero()[1]
        feature_values = features.toarray()[0]

        # Create DataFrame
        important_features = []
        for idx in non_zero_indices:
            if idx < len(feature_names):
                important_features.append({
                    'feature': feature_names[idx],
                    'value': feature_values[idx]
                })

        if important_features:
            df_features = pd.DataFrame(important_features)
            df_features = df_features.sort_values('value', ascending=False).head(top_k)
            return df_features

        return pd.DataFrame()


def create_demo_prediction() -> Tuple[str, float, np.ndarray]:
    """Create demo prediction when models aren't available"""
    # Simple heuristic-based demo
    return "Real", 0.75, np.array([0.75, 0.25])


def simple_analysis(text: str) -> Dict[str, Any]:
    """Perform simple text analysis without trained models"""
    words = text.split()

    # Count suspicious patterns
    suspicious_patterns = [
        r'BREAKING',
        r'SHOCKING',
        r'YOU WON\'T BELIEVE',
        r'EXPOSED',
        r'SECRET',
        r'CONSPIRACY',
        r'THEY DON\'T WANT YOU TO KNOW',
        r'WAKE UP',
        r'!!!',
        r'\?\?\?',
        r'100%',
        r'EXPOSED',
    ]

    suspicion_score = 0
    found_patterns = []

    for pattern in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            suspicion_score += 1
            found_patterns.append(pattern)

    # Check for excessive caps
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3:
        suspicion_score += 1
        found_patterns.append("Excessive capitalization")

    # Check for excessive punctuation
    punct_count = sum(1 for c in text if c in '!?')
    if punct_count > 5:
        suspicion_score += 1
        found_patterns.append("Excessive punctuation")

    # Normalize score
    fake_probability = min(suspicion_score / 5, 1.0)
    real_probability = 1 - fake_probability

    label = "Fake" if fake_probability > 0.5 else "Real"
    confidence = max(fake_probability, real_probability)

    return {
        'label': label,
        'confidence': confidence,
        'probabilities': np.array([real_probability, fake_probability]),
        'patterns': found_patterns,
        'word_count': len(words),
        'char_count': len(text)
    }


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<p class="main-header">Fake News Detector</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered detection of misinformation using machine learning</p>', unsafe_allow_html=True)

    # Initialize detector
    detector = FakeNewsDetector()

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Model selection
        model_options = ["logistic_regression", "svm", "naive_bayes", "complement_nb"]
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            format_func=lambda x: x.replace("_", " ").title()
        )

        # Try to load models
        models_loaded = detector.load_models(selected_model)

        if models_loaded:
            st.success("Models loaded successfully!")
        else:
            st.warning("Trained models not found. Using demo mode with heuristic analysis.")
            st.info("To use trained models, run the training pipeline first.")

        st.divider()

        # Info section
        st.header("About")
        st.markdown("""
        This application uses machine learning to detect potentially fake news articles.

        **Features:**
        - TF-IDF text vectorization
        - Multiple ML classifiers
        - Confidence scoring
        - Feature importance analysis

        **Disclaimer:**
        This tool is for educational purposes.
        Always verify news through multiple credible sources.
        """)

        st.divider()

        # Sample texts
        st.header("Try Sample Texts")

        sample_real = """Scientists at MIT have developed a new solar panel technology
        that increases energy efficiency by 15%. The research, published in Nature Energy,
        describes how the new photovoltaic cells use a novel silicon structure to capture
        more sunlight. The technology is expected to be commercially available within 5 years."""

        sample_fake = """BREAKING!!! Government EXPOSED hiding ALIENS!!!
        SECRET documents LEAKED prove EVERYTHING!!! They DON'T want you to KNOW!!!
        Share before this gets DELETED!!! 100% TRUE!!!"""

        if st.button("Load Real News Sample"):
            st.session_state['input_text'] = sample_real

        if st.button("Load Fake News Sample"):
            st.session_state['input_text'] = sample_fake

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Enter News Article")

        # Get input text
        default_text = st.session_state.get('input_text', '')
        text_input = st.text_area(
            "Paste your news article or headline here:",
            value=default_text,
            height=200,
            placeholder="Enter the news article text you want to analyze..."
        )

        analyze_button = st.button("Analyze", type="primary", use_container_width=True)

    with col2:
        st.header("Quick Stats")
        if text_input:
            word_count = len(text_input.split())
            char_count = len(text_input)
            sentence_count = len(re.split(r'[.!?]+', text_input))

            st.metric("Words", word_count)
            st.metric("Characters", char_count)
            st.metric("Sentences", sentence_count)

    # Analysis results
    if analyze_button and text_input:
        st.divider()
        st.header("Analysis Results")

        with st.spinner("Analyzing text..."):
            if detector.is_loaded:
                # Use trained model
                try:
                    label, confidence, probabilities = detector.predict(text_input)
                    analysis = {
                        'label': label,
                        'confidence': confidence,
                        'probabilities': probabilities,
                        'patterns': [],
                        'word_count': len(text_input.split()),
                        'char_count': len(text_input)
                    }
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
                    analysis = simple_analysis(text_input)
            else:
                # Use heuristic analysis
                analysis = simple_analysis(text_input)

        # Display results
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            # Prediction box
            if analysis['label'] == "Real":
                st.markdown(f"""
                <div class="prediction-box real-news">
                    <h2 style="color: #059669; margin: 0;">Likely Real News</h2>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                        Confidence: {analysis['confidence']*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box fake-news">
                    <h2 style="color: #DC2626; margin: 0;">Likely Fake News</h2>
                    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                        Confidence: {analysis['confidence']*100:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            # Probability chart
            fig = go.Figure(go.Bar(
                x=['Real', 'Fake'],
                y=analysis['probabilities'] * 100,
                marker_color=['#10B981', '#EF4444'],
                text=[f"{p*100:.1f}%" for p in analysis['probabilities']],
                textposition='outside'
            ))

            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                showlegend=False,
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        with result_col2:
            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=analysis['confidence'] * 100,
                title={'text': "Confidence Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#3B82F6"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FEE2E2"},
                        {'range': [50, 75], 'color': "#FEF3C7"},
                        {'range': [75, 100], 'color': "#D1FAE5"}
                    ],
                    'threshold': {
                        'line': {'color': "#1F2937", 'width': 4},
                        'thickness': 0.75,
                        'value': analysis['confidence'] * 100
                    }
                }
            ))

            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Suspicious patterns (for heuristic mode)
            if analysis['patterns']:
                st.subheader("Detected Patterns")
                for pattern in analysis['patterns']:
                    st.markdown(f"- {pattern}")

        # Feature importance (if models loaded)
        if detector.is_loaded:
            st.subheader("Important Features")
            try:
                important_features = detector.get_important_features(text_input)
                if not important_features.empty:
                    fig_features = px.bar(
                        important_features,
                        x='value',
                        y='feature',
                        orientation='h',
                        title="Top Features Contributing to Prediction"
                    )
                    fig_features.update_layout(height=400)
                    st.plotly_chart(fig_features, use_container_width=True)
            except Exception as e:
                st.info("Feature importance not available for this prediction.")

        # Disclaimer
        st.divider()
        st.markdown("""
        <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 0.5rem; border: 1px solid #F59E0B;">
            <strong>Disclaimer:</strong> This tool provides an AI-based assessment and should not be the sole
            basis for determining the credibility of news. Always cross-reference with multiple reliable sources
            and use critical thinking when evaluating news content.
        </div>
        """, unsafe_allow_html=True)

    elif analyze_button and not text_input:
        st.warning("Please enter some text to analyze.")

    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.875rem;">
        Built with Streamlit | Powered by Machine Learning
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
