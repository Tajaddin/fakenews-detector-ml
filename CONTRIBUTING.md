# Contributing to Fake News Detector ML

Thank you for your interest in contributing to this project! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs actual behavior
4. Your environment (OS, Python version, etc.)
5. Any relevant logs or screenshots

### Suggesting Features

Feature suggestions are welcome! Please:

1. Check existing issues to avoid duplicates
2. Describe the feature and its use case
3. Explain why it would benefit the project

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Follow the coding style** (see below)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Write clear commit messages**

## Development Setup

### Prerequisites

- Python 3.9+
- pip or conda
- Git

### Setting Up Your Environment

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fakenews-detector-ml.git
cd fakenews-detector-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Formatting

We use `black` for code formatting and `pylint` for linting:

```bash
# Format code
black src/ app/ tests/

# Check linting
pylint src/
```

## Coding Style

### Python Style Guide

- Follow [PEP 8](https://pep8.org/)
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions focused and concise
- Use meaningful variable names

### Example

```python
def classify_article(text: str, model: BaselineModel) -> Tuple[str, float]:
    """
    Classify a news article as real or fake.

    Args:
        text: The article text to classify
        model: Trained classification model

    Returns:
        Tuple of (label, confidence_score)

    Raises:
        ValueError: If text is empty
    """
    if not text.strip():
        raise ValueError("Text cannot be empty")

    # Process and predict
    features = extract_features(text)
    prediction = model.predict(features)

    return prediction
```

### Commit Messages

Use clear, descriptive commit messages:

- Start with a verb in present tense: "Add", "Fix", "Update", "Remove"
- Keep the first line under 50 characters
- Add details in the body if needed

Good examples:
```
Add LIME explainability for transformer model
Fix preprocessing bug with Unicode characters
Update README with installation instructions
```

## Project Structure

When adding new features, follow the existing structure:

```
src/
├── config.py          # Configuration constants
├── data_io.py         # Data loading/saving
├── features.py        # Feature extraction
├── models_baseline.py # ML model implementations
├── preprocess.py      # Text preprocessing
├── train.py           # Training pipeline
├── evaluate.py        # Evaluation metrics
└── utils.py           # Utility functions
```

## Adding New Models

To add a new classification model:

1. Add the model class in `src/models_baseline.py`
2. Add configuration parameters in `src/config.py`
3. Register the model in the training pipeline
4. Add tests for the new model
5. Update the README with model information

## Testing Guidelines

- Write unit tests for new functions
- Include edge cases and error conditions
- Use meaningful test names
- Mock external dependencies

Example:

```python
def test_preprocess_removes_urls():
    """Test that URLs are properly removed from text."""
    preprocessor = TextPreprocessor(remove_urls=True)
    text = "Check this link: https://example.com for more info"
    result = preprocessor.process(text)
    assert "https://" not in result
    assert "example.com" not in result
```

## Documentation

- Update docstrings when modifying functions
- Update README for significant changes
- Add inline comments for complex logic
- Include examples in docstrings

## Review Process

1. All PRs require at least one review
2. Address all review comments
3. Ensure CI checks pass
4. Squash commits if requested

## Getting Help

- Open an issue for questions
- Tag maintainers for urgent matters
- Check existing issues and documentation first

## Recognition

Contributors will be recognized in the README and release notes.

Thank you for contributing!
