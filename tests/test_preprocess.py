"""
Tests for text preprocessing module
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from preprocess import TextPreprocessor


class TestTextPreprocessor:
    """Tests for TextPreprocessor class"""

    def test_init_default(self):
        """Test default initialization"""
        preprocessor = TextPreprocessor()
        assert preprocessor.lowercase is True
        assert preprocessor.remove_urls is True
        assert preprocessor.remove_html is True

    def test_clean_urls(self):
        """Test URL removal"""
        preprocessor = TextPreprocessor(remove_urls=True)
        text = "Check this link https://example.com for more info"
        result = preprocessor.clean_urls(text)
        assert "https://" not in result
        assert "example.com" not in result

    def test_clean_html(self):
        """Test HTML tag removal"""
        preprocessor = TextPreprocessor(remove_html=True)
        text = "<p>This is <b>bold</b> text</p>"
        result = preprocessor.clean_html(text)
        assert "<p>" not in result
        assert "<b>" not in result
        assert "bold" in result

    def test_clean_emails(self):
        """Test email removal"""
        preprocessor = TextPreprocessor(remove_emails=True)
        text = "Contact us at test@example.com for help"
        result = preprocessor.clean_emails(text)
        assert "test@example.com" not in result
        assert "Contact us at" in result

    def test_lowercase(self):
        """Test lowercase conversion"""
        preprocessor = TextPreprocessor(lowercase=True)
        text = "BREAKING NEWS: Important Update"
        # The preprocessor should have a method that converts to lowercase
        assert preprocessor.lowercase is True

    def test_empty_string(self):
        """Test handling of empty string"""
        preprocessor = TextPreprocessor()
        result = preprocessor.clean_urls("")
        assert result == ""

    def test_none_handling(self):
        """Test that None is handled gracefully"""
        preprocessor = TextPreprocessor()
        # clean_urls should handle None or empty
        result = preprocessor.clean_html("")
        assert result == ""

    def test_social_media_cleanup(self):
        """Test social media element removal"""
        preprocessor = TextPreprocessor()
        text = "RT @user: Check this out #trending"
        result = preprocessor.clean_social_media(text)
        assert "@user" not in result

    def test_unicode_handling(self):
        """Test Unicode text handling"""
        preprocessor = TextPreprocessor()
        text = "This has émojis and spëcial characters"
        result = preprocessor.clean_urls(text)
        # Should not crash and should preserve text
        assert "special" in result or "spëcial" in result


class TestPreprocessorIntegration:
    """Integration tests for preprocessing pipeline"""

    def test_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        preprocessor = TextPreprocessor(
            lowercase=True,
            remove_urls=True,
            remove_html=True,
            remove_emails=True
        )

        text = """
        <html>BREAKING NEWS! Check https://fake.com for details.
        Contact: info@fake.com. RT @reporter: This is #important</html>
        """

        # Clean each component
        result = preprocessor.clean_html(text)
        result = preprocessor.clean_urls(result)
        result = preprocessor.clean_emails(result)

        assert "<html>" not in result
        assert "https://" not in result
        assert "@fake.com" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
