from langdetect import detect, DetectorFactory

# Set seed for consistent results
DetectorFactory.seed = 0

def detect_language(text):
    """
    Detects the language of the given text.

    Args:
        text (str): The text to detect language from.

    Returns:
        str: The detected language code (e.g., 'en', 'fr').
    """
    try:
        language = detect(text)
        return language
    except Exception:
        return 'en'  # Default to English if detection fails
