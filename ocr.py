import easyocr
import numpy as np
import cv2


# Load reader once (important for performance)
reader = easyocr.Reader(["en"], gpu=False)


def extract_text_from_image(image_input, min_confidence=0.4):
    """
    Extract text from image using EasyOCR.

    Args:
        image_input: image file path OR image bytes
        min_confidence (float): filter low-confidence text

    Returns:
        str: extracted text
    """

    try:
        # If input is bytes (Streamlit upload)
        if isinstance(image_input, bytes):
            np_arr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            image = image_input  # assume it's path

        results = reader.readtext(image)

        filtered_text = [
            text for (_, text, conf) in results if conf >= min_confidence
        ]

        return " ".join(filtered_text).strip()

    except Exception as e:
        return f"OCR Error: {str(e)}"
