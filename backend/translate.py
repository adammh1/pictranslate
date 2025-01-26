import sys
import re
import os
import logging
from PIL import Image
import cv2
import easyocr
import numpy as np
import aiohttp
import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

os.environ['PYTHONIOENCODING'] = 'utf-8'

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("translation.log", encoding="utf-8"),
        logging.StreamHandler(stream=sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

LANGUAGE_MAPPING = { 
    "English": {"easyocr": "en", "helsinki": "en", "facebook": "en_XX"},
    "French": {"easyocr": "fr", "helsinki": "fr", "facebook": "fr_XX"},
    "Spanish": {"easyocr": "es", "helsinki": "es", "facebook": "es_XX"},
    "Arabic": {"easyocr": "ar", "helsinki": "ar", "facebook": "ar_AR"},
    "German": {"easyocr": "de", "helsinki": "de", "facebook": "de_DE"},
    "Chinese": {"easyocr": "ch_sim", "helsinki": "zh", "facebook": "zh_CN"},
    "Russian": {"easyocr": "ru", "helsinki": "ru", "facebook": "ru_RU"}, 
    "Italian": {"easyocr": "it", "helsinki": "it", "facebook": "it_IT"}, 
    "Portuguese": {"easyocr": "pt", "helsinki": "pt", "facebook": "pt_XX"}, 
    "Japanese": {"easyocr": "ja", "helsinki": "ja", "facebook": "ja_XX"}
}

HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt"
HUGGINGFACE_API_TOKEN = os.getenv("Hugging_API_KEY")


def extract_text_from_image(image_path, language_code):
    """Extract text from an image using EasyOCR."""
    try:
        logger.info(f"Extracting text using EasyOCR with language: {language_code}")
        reader = easyocr.Reader([language_code])
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur = cv2.GaussianBlur(thresh, (5, 5), 0)
        cv2.imwrite("processed_image.jpg", blur)  
        
        result = [text for text in reader.readtext(blur) if text[2] > 0.7]
        extracted_text = " ".join([text[1] for text in result])
        
        if not extracted_text.strip():
            logger.warning("No text detected in the image.")
        return extracted_text.strip()
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        raise


def clean_extracted_text(text):
    """Clean extracted text by removing unnecessary characters."""
    try:
        logger.info("Cleaning extracted text.")
        text = re.sub(r"[^\w\s.,?!äöüÄÖÜß-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.error(f"Error during text cleaning: {e}")
        raise


async def query(payload):
    """Query the Hugging Face API asynchronously."""
    try:
        headers = {
            "Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(HUGGINGFACE_API_URL, headers=headers, json=payload) as response:
                response.raise_for_status()
                return await response.json()
    except Exception as e:
        logger.error(f"Error during translation API request: {e}")
        raise


async def translate_text(text, source_language, target_language):
    """Translate text using the Hugging Face API with full language names asynchronously."""
    try:
        logger.info(f"Translating text from {source_language} to {target_language} using Hugging Face API.")
        normalized_text = " ".join(text.split()).strip()
        payload = {
            "inputs": normalized_text.lower(),
            "parameters": {
                "src_lang": source_language,
                "tgt_lang": target_language
            }
        }

        logger.debug(f"Payload being sent to API: {payload}")
        response = await query(payload)

        logger.debug(f"API response: {response}")
        if isinstance(response, list) and len(response) > 0:
            translated_text = response[0].get("translation_text", "")
            if not translated_text:
                logger.warning("Translation not found in response.")
            return translated_text
        else:
            logger.error("Invalid response format or empty response received.")
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise


async def main(image_path, source_language, target_language):
    """Main workflow for text extraction and translation asynchronously."""
    try:
        if source_language.capitalize() not in LANGUAGE_MAPPING or target_language.capitalize() not in LANGUAGE_MAPPING:
            logger.error("Unsupported language provided.")
            return None
        
        source_lang_code = LANGUAGE_MAPPING[source_language.capitalize()]["easyocr"]
        source_helsinki_code = LANGUAGE_MAPPING[source_language.capitalize()]["facebook"]
        target_helsinki_code = LANGUAGE_MAPPING[target_language.capitalize()]["facebook"]
        
        extracted_text = extract_text_from_image(image_path, source_lang_code)
        logger.info(f"Extracted text: {extracted_text.encode('utf-8', 'ignore').decode('utf-8')}")
        if not extracted_text:
            return None
        
        cleaned_text = clean_extracted_text(extracted_text)
        logger.info(f"Cleaned text: {cleaned_text}")
        
        translated_text = await translate_text(cleaned_text, source_helsinki_code, target_helsinki_code)
        logger.info(f"Translated text: {translated_text}")
        
        result = {
            "originalText": cleaned_text,
            "translatedText": translated_text
        }
        print(json.dumps(result)) 
        return result
    except Exception as e:
        logger.error(f"Error in main workflow: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python translate.py <image_path> <source_language> <target_language>")
        sys.exit(1)

    image_path = sys.argv[1]
    source_language = sys.argv[2]
    target_language = sys.argv[3]
    asyncio.run(main(image_path, source_language, target_language))
