#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced OCR Image Reader with Enhanced Chinese Support
Combines multiple OCR engines for best text extraction results
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import easyocr
from typing import List, Tuple, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedOCR:
    """
    Advanced OCR class that combines multiple OCR engines and preprocessing techniques
    with enhanced Chinese language support
    """
    
    def __init__(self, use_gpu: bool = False, languages: List[str] = None):
        """
        Initialize OCR engines
        
        Args:
            use_gpu (bool): Whether to use GPU acceleration for EasyOCR
            languages (List[str]): List of language codes for EasyOCR
        """
        self.use_gpu = use_gpu
        
        # Default to English and Chinese languages if none specified
        if languages is None:
            languages = [
                'en',  # English
                'ch',  # Chinese Traditional
                'zh',  # Chinese Simplified
            ]
        
        self.languages = languages
        
        # Initialize EasyOCR reader with multiple languages
        try:
            logger.info(f"Initializing EasyOCR with languages: {', '.join(languages)}")
            self.easyocr_reader = easyocr.Reader(languages, gpu=use_gpu)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR with all languages: {e}")
            # Fallback to English and Chinese
            try:
                logger.info("Falling back to English and Chinese EasyOCR")
                self.easyocr_reader = easyocr.Reader(['en', 'ch'], gpu=use_gpu)
            except Exception as e2:
                logger.warning(f"Failed to initialize EasyOCR: {e2}")
                self.easyocr_reader = None
        
        # Check if Tesseract is available and set up Chinese language support
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR available")
            
            # Check available languages
            available_langs = pytesseract.get_languages()
            logger.info(f"Available Tesseract languages: {available_langs}")
            
            # Set up Tesseract language configuration
            self.tesseract_langs = []
            if 'eng' in available_langs:
                self.tesseract_langs.append('eng')
            if 'chi_sim' in available_langs:
                self.tesseract_langs.append('chi_sim')
            if 'chi_tra' in available_langs:
                self.tesseract_langs.append('chi_tra')
            
            if self.tesseract_langs:
                self.tesseract_lang_config = '+'.join(self.tesseract_langs)
                logger.info(f"Tesseract will use languages: {self.tesseract_lang_config}")
            else:
                self.tesseract_lang_config = 'eng'  # fallback
                logger.warning("No Chinese languages found for Tesseract, using English only")
                
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            self.tesseract_lang_config = 'eng'
    
    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply various preprocessing techniques to improve OCR accuracy for Chinese text
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[np.ndarray]: List of preprocessed images
        """
        preprocessed_images = []
        
        # Original image
        preprocessed_images.append(image.copy())
        
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Simple grayscale
        preprocessed_images.append(gray)
        
        # 2. Resize image for better OCR (Chinese characters often need higher resolution)
        height, width = gray.shape
        if height < 300 or width < 300:
            scale_factor = max(300 / height, 300 / width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            resized = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            preprocessed_images.append(resized)
        
        # 3. Gaussian blur + threshold
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Smaller kernel for Chinese text
        _, thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(thresh1)
        
        # 4. Adaptive threshold with smaller block size for Chinese characters
        adaptive_thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2
        )
        preprocessed_images.append(adaptive_thresh)
        
        # 5. Morphological operations - smaller kernel for Chinese text
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
        preprocessed_images.append(morph)
        
        # 6. Contrast enhancement using PIL
        pil_image = Image.fromarray(gray)
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.5)  # Less aggressive for Chinese text
        preprocessed_images.append(np.array(enhanced))
        
        # 7. Sharpen using PIL
        sharpened = pil_image.filter(ImageFilter.SHARPEN)
        preprocessed_images.append(np.array(sharpened))
        
        # 8. Bilateral filter for noise reduction while preserving edges
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        preprocessed_images.append(bilateral)
        
        return preprocessed_images
    
    def tesseract_ocr(self, image: np.ndarray, config: str = '') -> str:
        """
        Perform OCR using Tesseract with Chinese language support
        
        Args:
            image (np.ndarray): Input image
            config (str): Tesseract configuration string
            
        Returns:
            str: Extracted text
        """
        try:
            # Use configured languages for better Chinese support
            if not config:
                config = f'--oem 3 --psm 6 -l {self.tesseract_lang_config}'
            
            text = pytesseract.image_to_string(image, config=config)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return ""
    
    def easyocr_ocr(self, image: np.ndarray) -> str:
        """
        Perform OCR using EasyOCR with enhanced result handling
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            str: Extracted text
        """
        if not self.easyocr_reader:
            return ""
        
        try:
            results = self.easyocr_reader.readtext(image, paragraph=True)
            
            # Sort results by confidence and combine text
            if results:
                # Filter results by confidence threshold
                filtered_results = [result for result in results if len(result) >= 2 and 
                                  (len(result) < 3 or result[2] > 0.1)]  # confidence > 0.1
                
                # Sort by position (top to bottom, left to right)
                if len(filtered_results) > 0 and len(filtered_results[0]) >= 1:
                    filtered_results.sort(key=lambda x: (x[0][0][1], x[0][0][0]) if x[0] else (0, 0))
                
                text = ' '.join([result[1] for result in filtered_results])
                return text.strip()
            
            return ""
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return ""
    
    def contains_chinese(self, text: str) -> bool:
        """
        Check if text contains Chinese characters
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text contains Chinese characters
        """
        for char in text:
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                return True
        return False
    
    def extract_text(self, image_path: str, output_file: Optional[str] = None) -> Dict[str, str]:
        """
        Extract text from image using multiple OCR engines and preprocessing
        
        Args:
            image_path (str): Path to the input image
            output_file (Optional[str]): Path to save the extracted text
            
        Returns:
            Dict[str, str]: Dictionary containing results from different methods
        """
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return {}
        
        logger.info(f"Processing image: {image_path}")
        
        # Preprocess image
        preprocessed_images = self.preprocess_image(image)
        
        results = {}
        best_text = ""
        best_score = 0
        
        # Try different preprocessing methods with different OCR engines
        for i, processed_img in enumerate(preprocessed_images):
            method_name = f"preprocessing_{i}"
            
            # Tesseract OCR
            tesseract_text = self.tesseract_ocr(processed_img)
            if tesseract_text:
                score = len(tesseract_text) + (50 if self.contains_chinese(tesseract_text) else 0)
                results[f"tesseract_{method_name}"] = tesseract_text
                if score > best_score:
                    best_text = tesseract_text
                    best_score = score
                    logger.info(f"New best result from Tesseract {method_name}: {len(tesseract_text)} chars, has Chinese: {self.contains_chinese(tesseract_text)}")
            
            # EasyOCR
            easyocr_text = self.easyocr_ocr(processed_img)
            if easyocr_text:
                score = len(easyocr_text) + (50 if self.contains_chinese(easyocr_text) else 0)
                results[f"easyocr_{method_name}"] = easyocr_text
                if score > best_score:
                    best_text = easyocr_text
                    best_score = score
                    logger.info(f"New best result from EasyOCR {method_name}: {len(easyocr_text)} chars, has Chinese: {self.contains_chinese(easyocr_text)}")
        
        # Store the best result
        results["best_result"] = best_text
        results["has_chinese"] = str(self.contains_chinese(best_text))
        results["best_score"] = str(best_score)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("OCR Results\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("BEST RESULT:\n")
                    f.write("-" * 20 + "\n")
                    f.write(best_text + "\n")
                    f.write(f"\nContains Chinese: {self.contains_chinese(best_text)}\n")
                    f.write(f"Score: {best_score}\n\n")
                    
                    f.write("ALL RESULTS:\n")
                    f.write("-" * 20 + "\n")
                    for method, text in results.items():
                        if method not in ["best_result", "has_chinese", "best_score"] and text.strip():
                            f.write(f"\n[{method.upper()}]\n")
                            f.write(f"Text: {text}\n")
                            f.write(f"Contains Chinese: {self.contains_chinese(text)}\n")
                            f.write(f"Length: {len(text)}\n")
                
                logger.info(f"Results saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving results: {e}")
        
        return results
    
    def batch_process(self, input_folder: str, output_folder: str):
        """
        Process multiple images in a folder
        
        Args:
            input_folder (str): Folder containing input images
            output_folder (str): Folder to save text results
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            output_file = output_path / f"{image_file.stem}_ocr.txt"
            logger.info(f"Processing: {image_file.name}")
            
            results = self.extract_text(str(image_file), str(output_file))
            if results.get("best_result"):
                has_chinese = results.get("has_chinese", "False") == "True"
                chinese_indicator = " (包含中文)" if has_chinese else ""
                print(f"✓ {image_file.name}: {len(results['best_result'])} characters extracted{chinese_indicator}")
            else:
                print(f"✗ {image_file.name}: No text extracted")

def main():
    parser = argparse.ArgumentParser(description="Advanced OCR for images with Chinese support")
    parser.add_argument("input", help="Input image file or folder")
    parser.add_argument("-o", "--output", help="Output text file or folder")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration for EasyOCR")
    parser.add_argument("--batch", action="store_true", help="Process folder of images")
    parser.add_argument("--languages", nargs='+', default=['en', 'ch', 'zh'], 
                       help="Specify languages for EasyOCR (default: en ch zh)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize OCR
    ocr = AdvancedOCR(use_gpu=args.gpu, languages=args.languages)
    
    if args.batch:
        # Batch processing
        output_folder = args.output or f"{args.input}_ocr_results"
        ocr.batch_process(args.input, output_folder)
    else:
        # Single image processing
        output_file = args.output or f"{Path(args.input).stem}_ocr.txt"
        results = ocr.extract_text(args.input, output_file)
        
        if results.get("best_result"):
            has_chinese = results.get("has_chinese", "False") == "True"
            print("OCR Results:")
            print("=" * 50)
            print(results["best_result"])
            print(f"\nContains Chinese: {has_chinese}")
            print(f"Total characters: {len(results['best_result'])}")
        else:
            print("No text could be extracted from the image")

if __name__ == "__main__":
    main()