#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChatGPT Vision API OCR Example
Uses OpenAI's GPT-4 Vision to extract text from images
"""

import openai
import base64
import requests
from pathlib import Path
import json
from typing import Optional, Dict
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatGPTVisionOCR:
    """
    OCR using OpenAI's GPT-4 Vision API
    """
    
    def __init__(self, api_key: str):
        """
        Initialize with OpenAI API key
        
        Args:
            api_key (str): OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
    
    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64
        
        Args:
            image_path (str): Path to image file
            
        Returns:
            str: Base64 encoded image
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def extract_text_from_image(self, image_path: str, prompt: Optional[str] = None) -> Dict:
        """
        Extract text from image using GPT-4 Vision
        
        Args:
            image_path (str): Path to the image file
            prompt (str): Custom prompt for OCR (optional)
            
        Returns:
            Dict: Response containing extracted text and metadata
        """
        try:
            # Default OCR prompt
            if prompt is None:
                prompt = """Please extract all text from this image. 
                Return the text exactly as it appears, maintaining the original formatting and layout.
                If the image contains text in multiple languages (especially Chinese and English), 
                please extract all of it accurately.
                
                Format your response as:
                EXTRACTED TEXT:
                [the actual text here]
                
                LANGUAGE DETECTED:
                [languages found in the image]
                
                CONFIDENCE:
                [your confidence level: High/Medium/Low]"""
            
            # Get the base64 string
            base64_image = self.encode_image(image_path)
            
            # Determine image format
            image_format = Path(image_path).suffix.lower()
            if image_format == '.jpg':
                image_format = '.jpeg'
            
            mime_type = f"image/{image_format[1:]}"  # Remove the dot
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # or "gpt-4-vision-preview"
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                    "detail": "high"  # Use "high" for better OCR accuracy
                                }
                            }
                        ]
                    }
                ],
                max_tokens=4000,
                temperature=0  # Use 0 for consistent OCR results
            )
            
            result = {
                "success": True,
                "text": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": response.model,
                "image_path": image_path
            }
            
            logger.info(f"OCR completed successfully. Tokens used: {result['usage']['total_tokens']}")
            return result
            
        except Exception as e:
            logger.error(f"Error during OCR: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "image_path": image_path
            }
    
    def extract_structured_text(self, image_path: str) -> Dict:
        """
        Extract text with structured output (JSON format)
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Dict: Structured response with extracted text
        """
        prompt = """Please extract all text from this image and return it in the following JSON format:

        {
            "extracted_text": "the complete text as it appears in the image",
            "languages_detected": ["list", "of", "languages"],
            "text_blocks": [
                {
                    "text": "text content",
                    "position": "description of where this text appears (e.g., 'top-left', 'center', 'bottom')",
                    "language": "detected language for this block"
                }
            ],
            "confidence": "High/Medium/Low",
            "notes": "any additional observations about the text or image"
        }
        
        Make sure to extract ALL text visible in the image, including any Chinese characters, English text, numbers, or other scripts."""
        
        result = self.extract_text_from_image(image_path, prompt)
        
        if result["success"]:
            try:
                # Try to parse the JSON response
                text_content = result["text"]
                # Find JSON content (sometimes GPT adds extra text before/after JSON)
                start_idx = text_content.find('{')
                end_idx = text_content.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = text_content[start_idx:end_idx]
                    parsed_json = json.loads(json_str)
                    result["structured_data"] = parsed_json
                    logger.info("Successfully parsed structured response")
                else:
                    logger.warning("Could not find JSON in response")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Could not parse JSON response: {e}")
                # Keep the original text response
        
        return result
    
    def batch_ocr(self, image_folder: str, output_folder: str = None):
        """
        Process multiple images in a folder
        
        Args:
            image_folder (str): Folder containing images
            output_folder (str): Folder to save results (optional)
        """
        input_path = Path(image_folder)
        if output_folder:
            output_path = Path(output_folder)
            output_path.mkdir(exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        results = []
        total_tokens = 0
        
        for image_file in image_files:
            logger.info(f"Processing: {image_file.name}")
            
            result = self.extract_text_from_image(str(image_file))
            results.append(result)
            
            if result["success"]:
                total_tokens += result["usage"]["total_tokens"]
                print(f"✓ {image_file.name}: Text extracted successfully")
                
                # Save individual result if output folder specified
                if output_folder:
                    output_file = output_path / f"{image_file.stem}_chatgpt_ocr.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(f"ChatGPT Vision OCR Results\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(f"Image: {image_file.name}\n")
                        f.write(f"Model: {result['model']}\n")
                        f.write(f"Tokens used: {result['usage']['total_tokens']}\n\n")
                        f.write("EXTRACTED TEXT:\n")
                        f.write("-" * 20 + "\n")
                        f.write(result["text"])
                        
            else:
                print(f"✗ {image_file.name}: {result['error']}")
        
        # Save summary
        if output_folder:
            summary_file = output_path / "batch_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "total_images": len(image_files),
                    "successful": len([r for r in results if r["success"]]),
                    "failed": len([r for r in results if not r["success"]]),
                    "total_tokens_used": total_tokens,
                    "results": results
                }, f, indent=2, ensure_ascii=False)
        
        print(f"\nBatch processing complete. Total tokens used: {total_tokens}")
        return results

def main():
    """
    Example usage of ChatGPT Vision OCR
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="ChatGPT Vision OCR")
    parser.add_argument("image", help="Input image file or folder")
    parser.add_argument("--output", help="Output file or folder")
    parser.add_argument("--batch", action="store_true", help="Process folder of images")
    parser.add_argument("--structured", action="store_true", help="Use structured JSON output")
    parser.add_argument("--prompt", help="Custom prompt for OCR")
    
    args = parser.parse_args()
    
    # Initialize OCR
    ocr = ChatGPTVisionOCR(os.getenv("OPENAI_API_KEY"))
    
    if args.batch:
        # Batch processing
        output_folder = args.output or f"{args.image}_chatgpt_ocr_results"
        ocr.batch_ocr(args.image, output_folder)
    else:
        # Single image processing
        if args.structured:
            result = ocr.extract_structured_text(args.image)
        else:
            result = ocr.extract_text_from_image(args.image, args.prompt)
        
        if result["success"]:
            print("ChatGPT Vision OCR Results:")
            print("=" * 50)
            print(result["text"])
            print(f"\nTokens used: {result['usage']['total_tokens']}")
            
            # Save to file if specified
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result["text"])
                print(f"Results saved to: {args.output}")
                
        else:
            print(f"OCR failed: {result['error']}")

# Example usage functions
def example_simple_ocr():
    """Simple OCR example"""
    api_key = "your-openai-api-key-here"
    ocr = ChatGPTVisionOCR(api_key)
    
    result = ocr.extract_text_from_image("path/to/your/image.jpg")
    
    if result["success"]:
        print("Extracted text:")
        print(result["text"])
        print(f"Tokens used: {result['usage']['total_tokens']}")
    else:
        print(f"Error: {result['error']}")

def example_structured_ocr():
    """Structured OCR example"""
    api_key = "your-openai-api-key-here"
    ocr = ChatGPTVisionOCR(api_key)
    
    result = ocr.extract_structured_text("path/to/your/image.jpg")
    
    if result["success"] and "structured_data" in result:
        data = result["structured_data"]
        print("Extracted text:", data["extracted_text"])
        print("Languages detected:", data["languages_detected"])
        print("Confidence:", data["confidence"])
    else:
        print("Raw response:", result["text"])

def example_custom_prompt():
    """Custom prompt example"""
    api_key = os.getenv("OPENAI_API_KEY")
    ocr = ChatGPTVisionOCR(api_key)
    
    custom_prompt = """
    This image contains a document with both English and Chinese text.
    Please extract all text and organize it as follows:
    
    ENGLISH TEXT:
    [all English text here]
    
    CHINESE TEXT:
    [all Chinese text here]
    
    NUMBERS/DATES:
    [any numbers or dates found]
    """
    
    result = ocr.extract_text_from_image(r"C:\Projects\emptytest\data\logo.png", custom_prompt)
    
    if result["success"]:
        print(result["text"])

if __name__ == "__main__":
    main()