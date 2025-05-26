import cv2
import base64
import requests
import json
from datetime import datetime
import os

class CameraTranslator:
    def __init__(self, api_key, target_language="English"):
        """
        Initialize the camera translator
        
        Args:
            api_key (str): Your OpenAI API key
            target_language (str): Target language for translation (default: English)
        """
        self.api_key = api_key
        self.target_language = target_language
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def capture_screenshot(self, camera_index=0, save_image=True):
        """
        Capture a screenshot from the camera
        
        Args:
            camera_index (int): Camera index (0 for default camera)
            save_image (bool): Whether to save the captured image
            
        Returns:
            str: Base64 encoded image or None if capture failed
        """
        try:
            # Initialize camera
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                print(f"Error: Could not open camera {camera_index}")
                return None
            
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                cap.release()
                return None
            
            # Save image if requested
            if save_image:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"./img/camera_capture_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Image saved as: {filename}")
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Release camera
            cap.release()
            
            return image_base64
            
        except Exception as e:
            print(f"Error capturing screenshot: {e}")
            return None
    
    def translate_image(self, image_base64):
        """
        Send image to GPT-4 Vision for translation
        
        Args:
            image_base64 (str): Base64 encoded image
            
        Returns:
            str: Translation result or None if failed
        """
        try:
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please identify and translate any text you see in this image to {self.target_language}. If there are multiple languages, translate them all. Provide the original text and its translation."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 500
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                print(f"API Error: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Error calling GPT-4 Vision API: {e}")
            return None
    
    def capture_and_translate(self, camera_index=0, save_image=True):
        """
        Capture screenshot and translate in one step
        
        Args:
            camera_index (int): Camera index
            save_image (bool): Whether to save the captured image
            
        Returns:
            str: Translation result
        """
        print("Capturing screenshot from camera...")
        image_base64 = self.capture_screenshot(camera_index, save_image)
        
        if not image_base64:
            return "Failed to capture screenshot"
        
        print("Sending image to GPT-4 Vision for translation...")
        translation = self.translate_image(image_base64)
        
        if translation:
            return translation
        else:
            return "Failed to get translation"

def main():
    # Configuration
    API_KEY = os.getenv("OPENAI_API_KEY")  # Replace with your actual API key
    TARGET_LANGUAGE = "English"  # Change to your desired target language
    
    # Check if API key is set
    if API_KEY == "your-openai-api-key-here":
        print("Please set your OpenAI API key in the API_KEY variable")
        return
    
    # Initialize translator
    translator = CameraTranslator(API_KEY, TARGET_LANGUAGE)
    
    try:
        # Capture and translate
        result = translator.capture_and_translate(
            camera_index=0,  # Use default camera
            save_image=True   # Save captured image
        )
        
        print("\n" + "="*50)
        print("TRANSLATION RESULT:")
        print("="*50)
        print(result)
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()