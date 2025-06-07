import os
import base64
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
from typing import Optional

load_dotenv()

def read_and_analyze_image(image_path: str, prompt: str = "What do you see in this image? read all info and translate it to chinese, don't translate person name"):
    """
    Function to read an image file and send it to GPT-4o Vision for analysis using OpenAI client.
    
    Args:
        image_path (str): Path to the image file
        prompt (str): Question/prompt to ask about the image
        
    Returns:
        dict: Response from GPT-4o Vision API
    """

    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Read image file
        with open(image_path, 'rb') as image_file:
            image_data = image_file.read()
        
        # Convert to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Determine image format
        image_path_obj = Path(image_path)
        image_format = image_path_obj.suffix.lower().replace('.', '')
        if image_format == 'jpg':
            image_format = 'jpeg'
        
        # Send request to OpenAI API using the official client
        response = client.chat.completions.create(
            model="gpt-4o",
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
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        return {
            "success": True,
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "model": response.model
        }
        
    except FileNotFoundError:
        return {
            "success": False,
            "error": f"Image file not found: {image_path}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error: {str(e)}"
        }
    
def analyze_images_simple(folder_path: str, output_file: str = "image_results_full.txt"):
    """
    Simplified version that just processes images and saves content to file.
    
    Args:
        folder_path (str): Path to the folder containing JPG images
        output_file (str): Path to the output text file
    """
    
    folder = Path(folder_path)
    jpg_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG")) + list(folder.glob("*.JPEG"))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for image_path in sorted(jpg_files):
            result = read_and_analyze_image(str(image_path))
            if result["success"]:
                f.write(f"{result['content']}\n")
            else:
                f.write(f"ERROR: {result['error']}\n")
    
    print(f"Processed {len(jpg_files)} images. Results saved to {output_file}")


# Example usage:
if __name__ == "__main__":
    # Example 1: Full featured version
    # result = analyze_images_in_folder(
    #     folder_path="./images",
    #     output_file="analysis_results.txt",
    #     prompt="Describe what you see in this image in detail."
    # )
    
    # if result["success"]:
    #     print(f"Success! {result['message']}")
    #     print(f"Processed: {result['successful_analyses']}/{result['total_files']} images")
    # else:
    #     print(f"Error: {result['error']}")
    
    # Example 2: Simple version
    analyze_images_simple("./img/whatsapp", "./data/image_results_full.txt")