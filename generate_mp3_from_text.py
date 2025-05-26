import os
import argparse
from openai import OpenAI
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def text_to_speech(text, output_file, voice="nova", model="tts-1"):
    """
    Convert text to speech using OpenAI's TTS API and save as MP3 file
    
    Args:
        text (str): The text to convert to speech
        output_file (str): The path to save the MP3 file
        voice (str): The voice to use (default: "nova")
        model (str): The TTS model to use (default: "tts-1")
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Converting text to speech using {model} model with {voice} voice...")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate speech using OpenAI TTS
        response = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            speed=1.05
        )
        
        # Save to file
        response.stream_to_file(output_file)
        
        print(f"Audio saved successfully to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate MP3 from text using OpenAI TTS')
    parser.add_argument('--text', type=str, help='Text to convert to speech')
    parser.add_argument('--file', type=str, help='Text file to convert to speech')
    parser.add_argument('--output', type=str, default='output.mp3', help='Output MP3 file path')
    parser.add_argument('--voice', type=str, default='nova', 
                        choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], 
                        help='Voice to use')
    parser.add_argument('--model', type=str, default='tts-1', 
                        choices=['tts-1', 'tts-1-hd'], 
                        help='TTS model to use')
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key using:")
        print("  export OPENAI_API_KEY=your-api-key (Linux/Mac)")
        print("  set OPENAI_API_KEY=your-api-key (Windows)")
        return
    
    # Get text from either command line or file
    text = ""
    if args.text:
        text = args.text
    elif args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        print("Please provide text using --text or --file argument")
        parser.print_help()
        return
    
    # Generate speech
    if text:
        text_to_speech(text, args.output, args.voice, args.model)

if __name__ == "__main__":
    main()