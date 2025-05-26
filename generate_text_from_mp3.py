import os
import argparse
from openai import OpenAI
import tempfile
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio_file(audio_file_path):
    """
    Transcribe an MP3 file to text using OpenAI's Whisper model
    
    Args:
        audio_file_path (str): Path to the MP3 file
        
    Returns:
        str: Transcribed text
    """
    try:
        print(f"Transcribing audio file: {audio_file_path}")
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return None
        
        # Transcribe using OpenAI's Whisper model
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        transcribed_text = transcript.text
        print(f"Transcription complete! ({len(transcribed_text)} characters)")
        return transcribed_text
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def save_transcript(text, output_file):
    """
    Save transcribed text to a file
    
    Args:
        text (str): The transcribed text
        output_file (str): The path to save the text file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
            
        print(f"Transcript saved to: {output_file}")
        return True
    
    except Exception as e:
        print(f"Error saving transcript: {e}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Transcribe MP3 file to text using OpenAI Whisper')
    parser.add_argument('--input', type=str, help='Path to the MP3 file to transcribe')
    parser.add_argument('--output', type=str, help='Path to save the transcribed text (default: input_file_name.txt)')
    
    args = parser.parse_args()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key using:")
        print("  export OPENAI_API_KEY=your-api-key (Linux/Mac)")
        print("  set OPENAI_API_KEY=your-api-key (Windows)")
        return
    
    # Get the input file path
    if args.input:
        input_file = args.input
    else:
        # If no input is provided, check if there's a file in data/001.mp3
        default_file = os.path.join("data", "001.mp3")
        if os.path.exists(default_file):
            input_file = default_file
            print(f"No input file specified, using default: {default_file}")
        else:
            print("Error: No input file specified and default file not found")
            parser.print_help()
            return
    
    # Set the output file path
    if args.output:
        output_file = args.output
    else:
        # Create output filename based on input filename, but with .txt extension
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(os.path.dirname(input_file), f"{base_name}.txt")
    
    # Transcribe the audio file
    transcribed_text = transcribe_audio_file(input_file)
    
    # Save the transcript if transcription was successful
    if transcribed_text:
        save_transcript(transcribed_text, output_file)
        print("\nTranscription process completed successfully!")
    else:
        print("\nTranscription failed. See error messages above.")

if __name__ == "__main__":
    main()