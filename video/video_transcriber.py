#!/usr/bin/env python3
"""
Video Transcription Script
--------------------------
Extracts audio from video files and transcribes the speech to text.
"""

import os
import argparse
import whisper
from tqdm import tqdm

# Try different moviepy import approaches
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        # Alternative import method
        import moviepy
        VideoFileClip = moviepy.VideoFileClip
    except ImportError:
        raise ImportError("Could not import moviepy. Please ensure it's installed correctly with 'pip install moviepy'.")


def extract_audio(video_path, audio_output_path=None):
    """Extract audio from a video file and save it as a temporary WAV file."""
    print(f"Extracting audio from {video_path}...")
    
    if audio_output_path is None:
        # Create a temporary audio file in the same directory as the video
        base_name = os.path.splitext(video_path)[0]
        audio_output_path = f"{base_name}_temp_audio.wav"
    
    try:
        video = VideoFileClip(video_path)
        # Use the simplest form of write_audiofile without any optional parameters
        video.audio.write_audiofile(audio_output_path)
        print(f"Audio extracted and saved to {audio_output_path}")
        return audio_output_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        raise

from openai import OpenAI
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


def save_transcript(transcript, output_path=None, video_path=None):
    """Save the transcript to a text file."""
    if output_path is None and video_path is not None:
        # Create output file based on input video name
        base_name = os.path.splitext(video_path)[0]
        output_path = f"{base_name}_transcript.txt"
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        print(f"Transcript saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving transcript: {str(e)}")
        raise


def cleanup(audio_path):
    """Remove temporary audio file."""
    try:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print(f"Temporary audio file {audio_path} removed")
    except Exception as e:
        print(f"Warning: Could not remove temporary file {audio_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Extract and transcribe audio from video files")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("--output", "-o", help="Output path for the transcript (default: <video_name>_transcript.txt)")
    parser.add_argument("--keep-audio", "-k", action="store_true", help="Keep the extracted audio file")
    parser.add_argument("--model", "-m", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size to use (default: base)")
    
    args = parser.parse_args()
    
    try:
        # Make sure the video file exists
        if not os.path.exists(args.video_path):
            print(f"Error: Video file not found: {args.video_path}")
            return 1
            
        # Extract audio from the video
        audio_path = extract_audio(args.video_path)
        
        # Verify the audio file was created successfully
        if not os.path.exists(audio_path):
            print(f"Error: Audio extraction failed. No file created at {audio_path}")
            return 1
            
        # Print some debug info about the audio file
        print(f"Audio file details:")
        print(f"  - Path: {os.path.abspath(audio_path)}")
        print(f"  - Size: {os.path.getsize(audio_path)} bytes")
        print(f"  - Exists: {os.path.exists(audio_path)}")
        
        # Transcribe the audio
        try:
            transcript = transcribe_audio_file(audio_path)
            
            # Save the transcript
            save_transcript(transcript, args.output, args.video_path)
            
            # Cleanup temporary files
            if not args.keep_audio:
                cleanup(audio_path)
            
            print("Transcription completed successfully!")
        except Exception as e:
            print(f"Transcription failed: {str(e)}")
            print("The extracted audio file is still available at:", audio_path)
            return 1
        
    except Exception as e:
        print(f"Error during transcription process: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())