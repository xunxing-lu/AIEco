import os
import argparse
from openai import OpenAI
import tempfile
import time
from pydub import AudioSegment
import math

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# OpenAI Whisper API file size limit (25MB)
MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes

def get_file_size(file_path):
    """Get file size in bytes"""
    return os.path.getsize(file_path)

def convert_and_compress_audio(input_file, output_file, target_size_mb=20):
    """
    Convert and compress audio file to reduce size
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to output compressed file
        target_size_mb (int): Target size in MB (default 20MB for safety margin)
    
    Returns:
        str: Path to compressed file or None if failed
    """
    try:
        print(f"Loading audio file for compression...")
        
        # Load audio file (supports many formats including m4a, mp3, wav, etc.)
        audio = AudioSegment.from_file(input_file)
        
        # Calculate current bitrate and target bitrate
        duration_seconds = len(audio) / 1000.0
        current_size_mb = get_file_size(input_file) / (1024 * 1024)
        
        print(f"Original file: {current_size_mb:.1f}MB, Duration: {duration_seconds:.1f}s")
        
        # Calculate target bitrate (with some safety margin)
        target_size_bytes = target_size_mb * 1024 * 1024
        target_bitrate = int((target_size_bytes * 8) / duration_seconds * 0.9)  # 90% of theoretical max
        
        # Ensure minimum quality
        min_bitrate = 32000  # 32kbps minimum
        target_bitrate = max(target_bitrate, min_bitrate)
        
        print(f"Compressing to ~{target_size_mb}MB (target bitrate: {target_bitrate}bps)...")
        
        # Export as MP3 with target bitrate
        audio.export(
            output_file,
            format="mp3",
            bitrate=f"{target_bitrate}",
            parameters=["-ac", "1"]  # Convert to mono to save space
        )
        
        new_size_mb = get_file_size(output_file) / (1024 * 1024)
        print(f"Compressed file: {new_size_mb:.1f}MB")
        
        return output_file
        
    except Exception as e:
        print(f"Error during audio compression: {e}")
        print("Note: You may need to install pydub and ffmpeg:")
        print("  pip install pydub")
        print("  And install ffmpeg: https://ffmpeg.org/download.html  winget install ffmpeg")
        return None

def split_audio_file(input_file, chunk_duration_minutes=10):
    """
    Split audio file into smaller chunks
    
    Args:
        input_file (str): Path to input audio file 
        chunk_duration_minutes (int): Duration of each chunk in minutes
        
    Returns:
        list: List of chunk file paths
    """
    try:
        print(f"Splitting audio file into {chunk_duration_minutes}-minute chunks...")
        
        # Load audio file
        audio = AudioSegment.from_file(input_file)
        
        # Calculate chunk size in milliseconds
        chunk_size_ms = chunk_duration_minutes * 60 * 1000
        
        # Create chunks
        chunks = []
        base_name = os.path.splitext(input_file)[0]
        
        for i, chunk_start in enumerate(range(0, len(audio), chunk_size_ms)):
            chunk_end = min(chunk_start + chunk_size_ms, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            chunk_filename = f"{base_name}_chunk_{i+1:03d}.mp3"
            chunk.export(chunk_filename, format="mp3")
            chunks.append(chunk_filename)
            
            chunk_size_mb = get_file_size(chunk_filename) / (1024 * 1024)
            print(f"Created chunk {i+1}: {chunk_filename} ({chunk_size_mb:.1f}MB)")
        
        return chunks
        
    except Exception as e:
        print(f"Error during audio splitting: {e}")
        return None

def transcribe_audio_file(audio_file_path):
    """
    Transcribe an audio file to text using OpenAI's Whisper model
    
    Args:
        audio_file_path (str): Path to the audio file
        
    Returns:
        str: Transcribed text
    """
    try:
        print(f"Transcribing audio file: {audio_file_path}")
        
        # Check if file exists
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return None
        
        # Check file size
        file_size = get_file_size(audio_file_path)
        if file_size > MAX_FILE_SIZE:
            print(f"Error: File size ({file_size/1024/1024:.1f}MB) exceeds OpenAI's limit ({MAX_FILE_SIZE/1024/1024}MB)")
            return None
        
        # Transcribe using OpenAI's Whisper model
        with open(audio_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        # Handle both string and object responses
        if isinstance(transcript, str):
            transcribed_text = transcript
        else:
            transcribed_text = transcript.text if hasattr(transcript, 'text') else str(transcript)
        
        print(f"Transcription complete! ({len(transcribed_text)} characters)")
        return transcribed_text
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def transcribe_chunks(chunk_files):
    """
    Transcribe multiple audio chunks and combine results
    
    Args:
        chunk_files (list): List of chunk file paths
        
    Returns:
        str: Combined transcribed text
    """
    all_transcripts = []
    
    for i, chunk_file in enumerate(chunk_files):
        print(f"\nTranscribing chunk {i+1}/{len(chunk_files)}...")
        transcript = transcribe_audio_file(chunk_file)
        
        if transcript:
            all_transcripts.append(f"[Chunk {i+1}]\n{transcript}\n")
        else:
            print(f"Failed to transcribe chunk {i+1}")
    
    # Clean up chunk files
    for chunk_file in chunk_files:
        try:
            os.remove(chunk_file)
            print(f"Cleaned up: {chunk_file}")
        except:
            pass
    
    return "\n".join(all_transcripts)

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
    parser = argparse.ArgumentParser(description='Transcribe audio file to text using OpenAI Whisper')
    parser.add_argument('--input', type=str, help='Path to the audio file to transcribe')
    parser.add_argument('--output', type=str, help='Path to save the transcribed text (default: input_file_name.txt)')
    parser.add_argument('--compress', action='store_true', help='Compress audio file if it exceeds size limit')
    parser.add_argument('--split', action='store_true', help='Split audio file into chunks if it exceeds size limit')
    parser.add_argument('--chunk-minutes', type=int, default=10, help='Duration of each chunk in minutes (default: 10)')
    
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
    
    # Check file size
    file_size = get_file_size(input_file)
    print(f"Input file size: {file_size/1024/1024:.1f}MB")
    
    transcribed_text = None
    
    if file_size > MAX_FILE_SIZE:
        print(f"File exceeds OpenAI's {MAX_FILE_SIZE/1024/1024}MB limit")
        
        if args.compress:
            # Try compression first
            compressed_file = input_file.replace('.m4a', '_compressed.mp3').replace('.mp3', '_compressed.mp3')
            compressed_path = convert_and_compress_audio(input_file, compressed_file)
            
            if compressed_path and get_file_size(compressed_path) <= MAX_FILE_SIZE:
                print("Compression successful, proceeding with transcription...")
                transcribed_text = transcribe_audio_file(compressed_path)
                # Clean up compressed file
                try:
                    os.remove(compressed_path)
                except:
                    pass
            else:
                print("Compression failed or still too large, trying split method...")
                args.split = True
        
        if args.split and not transcribed_text:
            # Split into chunks
            print("split")
            chunks = split_audio_file(input_file, args.chunk_minutes)
            if chunks:
                transcribed_text = transcribe_chunks(chunks)
            else:
                print("Failed to split audio file")
                return
        
        if not args.compress and not args.split:
            print("File is too large for OpenAI's API. Try using --compress or --split options:")
            print("  --compress: Compress the audio file to reduce size")
            print("  --split: Split the audio file into smaller chunks")
            print(f"  --chunk-minutes: Set chunk duration (default: {args.chunk_minutes} minutes)")
            return
    else:
        # File is within size limit, transcribe directly
        transcribed_text = transcribe_audio_file(input_file)
    
    # Save the transcript if transcription was successful
    if transcribed_text:
        save_transcript(transcribed_text, output_file)
        print("\nTranscription process completed successfully!")
    else:
        print("\nTranscription failed. See error messages above.")

if __name__ == "__main__":
    main()