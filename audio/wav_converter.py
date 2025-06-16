import subprocess
import os

def convert_to_wav(input_file):
    """Convert audio file to WAV format using ffmpeg"""
    output_file = input_file.rsplit('.', 1)[0] + '_converted.wav'
    print(output_file)
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, 
            '-acodec', 'pcm_s16le', 
            '-ar', '16000',     
            output_file
        ], check=True, capture_output=True)

        print(output_file)

        return output_file
    except subprocess.CalledProcessError:
        return None
    
convert_to_wav(r"C:\Projects\AIEchoMain\AIEco\data\Margaret Demo Objective (PT solo).m4a")