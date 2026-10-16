import subprocess
import sys

def convert_wav_to_mp3(input_file, bitrate='192k'):
    """Convert WAV file to MP3 format using ffmpeg"""
    output_file = input_file.rsplit('.', 1)[0] + '_converted.mp3'
    try:
        subprocess.run([
            'ffmpeg', '-y', '-i', input_file,
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            output_file
        ], check=True, capture_output=True)

        print(output_file)

        return output_file
    except subprocess.CalledProcessError:
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_wav_to_mp3(sys.argv[1])
    # else:
    #     convert_wav_to_mp3(r"C:\Projects\AIEchoMain\AIEco\data\Margaret Demo Objective (PT solo)_converted.wav")
