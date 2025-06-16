import subprocess
import os
import sys

def convert_mp4_to_avi_ffmpeg(input_file, output_file=None):
    """
    Convert MP4 to AVI using ffmpeg command line tool
    
    Args:
        input_file (str): Path to input MP4 file
        output_file (str, optional): Path for output AVI file
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            return False
        
        # Generate output filename if not provided
        if output_file is None:
            base_name = os.path.splitext(input_file)[0]
            output_file = f"{base_name}.avi"
        
        print(f"Converting {input_file} to {output_file}...")
        
        # FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libxvid',  # Video codec
            '-c:a', 'mp3',      # Audio codec
            '-q:v', '3',        # Video quality (1-5, lower is better)
            '-y',               # Overwrite output file if exists
            output_file
        ]
        
        # Run the conversion
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Conversion completed successfully!")
            print(f"Output file: {output_file}")
            return True
        else:
            print(f"Error during conversion: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg first.")
        print("Visit: https://ffmpeg.org/download.html")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def check_ffmpeg():
    """Check if ffmpeg is installed and accessible"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ ffmpeg is installed and ready to use")
            return True
        else:
            print("✗ ffmpeg found but not working properly")
            return False
    except FileNotFoundError:
        print("✗ ffmpeg not found")
        print("Please install ffmpeg from: https://ffmpeg.org/download.html")
        return False

# Example usage
if __name__ == "__main__":
    # Check if ffmpeg is available
    if not check_ffmpeg():
        sys.exit(1)
    
    # Command line usage
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_mp4_to_avi_ffmpeg(input_file, output_file)
    else:
        print("Usage:")
        print("  python script.py input.mp4 [output.avi]")