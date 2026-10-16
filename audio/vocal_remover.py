import os
import shutil
import subprocess
import sys
import tempfile

def remove_vocals(input_file, model='htdemucs'):
    """Remove human voice from an MP3 using Demucs, keeping the background music"""
    base = input_file.rsplit('.', 1)[0]
    output_file = base + '_no_vocals.mp3'
    track = os.path.basename(base)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            subprocess.run([
                sys.executable, '-m', 'demucs',
                '--two-stems=vocals',
                '--mp3',
                '-n', model,
                '-o', tmp_dir,
                input_file
            ], check=True)

            shutil.move(os.path.join(tmp_dir, model, track, 'no_vocals.mp3'), output_file)

        print(output_file)

        return output_file
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        remove_vocals(sys.argv[1])
