import speech_recognition as sr
import pyaudio
import wave
import os
import numpy as np
import struct
import webrtcvad
from array import array
from openai import OpenAI
import tempfile
import time
from pygame import mixer
import pvporcupine
import threading
import queue
import audioop
import collections

# Initialize OpenAI client - requires API key in environment variables
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Common rate for VAD
CHUNK_DURATION_MS = 30  # Duration of each chunk in milliseconds
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # Number of samples per chunk
PADDING_DURATION_MS = 300  # Pre-padding and post-padding duration
NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)  # 240ms window for VAD
THRESHOLD = 3000  # Threshold for silence detection
SILENT_CHUNKS = 3  # Number of consecutive silent chunks for silence detection

# Global variables
audio_queue = queue.Queue()
stop_listening = threading.Event()
is_active = threading.Event()

class VADAudio:
    """Helper class for VAD-based audio recording"""
    
    def __init__(self, vad_aggressiveness=3):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames = []
        self.got_speech = False
    
    def start_stream(self):
        """Start the audio stream"""
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            stream_callback=self._callback
        )
        return self
    
    def _callback(self, in_data, frame_count, time_info, status):
        """Callback function for streaming audio"""
        if stop_listening.is_set():
            return (None, pyaudio.paComplete)
        
        audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def frame_generator(self):
        """Generate audio frames from queue"""
        while not stop_listening.is_set():
            try:
                frame = audio_queue.get(block=True, timeout=0.5)
                yield frame
            except queue.Empty:
                pass
    
    def detect_speech(self):
        """Detect speech using VAD"""
        num_padding_chunks = NUM_PADDING_CHUNKS
        ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
        triggered = False
        
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, RATE)
            
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                
                if num_voiced > 0.5 * ring_buffer.maxlen:
                    triggered = True
                    self.frames = []
                    self.frames.extend([f for f, s in ring_buffer])
                    ring_buffer.clear()
                    self.got_speech = True
                    is_active.set()
                    print("Speech detected, listening...")
            else:
                self.frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    is_active.clear()
                    print("Speech ended")
                    return self.create_audio_data()
        
        return None
    
    def create_audio_data(self):
        """Create AudioData object from recorded frames"""
        if not self.frames:
            return None
        
        audio_data = sr.AudioData(
            b''.join(self.frames),
            RATE,
            2  # 16-bit samples = 2 bytes
        )
        self.frames = []
        return audio_data
    
    def close(self):
        """Close the audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()


def keyword_detection_loop(access_key):
    """Run keyword detection in a separate thread"""
    try:
        porcupine = pvporcupine.create(
            access_key=access_key,
            keywords=['jarvis', 'computer', 'porcupine']  # Built-in keywords - can be customized
        )
        
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length
        )
        
        print("Keyword detection active. Say 'jarvis', 'computer', or 'porcupine' to activate...")
        
        while not stop_listening.is_set():
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
            
            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print(f"Wake word detected! ({porcupine.keywords[keyword_index]})")
                is_active.set()
                time.sleep(0.5)  # Give time to acknowledge before starting VAD
        
        audio_stream.close()
        pa.terminate()
        porcupine.delete()
        
    except Exception as e:
        print(f"Error in keyword detection: {e}")
        # Fall back to VAD-only mode if keyword detection fails
        is_active.set()


def record_with_vad():
    """Record audio with Voice Activity Detection"""
    print("Starting VAD-based recording...")
    vad_audio = VADAudio(vad_aggressiveness=3)
    vad_audio.start_stream()
    
    audio_data = vad_audio.detect_speech()
    vad_audio.close()
    
    return audio_data


def record_audio_simple():
    """Simple record audio from microphone (fallback method)"""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... (speak now)")
        
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("Recording complete!")
            return audio
        except sr.WaitTimeoutError:
            print("No speech detected")
            return None


def transcribe_audio(audio):
    """Transcribe speech to text using OpenAI's Whisper model"""
    try:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix='.wav') as temp_audio:
            temp_filename = temp_audio.name
            
        with open(temp_filename, "wb") as f:
            f.write(audio.get_wav_data())
        
        # Transcribe using OpenAI
        with open(temp_filename, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        
        os.unlink(temp_filename)  # Delete temporary file
        
        transcribed_text = transcript.text
        print(f"You said: {transcribed_text}")
        return transcribed_text
    
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


def get_gpt_response(transcribed_text):
    """Send text to GPT-4o and get response"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."},
                {"role": "user", "content": transcribed_text}
            ]
        )
        
        assistant_response = response.choices[0].message.content
        print(f"Assistant: {assistant_response}")
        return assistant_response
    
    except Exception as e:
        print(f"Error with GPT-4o: {e}")
        return "I'm sorry, I encountered an error processing your request."


def text_to_speech(text):
    """Convert text to speech using OpenAI's TTS"""
    try:
        speech_file_path = tempfile.NamedTemporaryFile(delete=True, suffix='.mp3').name
        
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        response.stream_to_file(speech_file_path)
        return speech_file_path
    
    except Exception as e:
        print(f"Error during text-to-speech: {e}")
        return None


def play_audio(file_path):
    """Play audio file using pygame mixer"""
    try:
        mixer.init()
        mixer.music.load(file_path)
        mixer.music.play()
        
        # Wait for the audio to finish playing
        while mixer.music.get_busy():
            time.sleep(0.1)
            
        mixer.quit()
        
        # Clean up temporary file
        os.unlink(file_path)
        
    except Exception as e:
        print(f"Error playing audio: {e}")


def main(use_keyword_activation=False, picovoice_access_key=None):
    """Main function to run the voice assistant"""
    print("Voice Assistant starting...")
    print("Press Ctrl+C to exit")
    
    # Import here to avoid circular imports
    import collections
    
    if use_keyword_activation and picovoice_access_key:
        # Start keyword detection in separate thread
        keyword_thread = threading.Thread(
            target=keyword_detection_loop,
            args=(picovoice_access_key,),
            daemon=True
        )
        keyword_thread.start()
        print("Keyword activation enabled")
    else:
        # Set always active if no keyword detection
        is_active.set()
        print("Using VAD only (no keyword activation)")
    
    try:
        while not stop_listening.is_set():
            # Wait for activation if needed
            if not is_active.is_set() and use_keyword_activation:
                time.sleep(0.1)
                continue
            
            print("\n--- New Conversation ---")
            try:
                # Record audio using VAD
                audio_data = record_with_vad()
            except Exception as e:
                print(f"Error with VAD recording: {e}")
                print("Falling back to simple recording method...")
                audio_data = record_audio_simple()
            
            if audio_data:
                # Speech to text
                transcribed_text = transcribe_audio(audio_data)
                
                if transcribed_text:
                    # Get response from GPT-4o
                    gpt_response = get_gpt_response(transcribed_text)
                    
                    # Text to speech
                    speech_file = text_to_speech(gpt_response)
                    
                    if speech_file:
                        # Play the response
                        play_audio(speech_file)
            
            # Reset activation if using keyword mode
            if use_keyword_activation:
                is_active.clear()
            
            # Small delay before next iteration
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nExiting Voice Assistant. Goodbye!")
        stop_listening.set()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Voice Assistant with VAD and optional keyword activation")
    parser.add_argument("--keyword", action="store_true", help="Enable keyword activation")
    parser.add_argument("--key", type=str, help="Picovoice access key for keyword detection")
    
    args = parser.parse_args()
    
    if args.keyword and not args.key:
        print("Error: Keyword activation requires a Picovoice access key")
        print("Get a free key at: https://console.picovoice.ai/")
        print("Running with VAD only...")
        args.keyword = False
    
    main(use_keyword_activation=args.keyword, picovoice_access_key=args.key)