import speech_recognition as sr
import pyaudio
import wave
import os
from openai import OpenAI
import tempfile
import time
from pygame import mixer
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client - requires API key in environment variables
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def record_audio():
    """Record audio from microphone until silence is detected"""
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
    recognizer = sr.Recognizer()
    
    try:
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            temp_filename = temp_audio.name

        
            
        with open(temp_filename, "wb") as f:
            f.write(audio.get_wav_data())

        print(f"Transcribing audio from {temp_filename}...")
        
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
        speech_file_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3').name
        
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

def main():
    print("Voice Assistant starting...")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            # Record audio from microphone
            print("\n--- New Conversation ---")
            audio_data = record_audio()
            
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
            
            # Small delay before next iteration
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nExiting Voice Assistant. Goodbye!")

if __name__ == "__main__":
    main()