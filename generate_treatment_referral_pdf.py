import os
import time
from openai import OpenAI
from fpdf import FPDF
import datetime
from pydub import AudioSegment
import tempfile
import re

# Initialize OpenAI client - update with your API key or use environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(audio_file_path):
    """
    Transcribe audio file using OpenAI's Whisper model
    
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
            
        # Convert mp3 to wav if needed for compatibility with OpenAI's API
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

def analyze_conversation(transcribed_text):
    """
    Analyze conversation using GPT-4 to generate a physiotherapy report
    and referral letter
    
    Args:
        transcribed_text (str): The transcribed conversation text
        
    Returns:
        tuple: (summary, referral_letter)
    """
    try:
        print("Analyzing conversation with GPT...")
        
        # Prompt engineering for better results
        system_prompt = """
        You are a medical professional assistant that specializes in creating physiotherapy 
        documentation. Analyze the provided conversation between a patient and healthcare provider.
        
        Create two sections:
        1. A clinical summary that includes:
           - Patient details (name, age, gender) if mentioned
           - Main complaint and symptoms
           - Medical history relevant to current condition
           - Current pain levels and limitations
           - Previous treatments tried
        
        2. A professional physiotherapy referral letter that includes:
           - Standard medical referral header with today's date
           - Patient details
           - Concise reason for referral
           - Relevant clinical findings
           - Specific treatment recommendations
           - Follow-up timeline recommendations
           - Professional closing
        
        Use a professional medical tone throughout. If certain information is not available 
        in the conversation, make reasonable assumptions based on what is mentioned, but
        indicate when you've made assumptions.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here is the transcribed conversation:\n\n{transcribed_text}"}
            ]
        )
        
        full_response = response.choices[0].message.content
        
        # Try to split the response into summary and referral letter sections
        sections = re.split(r'(?:\n\n|\n)(?:2\.|REFERRAL LETTER:|Referral Letter:)', full_response, 1)
        
        if len(sections) >= 2:
            summary = sections[0].strip()
            referral_letter = sections[1].strip()
            
            # Clean up any numbering from the summary
            summary = re.sub(r'^1\.[ \t]*', '', summary)
        else:
            # If we can't split it easily, use the whole response
            summary = full_response
            referral_letter = full_response
        
        print("Analysis complete!")
        return summary, referral_letter
    
    except Exception as e:
        print(f"Error during GPT analysis: {e}")
        return "Error generating summary", "Error generating referral letter"

def generate_pdf(summary, referral_letter, output_path):
    """
    Generate a PDF with the clinical summary and referral letter
    
    Args:
        summary (str): Clinical summary text
        referral_letter (str): Referral letter text
        output_path (str): Path to save the PDF file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Generating PDF at: {output_path}")
        
        # Create PDF object with UTF-8 encoding support
        pdf = FPDF()
        # Use built-in fonts that support basic UTF-8
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Encode text to handle UTF-8 characters
        def encode_for_pdf(text):
            return text.encode('latin-1', 'replace').decode('latin-1')
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Physiotherapy Assessment & Referral", ln=True, align="C")
        pdf.ln(5)
        
        # Add current date
        pdf.set_font("Arial", "I", 10)
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        pdf.cell(0, 10, f"Generated on: {current_date}", ln=True, align="R")
        pdf.ln(5)
        
        # Clinical Summary Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "CLINICAL SUMMARY", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Summary content
        pdf.set_font("Arial", "", 11)
        
        # Handle long text with multi_cell (with encoding for UTF-8)
        pdf.multi_cell(0, 6, encode_for_pdf(summary))
        pdf.ln(10)
        
        # Referral Letter Section
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "PHYSIOTHERAPY REFERRAL LETTER", ln=True)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)
        
        # Referral content
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, encode_for_pdf(referral_letter))
        
        # Output PDF
        pdf.output(output_path)
        print(f"PDF generated successfully at: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error generating PDF: {e}")
        return False

def main():
    print("Physiotherapy Referral Generator")
    print("--------------------------------")
    
    # Define paths
    audio_file_path = os.path.join("data", "sample_conversation_output.mp3")
    output_pdf_path = os.path.join("data", "physiotherapy_referral.pdf")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    # Step 1: Transcribe the audio
    transcribed_text = transcribe_audio(audio_file_path)
    
    if transcribed_text:
        # Step 2: Analyze the conversation
        summary, referral_letter = analyze_conversation(transcribed_text)
        
        # Step 3: Generate PDF
        generate_pdf(summary, referral_letter, output_pdf_path)
        
        print(f"\nProcess complete! PDF saved to: {output_pdf_path}")
    else:
        print("Failed to transcribe audio. Process aborted.")

if __name__ == "__main__":
    main()