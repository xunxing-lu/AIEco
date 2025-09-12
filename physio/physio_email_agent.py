from __future__ import annotations
from contextlib import AsyncExitStack
from typing import Any, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.console import Console
from rich.live import Live
import asyncio
import os

from pydantic import BaseModel, Field
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.mcp import MCPServerStdio
from pydantic_ai import Agent, RunContext

from docx import Document
from docx.shared import Inches
import docx
import re
import os
from datetime import datetime
import pypandoc
from pathlib import Path

load_dotenv()


gavin_case = {
    'ctpt': '../data/physio/Gavin/gavinpt.txt',
    'solo': '../data/physio/Gavin/gavinsolo.txt',
    'progress_note' : r"../data/physio/Gavin/Progress_Note_Gavin_updated.docx",
    'output_file' : r"../data/physio/Gavin/Email_Gavin.docx"
}

margrate_case = {
    'ctpt': '../data/physio/Margrate/margaretpt.txt',
    'solo': '../data/physio/Margrate/margaretsolo.txt',
    'progress_note' : r"../data/physio/Margrate/Progress_Note_Margrate_updated.docx",
    'output_file' : r"../data/physio/Margrate/Email_Margrate.docx",
    'sub_notes_folder': r"../data/physio/Margrate/sub"
}

test_case = {
    'ctpt': '../data/physio/Test/testpt.txt',
    'solo': '../data/physio/Test/testsolo.txt',
    'progress_note' : r"../data/physio/Test/Progress_Note_Test_updated.docx",
    'output_file' : r"../data/physio/Test/Email_Test.docx"
}

# ========== Helper function to get model configuration ==========
def get_g_model():
    llm = 'o3'
    # print(llm)
    base_url = 'https://api.openai.com/v1'
    # print(base_url)
    api_key = os.getenv("OPENAI_API_KEY")
    # print(api_key)
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))

def get_o_model():
    llm = 'google/gemini-2.5-pro'
    base_url = 'https://openrouter.ai/api/v1'
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))




def write_text_to_word(text, filename="document.docx", title=None):
    """
    Write text to a Word document.
    
    Args:
        text (str): The text content to write to the document
        filename (str): Name of the output file (default: "document.docx")
        title (str, optional): Optional title for the document
    
    Returns:
        str: Path to the created document
    """
    # Create a new Document
    doc = Document()
    
    # Add title if provided
    if title:
        title_paragraph = doc.add_heading(title, 0)
    
    # Add the main text content
    # Split text by paragraphs (double newlines) for better formatting
    paragraphs = text.split('\n\n')
    
    for paragraph_text in paragraphs:
        if paragraph_text.strip():  # Skip empty paragraphs
            doc.add_paragraph(paragraph_text.strip())
    
    # Save the document
    doc.save(filename)
    
    return filename

def read_word(file_path):
    """
    read text from word
    """
    try:
        # Load the Word document
        doc = docx.Document(file_path)
        
        # Extract all text from the document
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        
        text = '\n'.join(full_text)
        
        return text
    
    except Exception as e:
        return f"Error processing file: {str(e)}"
    

def is_valid_date_folder(folder_name):
    """Check if folder name is in YYYYMMDD format."""
    try:
        datetime.strptime(folder_name, '%Y%m%d')
        return True
    except ValueError:
        return False

def get_date_folders(sub_notes_folder):
    """Get all date folders sorted by date."""
    if not os.path.exists(sub_notes_folder):
        print(f"Error: Folder {sub_notes_folder} does not exist")
        return []
    
    # Get all subdirectories
    all_items = os.listdir(sub_notes_folder)
    date_folders = []
    
    for item in all_items:
        item_path = os.path.join(sub_notes_folder, item)
        if os.path.isdir(item_path) and is_valid_date_folder(item):
            date_folders.append(item)
    
    # Sort folders by date (YYYYMMDD format allows string sorting)
    date_folders.sort()
    return date_folders

def read_folder_content(folder_path, date_str):
    """Read conversation.txt and solo.txt from a date folder."""
    content = {
        'date': date_str,
        'note': ''
    }
    
    # Look for conversation.txt
    # Look for any .docx file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.docx'):
            note_path = os.path.join(folder_path, filename)
            content['note'] = read_word(note_path)
            break  # Read the first .docx file found
    
    return content

def process_notes_folder(picked_case):
    """
    Main function to process the sub_notes_folder and organize content for LLM.
    
    Args:
        picked_case: Dictionary containing 'sub_notes_folder' key
    
    Returns:
        Dictionary with organized content ready for LLM system prompt
    """
    sub_notes_folder = picked_case['sub_notes_folder']

    # print(sub_notes_folder)
    
    # Get sorted date folders
    date_folders = get_date_folders(sub_notes_folder)

    # print(date_folders)
    
    if not date_folders:
        print("No valid date folders found")
        return {'organized_content': [], 'summary': 'No content found'}
    
    # Read content from each folder
    all_content = []
    
    for date_folder in date_folders:
        folder_path = os.path.join(sub_notes_folder, date_folder)
        # print(folder_path)
        content = read_folder_content(folder_path, date_folder)
        
        # Only add if there's actual content
        if content['note']:
            all_content.append(content)
            print(f"Processed {date_folder}: Conv={len(content['note'])} chars")
    
    return {
        'organized_content': all_content,
        'total_dates': len(all_content),
        'date_range': f"{date_folders[0]} to {date_folders[-1]}" if date_folders else "None"
    }

def format_for_llm_system_prompt(organized_data):
    """
    Format the organized content into a system prompt for LLM.
    
    Args:
        organized_data: Output from process_sub_notes_folder()
    
    Returns:
        String formatted as system prompt
    """
    if not organized_data['organized_content']:
        return "No historical notes available."
    
    prompt_parts = [
        "=== HISTORICAL NOTES ===",
        f"Date range: {organized_data['date_range']}",
        f"Total entries: {organized_data['total_dates']}",
        ""
    ]
    
    for entry in organized_data['organized_content']:
        date_formatted = datetime.strptime(entry['date'], '%Y%m%d').strftime('%Y-%m-%d')
        
        prompt_parts.append(f"--- {date_formatted} ---")
        
        if entry['note']:
            prompt_parts.append("Note:")
            prompt_parts.append(entry['note'])
            prompt_parts.append("")
        
        prompt_parts.append("") # Extra spacing between dates
    
    return "\n".join(prompt_parts)
    
def convert_markdown_to_word(input_file, output_file=None):
    """
    Convert a Markdown file to a Word document with error handling.
    
    Args:
        input_file (str): Path to the input Markdown file
        output_file (str, optional): Path to the output Word file. 
                                   If None, creates output file with same name but .docx extension
    
    Returns:
        str: Path to the created Word document
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    # Generate output filename if not provided
    if output_file is None:
        input_path = Path(input_file)
        output_file = input_path.with_suffix('.docx')
    
    # Method 1: Try pypandoc without reference document first
    try:
        pypandoc.convert_file(
            input_file,
            'docx',
            outputfile=str(output_file)
            # Removed the problematic reference-doc argument
        )
        
        print(f"Successfully converted '{input_file}' to '{output_file}'")
        return str(output_file)
        
    except Exception as e:
        print(f"Pypandoc conversion failed: {e}")
        print("Attempting manual conversion...")
        
        # Method 2: Manual conversion fallback
        try:
            return manual_markdown_to_docx(input_file, output_file)
        except Exception as e2:
            print(f"Manual conversion also failed: {e2}")
            raise

def manual_markdown_to_docx(input_file, output_file):
    """
    Manually convert markdown to DOCX using python-docx as fallback
    """
    # Read the markdown file
    with open(input_file, 'r', encoding='utf-8') as f:
        markdown_content = f.read()
    
    # Create a new document
    doc = Document()
    
    # Split content by lines for basic processing
    lines = markdown_content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        if not line:  # Empty line
            doc.add_paragraph()
            continue
            
        # Handle headers
        if line.startswith('# '):
            doc.add_heading(line[2:], level=1)
        elif line.startswith('## '):
            doc.add_heading(line[3:], level=2)
        elif line.startswith('### '):
            doc.add_heading(line[4:], level=3)
        elif line.startswith('#### '):
            doc.add_heading(line[5:], level=4)
        
        # Handle bullet points
        elif line.startswith('- ') or line.startswith('* '):
            doc.add_paragraph(line[2:], style='List Bullet')
        
        # Handle numbered lists  
        elif re.match(r'^\d+\.\s', line):
            content = re.sub(r'^\d+\.\s', '', line)
            doc.add_paragraph(content, style='List Number')
        
        # Regular paragraph
        else:
            # Handle basic bold formatting **text**
            if '**' in line:
                p = doc.add_paragraph()
                parts = line.split('**')
                for i, part in enumerate(parts):
                    if i % 2 == 0 and part:  # Normal text
                        p.add_run(part)
                    elif part:  # Bold text
                        p.add_run(part).bold = True
            else:
                doc.add_paragraph(line)
    
    # Save the document
    doc.save(output_file)
    print(f"Successfully converted '{input_file}' to '{output_file}' using manual conversion")
    return str(output_file)


picked_case = margrate_case
selected_model = get_g_model()



# reference
# https://www.sralab.org/rehabilitation-measures    
conversation1 = ''
with open(picked_case['ctpt'], 'r', encoding='utf-8') as file:
    conversation1 = file.read()

conversation2 = ''
with open(picked_case['solo'], 'r', encoding='utf-8') as file:
    conversation2 = file.read()

email1 = read_word('../data/physio/Gavin/Gavin Email.docx')
email2 = read_word('../data/physio/Margrate/Margaret Demo Email.docx')

progress_note = read_word(picked_case['progress_note'])

organized_historical_notes = process_notes_folder(picked_case)
# Format for LLM

historical_past_notes = format_for_llm_system_prompt(organized_historical_notes)


email_rule = read_word('../data/physio/Email system prompt.docx')

primary_agent = Agent(
    selected_model,
    system_prompt=f"""
    You are a Senior Physiotherapist who is very experienced in writing email to the care coordinator stakeholder based on current conversations with patient and historical progress notes.
    You will review the conversation with the patient to understand their current symptoms, concerns, and goals.

    You will use the following conversation to extract information and fill in the fields of the progress note:
        1, Conversation 1, which is between a physiotherapist and a patient: {conversation1} .
        2, Conversation 2, which is physiotherapist solo: {conversation2} .

    You will refer to the past progress notes provided to ensure you understand history of the notes of the patient and see the expected standards and format:
    Historical notes: {historical_past_notes} .

    You will also refer to the email rule for guidance on how to structure the email: {email_rule} .
    
    Here are some sample emails for you to learn how to write to the care coordinator stakeholder based on conversations and progress note:
        1, email sample 1: {email1} .
        2, email sample 2: {email2} .

    Return the response as a marked down format, which will be used to convert to the word document later on.

    Format the email content as clear, professional paragraphs.

    """
)

content = f"""
Please write a professional email to the care coordinator stakeholder with word friendly format.  
"""
# print(file_content)

result = primary_agent.run_sync(content)
email_result = result.data

output_path = picked_case['output_file']

# Replace the last few lines with this:
try:
    # Save markdown content to file first
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(email_result)
    
    # Convert markdown to Word document
    word_output_path = output_path.replace('.docx', '_converted.docx')
    convert_markdown_to_word(output_path, word_output_path)
    print(f"Final Word document saved to: {word_output_path}")
    
except Exception as e:
    print(f"Error in document conversion: {e}")
    print("Markdown content was still saved to:", output_path)