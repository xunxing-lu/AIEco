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
from pathlib import Path

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

from pydantic_ai.models.gemini import GeminiModel
from pydantic_ai.providers.google_gla import GoogleGLAProvider

import logging
import pypandoc
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

gavin_case = {
    'ctpt': '../data/physio/Gavin/gavinpt.txt',
    'solo': '../data/physio/Gavin/gavinsolo.txt',
    'template_file' : r"../data/physio/Progress_Note_Template.docx",
    'output_file' : r"../data/physio/Gavin/Progress_Note_Gavin_updated.docx",
    'sub_notes_folder': r""
}

margrate_case = {
    'ctpt': '../data/physio/Margrate/margaret_pt_session2.txt',
    'solo': '../data/physio/Margrate/margaret_solo_session2.txt',
    'template_file' : r"../data/physio/Progress_Note_Template.docx",
    'output_file' : r"../data/physio/Margrate/Progress_Note_Margrate_sub.docx",
    'sub_notes_folder': r"../data/physio/Margrate/sub"
}

test_case = {
    'ctpt': '../data/physio/Test/testpt.txt',
    'solo': '../data/physio/Test/testsolo.txt',
    'template_file' : r"../data/physio/Progress_Note_Template.docx",
    'output_file' : r"../data/physio/Test/Progress_Note_Test_updated.docx",
    'sub_notes_folder': r""
}

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
    # logger.info(f"Using model: {llm}")
    base_url = 'https://openrouter.ai/api/v1'
    api_key = os.getenv("OPEN_ROUTER_API_KEY")
    return OpenAIModel(llm, provider=OpenAIProvider(base_url=base_url, api_key=api_key))



def update_word_template(template_path, output_path, replacement_dict, image_replacements):
    doc = Document(template_path)
    pattern = r'\[(.*?)\]'

    def process_paragraphs(paragraphs):
        for paragraph in paragraphs:
            full_text = ''.join(run.text for run in paragraph.runs)

            if '[Scooter_Image]' in full_text:
                print("found image")


                # Clear all runs
                for run in paragraph.runs:
                    run.text = ''

                # Split around the image placeholder
                parts = full_text.split('[Scooter_Image]')
                
                # Rebuild the paragraph
                if parts[0]:
                    paragraph.add_run(parts[0])
                # Add the image
                paragraph.add_run().add_picture(image_replacements['Scooter_Image'], width=Inches(3))
                if len(parts) > 1:
                    paragraph.add_run(parts[1])
            else:
                for match in re.finditer(pattern, paragraph.text):
                    key = match.group(1)
                    if key in replacement_dict:
                        paragraph.text = paragraph.text.replace(f'[{key}]', replacement_dict[key])

    process_paragraphs(doc.paragraphs)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                process_paragraphs(cell.paragraphs)

    for section in doc.sections:
        for footer in [section.footer, section.first_page_footer, section.even_page_footer]:
            if footer is not None:
                for p in footer.paragraphs:
                    print("Footer paragraph:", p.text)

                process_paragraphs(footer.paragraphs)
                for table in footer.tables:
                    for row in table.rows:
                        for cell in row.cells:
                            # for p in cell.paragraphs:
                            #     print("Cell paragraph:", p.text)
                            process_paragraphs(cell.paragraphs)

    # for section in doc.sections:
    #     footer = section.footer
    #     # for p in footer.paragraphs:
    #     #     print("Footer paragraph:", p.text)
        
    #     process_paragraphs(footer.paragraphs)

    #     # If your footer contains tables
    #     for table in footer.tables:
    #         for row in table.rows:
    #             for cell in row.cells:
    #                 # for p in cell.paragraphs:
    #                 #     print("Cell paragraph:", p.text)
    #                 process_paragraphs(cell.paragraphs)


    for section in doc.sections:
        header = section.header
        process_paragraphs(header.paragraphs)

        for table in header.tables:
            for row in table.rows:
                for cell in row.cells:
                    process_paragraphs(cell.paragraphs)



    doc.save(output_path)
    print(f"Document saved to {output_path}")




picked_case = margrate_case
selected_model = get_g_model()


# Replace your existing convert_markdown_to_word function with this updated version:

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

# Also, update your main execution section at the bottom:


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

def read_text_file(file_path):
    """Read a text file and return its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

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
    note_path = os.path.join(folder_path, 'conversation.txt')
    if os.path.exists(note_path):
        content['note'] = read_word(note_path)
    
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
    
    # Get sorted date folders
    date_folders = get_date_folders(sub_notes_folder)
    
    if not date_folders:
        print("No valid date folders found")
        return {'organized_content': [], 'summary': 'No content found'}
    
    # Read content from each folder
    all_content = []
    
    for date_folder in date_folders:
        folder_path = os.path.join(sub_notes_folder, date_folder)
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

conversation1 = ''
with open(picked_case['ctpt'], 'r', encoding='utf-8') as file:
    conversation1 = file.read()

conversation2 = ''
with open(picked_case['solo'], 'r', encoding='utf-8') as file:
    file_content = file.read()

sample_progress_note_1 = read_word('../data/physio/Gavin/Gavin Progress Note.docx')
sample_progress_note_2 = read_word('../data/physio/Margrate/Margaret Demo Progress Note.docx')

initial_note = read_word('../data/physio/Subsequent note system prompt.docx')
# loop to read
# sub_past_notes = read_word(picked_case['sub_notes_folder']) 

organized_historical_notes = process_notes_folder(picked_case)
# Format for LLM
historical_past_notes = format_for_llm_system_prompt(organized_historical_notes)

sub_note_rule = read_word('../data/physio/Subsequent note system prompt.docx')

primary_agent = Agent(
    selected_model,
    # get_gemini_model(),
    system_prompt=f"""
    You are a Senior Physiotherapist with extensive experience in conducting physiotherapy assessments and writing professional subsequent progress notes for patient records.
    You will refer to the past progress notes provided to ensure you understand history of the notes of the patient and see the expected standards and format:
    Historical notes: {historical_past_notes} .

    You will review the conversation with the patient to understand their current symptoms, concerns, and goals.

    You will use the following conversation to extract information and fill in the fields of the progress note:
        1, Conversation 1, which is between a physiotherapist and a patient: {conversation1} .
        2, Conversation 2, which is physiotherapist solo: {conversation2} .

    After that, you will reflect privately to recall and summarise the key points from the conversation.
    Then, you will write a clear, concise, and clinically accurate sub progress note, following professional physiotherapy documentation standards.

    You will use the provided sub note rule as a guide for the structure and content of the progress note.
    The progress note should be well-organised, using appropriate medical terminology and clear language.
    It should include all relevant information from the conversation, including subjective and objective findings, assessment, and plan for future care.
    You will also refer to the sample progress notes provided to ensure your note meets the expected standards and format.

    You will also refer to the sub note rule for guidance on how to structure the sub progress note: {sub_note_rule} .

    You will use the following terminology to ensure professional and simplified descriptions:
    Use professional physiotherapy shorthand and abbreviations where clinically appropriate. Ensure abbreviations are correct, contextually accurate, and match their standard meaning. When in doubt, maintain full terminology. The following list provides common terms and their meanings:

        FAEO — Feet Apart, Eyes Open (record time held, note gait aid if used; e.g., “FAEO nil gait aide 7s”).
        FAEC — Feet Apart, Eyes Closed.
        FTEO — Feet Together, Eyes Open.
        FTEC — Feet Together, Eyes Closed.
        UL — Upper Limbs; muscle strength graded 0/5 to 5/5 (5 = normal, 4 = good, etc.).
        TF — Task/Function.
        5x STS — Five Times Sit-to-Stand Test (record time taken).
        4WW — 4-Wheeled Walker.
        WC — Wheelchair.
        R/V — Review (e.g., R/V 1/52 = Review in 1 week).
        A+O — Alert and Oriented.
        TPP — Time, Place, Person.
        I/M — Intermittent.
        Dx — Diagnosis.
        Hx — History.
        a/a — As Above.
        Formal Dx ~6/12 ago — Formal diagnosis made ~6 months ago.
        Cx / Tx / Lx — Cervical / Thoracic / Lumbar spine.
        ROM — Range of Motion (A = active, P = passive; record values).
        Modified 30s — Modified sit-to-stand test in 30 seconds.
        Gait speed — Record in m/s.
        Ax — Assessment.
        BBS — Berg Balance Scale.
        PMS — Physical Mobility Scale.
        Cont. — Continue.
        PMHx — Past Medical History.
        SC — Specialist Consultant / Service Coordinator (clarify in context).
        PMD — Personal Mobility Device.
        ERC — Equipment Resource Centre.
        CV — Cardiovascular.
        OT — Occupational Therapy.
        RV — Review.
        PT — Physiotherapy / Physical Therapy.
        COPD — Chronic Obstructive Pulmonary Disease.
        ADLs — Activities of Daily Living.
        SOB — Shortness of Breath.
        OM — Outcome Measures.

    These terms are not exhaustive. Use additional professional physiotherapy abbreviations as appropriate, and ensure all entries remain accurate and clinically relevant.

    Return the response as a marked down format, which will be used to convert to the word document later on.

    Format the progress note content as clear, professional paragraphs.
    """
)

content = f"""
    Please generate the initial note for me
"""
# print(file_content)
result = primary_agent.run_sync(content)
assess_result = result.data
print(f"Progress Note Data: {assess_result}")

# File paths
output_path = picked_case['output_file']


# Replace the last few lines with this:
try:
    # Save markdown content to file first
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(assess_result)
    
    # Convert markdown to Word document
    word_output_path = output_path.replace('.docx', '_converted.docx')
    convert_markdown_to_word(output_path, word_output_path)
    print(f"Final Word document saved to: {word_output_path}")
    
except Exception as e:
    print(f"Error in document conversion: {e}")
    print("Markdown content was still saved to:", output_path)