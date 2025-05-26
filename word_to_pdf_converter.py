import os
import sys
import argparse
from pathlib import Path

def convert_with_docx2pdf():
    """Convert using docx2pdf (requires Microsoft Word)"""
    try:
        from docx2pdf import convert
        return convert
    except ImportError:
        return None

def convert_with_libreoffice(input_file, output_file=None):
    """Convert using LibreOffice (requires LibreOffice installation)"""
    import subprocess
    
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.pdf'))
    
    # Determine the LibreOffice executable name based on OS
    if sys.platform == "win32":
        soffice_path = "soffice.exe"
    else:
        soffice_path = "soffice"
    
    try:
        subprocess.run([
            soffice_path,
            '--headless',
            '--convert-to', 'pdf',
            '--outdir', os.path.dirname(output_file),
            input_file
        ], check=True)
        
        # LibreOffice creates the PDF with the same name as input but with .pdf extension
        # If a different output name is requested, we need to rename
        default_output = str(Path(input_file).with_suffix('.pdf'))
        if output_file != default_output and os.path.exists(default_output):
            os.rename(default_output, output_file)
            
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def convert_with_comtypes(input_file, output_file=None):
    """Convert using comtypes (Windows only, requires Word)"""
    if sys.platform != "win32":
        return False
    
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.pdf'))
    
    try:
        import comtypes.client
        
        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        
        doc = word.Documents.Open(os.path.abspath(input_file))
        doc.SaveAs(os.path.abspath(output_file), FileFormat=17)  # 17 = PDF format
        doc.Close()
        word.Quit()
        
        return True
    except (ImportError, Exception):
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert Word documents to PDF')
    parser.add_argument('input_file', help='Path to the Word document')
    parser.add_argument('-o', '--output', help='Path to the output PDF file (optional)')
    args = parser.parse_args()
    
    input_file = args.input_file
    output_file = args.output
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return 1
    
    if not input_file.lower().endswith(('.doc', '.docx')):
        print(f"Error: Input file '{input_file}' is not a Word document (.doc or .docx).")
        return 1
    
    if output_file is None:
        output_file = str(Path(input_file).with_suffix('.pdf'))
    
    # Try different conversion methods
    print(f"Converting '{input_file}' to '{output_file}'...")
    
    # Method 1: docx2pdf
    docx2pdf_convert = convert_with_docx2pdf()
    if docx2pdf_convert:
        try:
            docx2pdf_convert(input_file, output_file)
            print(f"Conversion successful using docx2pdf! Output saved to: {output_file}")
            return 0
        except Exception as e:
            print(f"docx2pdf conversion failed: {e}")
    
    # Method 2: comtypes (Windows only)
    if sys.platform == "win32":
        print("Trying conversion with comtypes...")
        if convert_with_comtypes(input_file, output_file):
            print(f"Conversion successful using comtypes! Output saved to: {output_file}")
            return 0
    
    # Method 3: LibreOffice
    print("Trying conversion with LibreOffice...")
    if convert_with_libreoffice(input_file, output_file):
        print(f"Conversion successful using LibreOffice! Output saved to: {output_file}")
        return 0
    
    print("\nAll conversion methods failed. Please ensure one of the following is installed:")
    print("- Microsoft Word (for docx2pdf or comtypes methods)")
    print("- LibreOffice (for LibreOffice method)")
    print("\nYou may also need to install required Python packages:")
    print("pip install docx2pdf comtypes")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())