import os
import io
from typing import List, Dict, Any, Tuple
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from datetime import datetime, timedelta, timezone
from supabase import create_client, Client
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

# Scopes needed for Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

load_dotenv()

@dataclass
class ProcessedChunk:
    chunk_number: int
    content: str  
    metadata: Dict[str, Any]
    embedding: List[float]

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class GoogleDriveFolderReader:
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.authenticate()
    
    def authenticate(self):
        """Authenticate and create Google Drive service"""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        
        # If no valid credentials, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=11111)
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        print("Successfully authenticated with Google Drive API")
    
    def find_folder_by_name(self, folder_name, parent_id=None):
        """Find folder ID by name"""
        try:
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
            if parent_id:
                query += f" and '{parent_id}' in parents"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name, parents)"
            ).execute()
            
            folders = results.get('files', [])
            
            if not folders:
                print(f"No folder found with name: {folder_name}")
                return None
            elif len(folders) > 1:
                print(f"Multiple folders found with name '{folder_name}':")
                for i, folder in enumerate(folders):
                    print(f"  {i+1}. ID: {folder['id']}")
                return folders[0]['id']  # Return first one
            else:
                print(f"Found folder '{folder_name}' with ID: {folders[0]['id']}")
                return folders[0]['id']
                
        except HttpError as error:
            print(f"Error finding folder: {error}")
            return None
    
    def get_files_in_folder(self, folder_id, include_subfolders=True):
        """Get all files in a folder (and optionally subfolders)"""
        all_files = []
        
        try:
            # Get files directly in this folder
            query = f"'{folder_id}' in parents"
            
            page_token = None
            while True:
                results = self.service.files().list(
                    q=query,
                    pageSize=100,
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime, parents)",
                    pageToken=page_token
                ).execute()
                
                files = results.get('files', [])
                
                for file in files:
                    if file['mimeType'] == 'application/vnd.google-apps.folder':
                        # It's a folder
                        if include_subfolders:
                            print(f"Found subfolder: {file['name']}")
                            # Recursively get files from subfolder
                            subfolder_files = self.get_files_in_folder(file['id'], include_subfolders)
                            all_files.extend(subfolder_files)
                    else:
                        # It's a file
                        all_files.append(file)
                
                page_token = results.get('nextPageToken')
                if not page_token:
                    break
            
            return all_files
            
        except HttpError as error:
            print(f"Error getting files from folder: {error}")
            return []
    
    def is_readable_file(self, mime_type):
        """Check if file type is readable as text"""
        readable_types = [
            'text/plain',
            'application/json',
            'text/csv',
            'application/xml',
            'text/xml',
            'text/html',
            'text/css',
            'text/javascript',
            'application/javascript',
            'text/markdown',
            'application/rtf',
            'application/vnd.google-apps.document',  # Google Docs
            'application/vnd.google-apps.spreadsheet',  # Google Sheets
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',  # Word
        ]
        return mime_type in readable_types
    
    def read_file_content(self, file_id, mime_type, file_name):
        """Read content from a Google Drive file"""
        try:
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                request = self.service.files().export_media(
                    fileId=file_id, mimeType='text/plain')
            elif mime_type == 'application/vnd.google-apps.spreadsheet':
                # Export Google Sheet as CSV
                request = self.service.files().export_media(
                    fileId=file_id, mimeType='text/csv')
            elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # For Word docs, export as plain text
                request = self.service.files().export_media(
                    fileId=file_id, mimeType='text/plain')
            else:
                # Download regular file
                request = self.service.files().get_media(fileId=file_id)
            
            file_content = io.BytesIO()
            downloader = MediaIoBaseDownload(file_content, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Get content as string
            content = file_content.getvalue().decode('utf-8')
            return content
            
        except HttpError as error:
            print(f"Error reading file {file_name}: {error}")
            return None
        except UnicodeDecodeError as error:
            print(f"Could not decode file {file_name} as UTF-8: {error}")
            # Try with different encoding
            try:
                content = file_content.getvalue().decode('latin1')
                return content
            except:
                return None
        except Exception as error:
            print(f"Unexpected error reading {file_name}: {error}")
            return None
        
    def chunk_text(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into chunks, respecting code blocks and paragraphs."""
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Calculate end position
            end = start + chunk_size

            # If we're at the end of the text, just take what's left
            if end >= text_length:
                chunks.append(text[start:].strip())
                break

            # Try to find a code block boundary first (```)
            chunk = text[start:end]
            code_block = chunk.rfind('```')
            if code_block != -1 and code_block > chunk_size * 0.3:
                end = start + code_block

            # If no code block, try to break at a paragraph
            elif '\n\n' in chunk:
                # Find the last paragraph break
                last_break = chunk.rfind('\n\n')
                if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                    end = start + last_break

            # If no paragraph break, try to break at a sentence
            elif '. ' in chunk:
                # Find the last sentence break
                last_period = chunk.rfind('. ')
                if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                    end = start + last_period + 1

            # Extract chunk and clean it up
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk
            start = max(start + 1, end)

        return chunks
    
    def read_folder(self, folder_name_or_id, include_subfolders=True, save_to_disk=True, output_dir="downloaded_files"):
        """Read all files from a Google Drive folder"""
        
        # Determine if input is folder name or ID
        if len(folder_name_or_id) > 20 and folder_name_or_id.replace('_', '').replace('-', '').isalnum():
            # Looks like a folder ID
            folder_id = folder_name_or_id
            print(f"Using folder ID: {folder_id}")
        else:
            # Treat as folder name
            folder_id = self.find_folder_by_name(folder_name_or_id)
            if not folder_id:
                return []
        
        print(f"Reading files from folder {'(including subfolders)' if include_subfolders else '(current folder only)'}")
        
        # Get all files in folder
        files = self.get_files_in_folder(folder_id, include_subfolders)
        print(f"Found {len(files)} files")
        
        # Create output directory if saving to disk
        if save_to_disk:
            os.makedirs(output_dir, exist_ok=True)
        
        file_contents = []
        
        for i, file_info in enumerate(files, 1):
            file_name = file_info['name']
            file_id = file_info['id']
            mime_type = file_info['mimeType']
            size = file_info.get('size', 'unknown')
            
            print(f"\n[{i}/{len(files)}] Processing: {file_name}")
            print(f"  Type: {mime_type}")
            print(f"  Size: {size} bytes" if size != 'unknown' else "  Size: unknown")
            
            if self.is_readable_file(mime_type):
                print("  Reading content...")
                content = self.read_file_content(file_id, mime_type, file_name)
                
                if content:
                    file_data = {
                        'name': file_name,
                        'id': file_id,
                        'mime_type': mime_type,
                        'size': size,
                        'content': content
                    }
                    file_contents.append(file_data)
                    
                    print(f"  ✓ Successfully read {len(content)} characters")
                    
                    # Save to disk if requested
                    if save_to_disk:
                        # Clean filename for saving
                        safe_filename = "".join(c for c in file_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                        if not safe_filename:
                            safe_filename = f"file_{file_id}"
                        
                        # Add .txt extension if no extension
                        if '.' not in safe_filename:
                            safe_filename += '.txt'
                        
                        file_path = os.path.join(output_dir, safe_filename)
                        
                        # Handle duplicate names
                        counter = 1
                        original_path = file_path
                        while os.path.exists(file_path):
                            name, ext = os.path.splitext(original_path)
                            file_path = f"{name}_{counter}{ext}"
                            counter += 1
                        
                        try:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                f.write(content)
                            print(f"  ✓ Saved to: {file_path}")
                        except Exception as e:
                            print(f"  ✗ Error saving file: {e}")
                else:
                    print("  ✗ Failed to read content")
            else:
                print(f"  ⊘ Skipping non-text file (type: {mime_type})")
        
        print(f"\n{'='*50}")
        print(f"Summary: Successfully read {len(file_contents)} out of {len(files)} files")
        
        return file_contents
    
    def list_folders(self, parent_folder_name=None):
        """List all folders (optionally under a parent folder)"""
        try:
            query = "mimeType='application/vnd.google-apps.folder'"
            
            if parent_folder_name:
                parent_id = self.find_folder_by_name(parent_folder_name)
                if parent_id:
                    query += f" and '{parent_id}' in parents"
                else:
                    print(f"Parent folder '{parent_folder_name}' not found")
                    return []
            
            results = self.service.files().list(
                q=query,
                pageSize=50,
                fields="files(id, name, parents)"
            ).execute()
            
            folders = results.get('files', [])
            
            print(f"Found {len(folders)} folders:")
            for folder in folders:
                print(f"  - {folder['name']} (ID: {folder['id']})")
            
            return folders
            
        except HttpError as error:
            print(f"Error listing folders: {error}")
            return []
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding vector from OpenAI."""
        try:
            response = openai_client.embeddings.create(
                model= embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0] * 1536  # Return zero vector on error
        
    def process_file(self, file: str) -> dict[str, any]:
        """Process a single text file and extract its title, content, and category."""
        try:
            file_name = file['name']
            file_id = file['id']
            mime_type = file['mimeType']
            size = file.get('size', 'unknown')
            
            print(f"  Type: {mime_type}")
            print(f"  Size: {size} bytes" if size != 'unknown' else "  Size: unknown")
            
            if self.is_readable_file(mime_type):
                print("  Reading content...")
                content = self.read_file_content(file_id, mime_type, file_name)
                
                if content:
                    file_data = {
                        'name': file_name,
                        'id': file_id,
                        'mime_type': mime_type,
                        'size': size,
                        'content': content
                    }
                    return file_data
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return file_name, f"Error reading file: {str(e)}", "Other"
    
    def process_chunk(self, chunk: str, chunk_number: int, file_date:dict[str, any]) -> dict[str, any]:
        """Process a single chunk of text."""
        # Get embedding
        embedding = self.get_embedding(chunk)
        # Create metadata
        metadata = {
            "source": f"blob",
            "file_id": file_date['id'],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "blobType": "text/plain"
        }
        
        return ProcessedChunk(
            chunk_number=chunk_number,
            content=chunk,
            metadata=metadata,
            embedding=embedding
        )
    
    def insert_chunk(self, chunk: ProcessedChunk):
        """Insert a processed chunk into Supabase."""
        try:
            data = {
                "content": chunk.content,	
                "metadata": chunk.metadata,
                "embedding": chunk.embedding
            }

            # todo delete based on file id
            print(chunk.metadata["file_id"])
            result = supabase.table("documents").delete().eq('metadata->>file_id', chunk.metadata["file_id"]).execute()
            deleted_count = len(result.data) if result.data else 0
            print(f"Deleted {deleted_count} records with file_id in metadata: {chunk.metadata["file_id"]}")
            
            result = supabase.table("documents").insert(data).execute()
            print(f"Inserted chunk {chunk.chunk_number} for {chunk.metadata["file_id"]}")
            return result
        except Exception as e:
            print(f"Error inserting chunk: {e}")
            return None
        
    def get_changed_files(self, minutes_back=60):
        """Get files changed or created in the last N minutes"""
        try:
            # Calculate time threshold
            time_threshold = datetime.utcnow() - timedelta(minutes=minutes_back)
            time_str = time_threshold.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            
            # Query for recently modified files
            query = f"modifiedTime > '{time_str}'"
            
            results = self.service.files().list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, createdTime, parents, size)"
            ).execute()
            
            files = results.get('files', [])
            print(f"Found {len(files)} changed/created files in last {minutes_back} minutes")

            for file in files:
                file_data = self.process_file(file)
                if file_data:
                    chunks = self.chunk_text(file_data['content'])

                    # Use ThreadPoolExecutor for parallel processing
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        # Process chunks in parallel
                        chunk_futures = [
                            executor.submit(self.process_chunk, chunk, i, file_data)
                            for i, chunk in enumerate(chunks)
                        ]
                        processed_chunks = [future.result() for future in chunk_futures]
                        
                        # Store chunks in parallel
                        insert_futures = [
                            executor.submit(self.insert_chunk, chunk)
                            for chunk in processed_chunks
                        ]
                        [future.result() for future in insert_futures]
            
            return files
        
        except HttpError as error:
            print(f"An error occurred: {error}")
            return []

def main():
    """Example usage"""
    print("Google Drive Folder Reader")
    print("=" * 50)
    
    # Initialize reader
    reader = GoogleDriveFolderReader()
    
    # Example 1: List all folders
    print("\n1. Listing folders in your Drive:")
    # reader.list_folders()
    
    # Example 2: Read files from a specific folder
    # folder_name = input("\nEnter folder name to read files from (or press Enter to skip): ").strip()
    
    # if folder_name:
    #     print(f"\n2. Reading files from folder: {folder_name}")
    #     files_content = reader.read_folder(
    #         folder_name, 
    #         include_subfolders=True, 
    #         save_to_disk=True,
    #         output_dir=f"downloads_{folder_name.replace(' ', '_')}"
    #     )
        
    #     # Display summary
    #     if files_content:
    #         print(f"\nRead {len(files_content)} files:")
    #         for file_data in files_content:
    #             print(f"  - {file_data['name']}: {len(file_data['content'])} characters")

    
    reader.get_changed_files(minutes_back=180)
    # Example 3: Read files by folder ID (if you know it)
    # folder_id = "your_folder_id_here"
    # files_content = reader.read_folder(folder_id, include_subfolders=True)

if __name__ == "__main__":
    main()