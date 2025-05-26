import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import io
from googleapiclient.http import MediaIoBaseDownload

# Scopes needed for Drive API
SCOPES = ['https://www.googleapis.com/auth/drive']

class GoogleDriveMonitor:
    def __init__(self, credentials_file='credentials.json', token_file='token.json'):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self.file_registry = {}  # Store file info and unique IDs
        self.registry_file = 'file_registry.json'
        self.authenticate()
        self.load_registry()
    
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
                creds = flow.run_local_server(port=0)
            
            # Save credentials for next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        print("Successfully authenticated with Google Drive API")
    
    def load_registry(self):
        """Load existing file registry from disk"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    self.file_registry = json.load(f)
                print(f"Loaded {len(self.file_registry)} files from registry")
            except Exception as e:
                print(f"Error loading registry: {e}")
                self.file_registry = {}
    
    def save_registry(self):
        """Save file registry to disk"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.file_registry, f, indent=2)
        except Exception as e:
            print(f"Error saving registry: {e}")
    
    def generate_unique_id(self, file_info):
        """Generate unique ID for a file based on its properties"""
        # Create unique ID from file path, name, and drive file ID
        identifier = f"{file_info['name']}_{file_info['id']}_{file_info.get('parents', [''])[0]}"
        return hashlib.md5(identifier.encode()).hexdigest()
    
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
            
            return files
            
        except HttpError as error:
            print(f"An error occurred: {error}")
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
            'application/vnd.google-apps.document',  # Google Docs
            'application/vnd.google-apps.spreadsheet',  # Google Sheets
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
        except UnicodeDecodeError:
            print(f"Could not decode file {file_name} as text")
            return None
    
    def process_file(self, file_info):
        """Process a single file - generate ID and read content"""
        file_id = file_info['id']
        file_name = file_info['name']
        mime_type = file_info['mimeType']
        
        # Generate unique ID
        unique_id = self.generate_unique_id(file_info)
        
        # Check if we've seen this file before
        if unique_id in self.file_registry:
            last_modified = self.file_registry[unique_id].get('last_modified')
            current_modified = file_info['modifiedTime']
            
            if last_modified == current_modified:
                print(f"File {file_name} unchanged, skipping")
                return
        
        print(f"\nProcessing file: {file_name}")
        print(f"Drive ID: {file_id}")
        print(f"Unique ID: {unique_id}")
        print(f"MIME Type: {mime_type}")
        
        # Update registry
        self.file_registry[unique_id] = {
            'drive_id': file_id,
            'name': file_name,
            'mime_type': mime_type,
            'last_modified': file_info['modifiedTime'],
            'created_time': file_info['createdTime'],
            'size': file_info.get('size', 'unknown')
        }
        
        # Try to read content if it's a readable file type
        if self.is_readable_file(mime_type):
            print("Reading file content...")
            content = self.read_file_content(file_id, mime_type, file_name)
            
            if content:
                self.file_registry[unique_id]['content_preview'] = content[:500]  # Store first 500 chars
                print(f"Content preview:\n{content[:200]}...")
                
                # Optionally save full content to separate file
                content_filename = f"content_{unique_id}.txt"
                try:
                    with open(content_filename, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Full content saved to: {content_filename}")
                    self.file_registry[unique_id]['content_file'] = content_filename
                except Exception as e:
                    print(f"Error saving content file: {e}")
        else:
            print(f"File type {mime_type} not readable as text")
    
    def monitor_changes(self, minutes_back=60, continuous=False, interval=300):
        """Monitor Google Drive for changes"""
        if continuous:
            print(f"Starting continuous monitoring (checking every {interval} seconds)")
            while True:
                try:
                    files = self.get_changed_files(minutes_back)
                    
                    for file_info in files:
                        self.process_file(file_info)
                    
                    self.save_registry()
                    
                    if files:
                        print(f"\nProcessed {len(files)} files")
                    
                    print(f"Waiting {interval} seconds before next check...")
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    print("\nMonitoring stopped by user")
                    break
                except Exception as e:
                    print(f"Error during monitoring: {e}")
                    time.sleep(60)  # Wait a minute before retrying
        else:
            # Single run
            files = self.get_changed_files(minutes_back)
            
            for file_info in files:
                self.process_file(file_info)
            
            self.save_registry()
            print(f"\nProcessed {len(files)} files")
    
    def list_registry(self):
        """Display current file registry"""
        print(f"\nFile Registry ({len(self.file_registry)} files):")
        print("-" * 80)
        
        for unique_id, info in self.file_registry.items():
            print(f"Unique ID: {unique_id}")
            print(f"Name: {info['name']}")
            print(f"Drive ID: {info['drive_id']}")
            print(f"Last Modified: {info['last_modified']}")
            print(f"Size: {info['size']}")
            if 'content_file' in info:
                print(f"Content File: {info['content_file']}")
            print("-" * 40)

def main():
    """Main function to demonstrate usage"""
    print("Google Drive File Monitor")
    print("=" * 50)
    
    # Initialize monitor
    monitor = GoogleDriveMonitor()
    
    # Example usage
    print("\n1. Checking for files changed in last 60 minutes...")
    monitor.monitor_changes(minutes_back=60, continuous=False)
    
    print("\n2. Current registry:")
    monitor.list_registry()
    
    # Uncomment below for continuous monitoring
    # print("\n3. Starting continuous monitoring...")
    # monitor.monitor_changes(minutes_back=30, continuous=True, interval=300)

if __name__ == "__main__":
    main()