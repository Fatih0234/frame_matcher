"""
Google Drive integration utility for uploading dataset files.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

class GoogleDriveUploader:
    """Handle Google Drive operations for uploading datasets."""
    
    # Google Drive API scopes
    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    
    def __init__(self, credentials_file: str = 'credentials.json', token_file: str = 'token.json'):
        """
        Initialize Google Drive uploader.
        
        Args:
            credentials_file: Path to Google API credentials JSON file
            token_file: Path to store/load authentication token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.service = None
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Drive API."""
        creds = None
        
        # Load existing token
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, self.SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"Google API credentials file '{self.credentials_file}' not found. "
                        "Please download it from Google Cloud Console."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        self.service = build('drive', 'v3', credentials=creds)
        print("✅ Successfully authenticated with Google Drive")
    
    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """
        Create a folder in Google Drive.
        
        Args:
            folder_name: Name of the folder to create
            parent_folder_id: ID of parent folder (None for root)
            
        Returns:
            ID of the created folder
        """
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_folder_id:
            folder_metadata['parents'] = [parent_folder_id]
        
        folder = self.service.files().create(body=folder_metadata, fields='id').execute()
        folder_id = folder.get('id')
        
        print(f"📁 Created folder '{folder_name}' with ID: {folder_id}")
        return folder_id
    
    def find_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> Optional[str]:
        """
        Find a folder by name in Google Drive.
        
        Args:
            folder_name: Name of the folder to find
            parent_folder_id: ID of parent folder to search in
            
        Returns:
            Folder ID if found, None otherwise
        """
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'"
        if parent_folder_id:
            query += f" and '{parent_folder_id}' in parents"
        
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        folders = results.get('files', [])
        
        if folders:
            folder_id = folders[0]['id']
            print(f"📁 Found folder '{folder_name}' with ID: {folder_id}")
            return folder_id
        
        return None
    
    def get_or_create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """
        Get existing folder or create new one.
        
        Args:
            folder_name: Name of the folder
            parent_folder_id: ID of parent folder
            
        Returns:
            Folder ID
        """
        folder_id = self.find_folder(folder_name, parent_folder_id)
        if not folder_id:
            folder_id = self.create_folder(folder_name, parent_folder_id)
        return folder_id
    
    def upload_file(self, local_path: Path, folder_id: str, 
                   filename: Optional[str] = None) -> str:
        """
        Upload a file to Google Drive.
        
        Args:
            local_path: Path to local file
            folder_id: ID of destination folder
            filename: Custom filename (uses local filename if None)
            
        Returns:
            ID of uploaded file
        """
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        if filename is None:
            filename = local_path.name
        
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(str(local_path))
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        file_id = file.get('id')
        print(f"📤 Uploaded '{filename}' with ID: {file_id}")
        return file_id
    
    def upload_dataset(self, dataset_path: Path, drive_folder_name: str,
                      parent_folder_id: Optional[str] = None) -> Dict[str, str]:
        """
        Upload entire YOLO dataset to Google Drive.
        
        Args:
            dataset_path: Path to local dataset directory
            drive_folder_name: Name for the dataset folder in Drive
            parent_folder_id: ID of parent folder in Drive
            
        Returns:
            Dictionary with uploaded file information
        """
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        print(f"🚀 Starting upload of dataset from {dataset_path}")
        
        # Create main dataset folder
        dataset_folder_id = self.get_or_create_folder(drive_folder_name, parent_folder_id)
        
        uploaded_files = {}
        
        # Upload images folder
        images_dir = dataset_path / "images"
        if images_dir.exists():
            images_folder_id = self.get_or_create_folder("images", dataset_folder_id)
            
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            print(f"📸 Uploading {len(image_files)} images...")
            
            for i, image_file in enumerate(image_files, 1):
                file_id = self.upload_file(image_file, images_folder_id)
                uploaded_files[f"image_{i}"] = file_id
                
                if i % 100 == 0:  # Progress update every 100 files
                    print(f"📸 Uploaded {i}/{len(image_files)} images...")
        
        # Upload labels folder
        labels_dir = dataset_path / "labels"
        if labels_dir.exists():
            labels_folder_id = self.get_or_create_folder("labels", dataset_folder_id)
            
            label_files = list(labels_dir.glob("*.txt"))
            print(f"🏷️  Uploading {len(label_files)} label files...")
            
            for i, label_file in enumerate(label_files, 1):
                file_id = self.upload_file(label_file, labels_folder_id)
                uploaded_files[f"label_{i}"] = file_id
                
                if i % 100 == 0:  # Progress update every 100 files
                    print(f"🏷️  Uploaded {i}/{len(label_files)} labels...")
        
        # Upload config files
        config_files = ['data.yaml', 'classes.txt']
        for config_file in config_files:
            config_path = dataset_path / config_file
            if config_path.exists():
                file_id = self.upload_file(config_path, dataset_folder_id)
                uploaded_files[config_file] = file_id
        
        print(f"✅ Dataset upload complete! Total files uploaded: {len(uploaded_files)}")
        print(f"📁 Dataset folder ID: {dataset_folder_id}")
        
        return uploaded_files
    
    def get_folder_url(self, folder_id: str) -> str:
        """Get shareable URL for a Google Drive folder."""
        return f"https://drive.google.com/drive/folders/{folder_id}"
