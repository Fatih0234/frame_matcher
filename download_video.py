from label_studio_sdk import Client
import requests
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials from environment variables
url = os.getenv('LABEL_STUDIO_URL', 'http://localhost:8080')
api_key = os.getenv('LABEL_STUDIO_API_KEY')
project_id = int(os.getenv('PROJECT_ID', 5))

if not api_key:
    raise ValueError("LABEL_STUDIO_API_KEY not found in environment variables")

ls = Client(url=url, api_key=api_key)

# Get all tasks from the project
tasks = ls.get_project(project_id).get_tasks()

# Create a directory to save videos
os.makedirs('exported_videos', exist_ok=True)

for task in tasks:
    # Get the media file URL from task data
    if 'video' in task['data']:
        video_url = task['data']['video']
        
        # If it's a relative URL, make it absolute
        if video_url.startswith('/'):
            video_url = f"{ls.url}{video_url}"
        
        # Get filename from URL or use task ID
        parsed_url = urlparse(video_url)
        filename = os.path.basename(parsed_url.path) or f"task_{task['id']}.mp4"
        
        # Download the video file
        headers = {'Authorization': f'Token {ls.api_key}'}
        response = requests.get(video_url, headers=headers, stream=True)
        
        if response.status_code == 200:
            filepath = os.path.join('exported_videos', filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded: {filename}")
        else:
            print(f"Failed to download {video_url}: {response.status_code}")