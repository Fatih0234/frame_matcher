from label_studio_sdk import Client
import os
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

project = ls.get_project(project_id)

export_result = project.export_snapshot_create(
    title='Export with Interpolated Keyframes',
    interpolate_key_frames=True
)

export_id = export_result['id']

status, filename = project.export_snapshot_download(
    export_id, export_type='JSON_MIN', path='.'
)