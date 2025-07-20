# Google Drive Integration Setup

This guide explains how to set up Google Drive integration to automatically upload your YOLO/COCO datasets to Google Drive.

## Prerequisites

1. A Google account with Google Drive access
2. Google Cloud Console project with Drive API enabled

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Go to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click on it and press "Enable"

## Step 2: Create Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in required fields (app name, user support email, developer email)
   - Add your email to test users
   - Save and continue through all steps
4. Back in Credentials, click "Create Credentials" > "OAuth client ID"
5. Choose "Desktop application"
6. Give it a name (e.g., "Frame Matcher Drive Upload")
7. Download the JSON file
8. Rename it to `credentials.json` and place it in your project root directory

## Step 3: Install Dependencies

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

Or if you've already updated requirements.txt:

```bash
pip install -r requirements.txt
```

## Step 4: First Time Authentication

The first time you use Google Drive upload, you'll be prompted to authenticate:

1. A browser window will open
2. Sign in to your Google account
3. Grant permissions to the app
4. The authentication token will be saved as `token.json` for future use

## Usage Examples

### Basic Upload
```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"pedestrian":2}' \
  --output ./dataset \
  --upload-to-drive
```

### Upload with Custom Folder Name
```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"pedestrian":2}' \
  --output ./dataset \
  --upload-to-drive \
  --drive-folder "My_YOLO_Dataset_2024"
```

### Upload to Specific Parent Folder
```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"pedestrian":2}' \
  --output ./dataset \
  --upload-to-drive \
  --drive-folder "Training_Data" \
  --drive-parent-id "1a2b3c4d5e6f7g8h9i0j"
```

## How to Find Google Drive Folder ID

1. Open the folder in Google Drive web interface
2. Look at the URL: `https://drive.google.com/drive/folders/FOLDER_ID_HERE`
3. Copy the FOLDER_ID_HERE part

## File Structure in Google Drive

Your dataset will be uploaded with this structure:

```
Your_Dataset_Folder/
├── images/
│   ├── frame_video1_000001.jpg
│   ├── frame_video1_000002.jpg
│   └── ...
├── labels/
│   ├── frame_video1_000001.txt
│   ├── frame_video1_000002.txt
│   └── ...
├── data.yaml
└── classes.txt
```

## Troubleshooting

### "credentials.json not found"
- Make sure you've downloaded the OAuth credentials from Google Cloud Console
- Rename the file to exactly `credentials.json`
- Place it in the same directory as `main.py`

### "Authentication failed"
- Delete `token.json` and try again
- Make sure you're using the correct Google account
- Check that the Google Drive API is enabled in your project

### "Permission denied"
- Make sure your Google account has access to the parent folder (if specified)
- Check that the OAuth consent screen is properly configured

### "Quota exceeded"
- Google Drive API has daily quotas
- For large datasets, consider uploading in smaller batches
- Check your quota usage in Google Cloud Console

## Security Notes

- Keep your `credentials.json` file secure and don't commit it to version control
- The `token.json` file contains your access token - also keep it secure
- Both files are already included in the `.gitignore`

## Alternative: Google Drive Desktop Client

If you prefer a simpler approach without API setup:

1. Install Google Drive Desktop client
2. Sync a local folder with Google Drive
3. Set your output path to that synced folder:

```bash
python main.py \
  --format yolo \
  --classes '{"cyclist":0,"person":1,"pedestrian":2}' \
  --output "~/Google Drive/My Datasets/yolo_dataset"
```
