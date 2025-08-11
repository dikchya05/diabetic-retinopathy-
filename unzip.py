import zipfile
import os

# Path to your ZIP file and extraction directory
zip_file_path = 'aptos2019-blindness-detection.zip'
extract_to_path = './data'

# Create the target directory if it doesn't exist
os.makedirs(extract_to_path, exist_ok=True)

# Extract the ZIP file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)

print(f"Extracted files to {extract_to_path}")
