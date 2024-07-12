import gdown
import zipfile
from pathlib import Path

# Google Drive file ID
file_id = '1UMq0CP20lKcraOTvsFMjiLjPfDam9jAp'

# URL to download the file
url = f'https://drive.google.com/uc?id={file_id}'

# Local file path
output_dir = Path('../data')
output_dir.mkdir(parents=True, exist_ok=True)

# Local file path for the zip file
output_zip = output_dir / 'pidray.zip'
print(output_zip.absolute())

# Download the file
gdown.download(url, str(output_zip), quiet=False)

# Extract the zip file
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f'Files extracted to {output_dir}')
