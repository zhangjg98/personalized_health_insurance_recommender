import os
from supabase_client import supabase

BUCKET_NAME = "models"
CACHE_DIR = "cached_files"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_file_if_needed(remote_path, local_filename=None):
    if not local_filename:
        local_filename = os.path.basename(remote_path)

    local_path = os.path.join(CACHE_DIR, local_filename)

    if not os.path.exists(local_path):
        print(f"Downloading {remote_path} from Supabase...")
        res = supabase.storage.from_(BUCKET_NAME).download(remote_path)
        with open(local_path, "wb") as f:
            f.write(res)
    else:
        print(f"Using cached file: {local_path}")

    return local_path
