import os
from supabase_client import supabase

BUCKET_NAME = "models"  # Ensure this matches your Supabase bucket name
CACHE_DIR = "cached_files"
os.makedirs(CACHE_DIR, exist_ok=True)

def download_file_if_needed(remote_path, local_filename=None):
    """
    Download a file from Supabase if it is not already cached locally.

    Parameters:
        remote_path (str): The path to the file in the Supabase bucket.
        local_filename (str): Optional local filename to save the file as.

    Returns:
        str: The local path to the downloaded (or cached) file.
    """
    if not local_filename:
        local_filename = os.path.basename(remote_path)

    local_path = os.path.join(CACHE_DIR, local_filename)

    if not os.path.exists(local_path):
        print(f"Downloading {remote_path} from Supabase...")
        try:
            res = supabase.storage.from_(BUCKET_NAME).download(remote_path)
            with open(local_path, "wb") as f:
                f.write(res)
            print(f"Downloaded {remote_path} to {local_path}")
        except Exception as e:
            print(f"Error downloading {remote_path} from Supabase: {e}")
            raise
    else:
        print(f"Using cached file: {local_path}")

    return local_path
