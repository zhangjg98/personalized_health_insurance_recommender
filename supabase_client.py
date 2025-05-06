import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL or Service Role Key is missing. Check your environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Supabase client initialized successfully.")
