from functools import lru_cache
import pandas as pd
from model_loader import load_ncf_model  # Import from model_loader
from supabase_storage import download_file_if_needed

@lru_cache()
def get_user_item_matrix():
    """
    Load the user-item matrix from Supabase or a cached file.
    """
    try:
        print("Loading user-item matrix...")  # Debugging log
        user_item_matrix_path = download_file_if_needed("user_item_matrix.csv")
        print(f"User-item matrix path: {user_item_matrix_path}")  # Debugging log
        user_item_matrix = pd.read_csv(user_item_matrix_path, index_col=0)
        print(f"User-item matrix loaded with shape: {user_item_matrix.shape}")  # Debugging log
        return user_item_matrix
    except Exception as e:
        print(f"Error loading user-item matrix: {e}")
        return None

@lru_cache()
def get_ncf_model():
    """
    Load the NCF model from Supabase or a cached file.
    """
    try:
        print("Loading NCF model...")  # Debugging log
        user_item_matrix = get_user_item_matrix()
        if user_item_matrix is None or user_item_matrix.empty:
            print("User-item matrix is empty or not loaded. Skipping NCF model loading.")
            return None

        num_users, num_items = user_item_matrix.shape
        model_path = download_file_if_needed("ncf_model.pth")
        print(f"NCF model path: {model_path}")  # Debugging log
        model = load_ncf_model(
            model_path=model_path,
            user_item_matrix=user_item_matrix,
            num_users=num_users,
            num_items=num_items,
            latent_dim=20,
            hidden_dim=64
        )
        print("NCF model loaded successfully.")  # Debugging log
        return model
    except Exception as e:
        print(f"Error loading NCF model: {e}")
        return None
