from flask_backend import app
from model_loader import load_ncf_model  # Import from model_loader
from neural_collaborative_filtering import predict_user_item_interactions
from supabase_storage import download_file_if_needed
import pandas as pd

def test_collaborative_filtering():
    with app.app_context():  # Ensure Flask application context is active
        # Download necessary files
        user_item_matrix_path = download_file_if_needed("user_item_matrix.csv")
        model_path = download_file_if_needed("ncf_model.pth")

        # Load the user-item matrix
        print(f"Loading user-item matrix from {user_item_matrix_path}...")
        user_item_matrix = pd.read_csv(user_item_matrix_path, index_col=0)
        print(f"User-item matrix loaded successfully with shape: {user_item_matrix.shape}")

        # Load the NCF model
        print(f"Loading NCF model from {model_path}...")
        num_users, num_items = user_item_matrix.shape
        model = load_ncf_model(
            model_path=model_path,
            user_item_matrix=user_item_matrix,
            num_users=num_users,
            num_items=num_items,
            latent_dim=20,
            hidden_dim=64,
            dropout_rate=0.3
        )
        print("NCF model loaded successfully.")

        # Test predictions for a specific user
        user_id = 0  # Test with the first user
        top_k = 5
        matrix_index_to_item_id = {i: item_id for i, item_id in enumerate(user_item_matrix.columns)}

        print(f"Generating top-{top_k} recommendations for user_id={user_id}...")
        recommendations = predict_user_item_interactions(
            model=model,
            user_item_matrix=user_item_matrix.values,
            user_id=user_id,
            top_k=top_k,
            matrix_index_to_item_id=matrix_index_to_item_id
        )
        print(f"Recommendations for user_id={user_id}: {recommendations}")

if __name__ == "__main__":
    test_collaborative_filtering()
