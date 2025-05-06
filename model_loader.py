import os
import torch
import pandas as pd
from models import NeuralCollaborativeFiltering  # Import from models.py
from training_utils import train_and_save_model  # Import from training_utils.py

def load_ncf_model(model_path="ncf_model.pth", user_item_matrix=None, num_users=None, num_items=None, latent_dim=20, hidden_dim=64, dropout_rate=0.3):
    """
    Load NCF model and ensure compatibility with the current matrix shape.
    Handles both metadata-included and legacy model files.
    Retrains the model if metadata does not match.
    """
    print(f"Loading NCF model from {model_path}...")

    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found. Retraining the model...")
        if isinstance(user_item_matrix, str):
            print(f"Loading user-item matrix from file: {user_item_matrix}")
            user_item_matrix = pd.read_csv(user_item_matrix, index_col=0).values  # Load as NumPy array
        if user_item_matrix is not None:
            return train_and_save_model(
                user_item_matrix=user_item_matrix,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                model_path=model_path
            )
        else:
            print("Error: user_item_matrix is required to retrain the model.")
            return None

    data = torch.load(model_path)

    # Determine if metadata is included
    if isinstance(data, dict) and 'model_state_dict' in data:
        # Load full config from saved metadata
        saved_num_users = data['num_users']
        saved_num_items = data['num_items']
        saved_latent_dim = data['latent_dim']
        saved_hidden_dim = data['hidden_dim']

        # Check for dimension mismatches
        if (saved_num_users != num_users or
            saved_num_items != num_items or
            saved_latent_dim != latent_dim or
            saved_hidden_dim != hidden_dim):
            print("Model architecture mismatch detected:")
            print(f"Saved model: num_users={saved_num_users}, num_items={saved_num_items}, latent_dim={saved_latent_dim}, hidden_dim={saved_hidden_dim}")
            print(f"Current model: num_users={num_users}, num_items={num_items}, latent_dim={latent_dim}, hidden_dim={hidden_dim}")
            print("Retraining the model with updated dimensions...")
            if isinstance(user_item_matrix, str):
                print(f"Loading user-item matrix from file: {user_item_matrix}")
                user_item_matrix = pd.read_csv(user_item_matrix, index_col=0).values  # Load as NumPy array
            if user_item_matrix is not None:
                return train_and_save_model(
                    user_item_matrix=user_item_matrix,
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    model_path=model_path
                )
            else:
                print("Error: user_item_matrix is required to retrain the model.")
                return None

        model = NeuralCollaborativeFiltering(
            num_users=saved_num_users,
            num_items=saved_num_items,
            latent_dim=saved_latent_dim,
            hidden_dim=saved_hidden_dim,
            dropout_rate=dropout_rate
        )
        model.load_state_dict(data['model_state_dict'])
    else:
        # Fallback for old format: raw state_dict only
        if num_users is None or num_items is None:
            print("Error: num_users and num_items must be provided for legacy model files.")
            return None

        model = NeuralCollaborativeFiltering(num_users, num_items, latent_dim, hidden_dim, dropout_rate)
        model.load_state_dict(data)

    print("Model loaded successfully.")
    return model
