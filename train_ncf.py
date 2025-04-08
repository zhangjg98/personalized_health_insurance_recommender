import os
import pandas as pd
from database import db, Interaction, Item  # Import database models
from neural_collaborative_filtering import train_and_save_model, evaluate_model, load_ncf_model
from flask_sqlalchemy import SQLAlchemy
from flask import Flask

# Initialize a standalone Flask app for database access
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://recommender_user:securepassword@localhost/health_insurance_recommender'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def rebuild_user_item_matrix():
    """
    Rebuild the user-item matrix dynamically from the Interactions table.
    """
    print("Rebuilding user-item matrix from Interactions table...")  # Debugging log
    with app.app_context():
        interactions = Interaction.query.all()
        if not interactions:
            print("No interactions found in the database. Returning an empty matrix.")  # Debugging log
            return pd.DataFrame()

        # Create a DataFrame from interactions
        data = [(i.user_id, int(i.item_id), i.rating, i.timestamp) for i in interactions]  # Ensure item_id is an integer
        df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])

        # Handle duplicate (user_id, item_id) pairs by taking the most recent rating
        df = df.sort_values(by="timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="last")

        # Pivot the DataFrame to create the user-item matrix
        user_item_matrix = df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

        # Debugging log: Check the shape of the rebuilt matrix
        print(f"Rebuilt user-item matrix shape: {user_item_matrix.shape}")
        return user_item_matrix

# Rebuild the user-item matrix
user_item_matrix = rebuild_user_item_matrix()

# Validate the user-item matrix
if user_item_matrix.empty:
    print("The user-item matrix is empty. Skipping model training.")  # Debugging log
    exit(0)  # Exit gracefully without raising an error

# Save the rebuilt matrix to a CSV file
if not user_item_matrix.empty:
    user_item_matrix.to_csv("user_item_matrix.csv")
    print("User-item matrix saved to user_item_matrix.csv")  # Debugging log
else:
    print("User-item matrix is empty. Skipping save operation.")  # Debugging log

# Convert the matrix to a NumPy array
user_item_matrix = user_item_matrix.values

# Get the number of users and items
num_users, num_items = user_item_matrix.shape

# Check if the NCF model exists
if not os.path.exists("ncf_model.pth"):
    print("ncf_model.pth not found. Retraining the model from scratch...")
    NCF_MODEL = train_and_save_model(
        user_item_matrix,
        latent_dim=50,  # Ensure this matches the saved model's latent dimension
        hidden_dim=128,
        epochs=20,
        lr=0.001,
        model_path="ncf_model.pth"
    )
else:
    print("Loading the Neural Collaborative Filtering (NCF) model...")
    try:
        NCF_MODEL = load_ncf_model(
            model_path="ncf_model.pth",
            num_users=num_users,
            num_items=num_items,
            latent_dim=50,
            hidden_dim=128
        )
        print("NCF model loaded successfully.")  # Debugging log
    except (ValueError, RuntimeError) as e:
        print(f"Error loading NCF model: {e}. Retraining the model from scratch.")  # Debugging log
        NCF_MODEL = train_and_save_model(
            user_item_matrix,
            latent_dim=50,  # Ensure this matches the saved model's latent dimension
            hidden_dim=128,
            epochs=20,
            lr=0.001,
            model_path="ncf_model.pth"
        )

# Evaluate the model
try:
    mse, f1 = evaluate_model(NCF_MODEL, user_item_matrix)
    print(f"Model Evaluation - Mean Squared Error (MSE): {mse:.4f}, F1-score: {f1:.4f}")
except ValueError as e:
    print(f"Error during model evaluation: {e}")
