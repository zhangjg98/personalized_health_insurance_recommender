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

# Load the updated user-item matrix
try:
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
    print(f"Loaded user-item matrix with shape: {user_item_matrix.shape}")  # Debugging log
except FileNotFoundError:
    raise FileNotFoundError("The user_item_matrix.csv file was not found. Ensure it exists before training.")

# Validate the user-item matrix
if user_item_matrix.empty:
    raise ValueError("The user-item matrix is empty. Ensure that interactions are logged before training.")

# Check for missing or invalid values
if user_item_matrix.isnull().values.any():
    raise ValueError("The user-item matrix contains missing values. Please clean the data before training.")

# Ensure all values are numeric
if not pd.api.types.is_numeric_dtype(user_item_matrix.values):
    raise ValueError("The user-item matrix contains non-numeric values. Please clean the data before training.")

# Remove duplicate columns by aggregating their values (e.g., taking the mean)
user_item_matrix = user_item_matrix.groupby(user_item_matrix.columns, axis=1).mean()

# Debugging log: Check aggregated column names
print("Aggregated column names:", user_item_matrix.columns)

# Ensure all column names are integers
try:
    user_item_matrix.columns = user_item_matrix.columns.map(lambda x: int(float(x)) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x)
except ValueError as e:
    print(f"Error: Invalid column names in user-item matrix: {user_item_matrix.columns}")  # Debugging log
    raise ValueError("All column names in the user-item matrix must be valid integers.") from e

# Debugging log: Check sanitized column names
print("Sanitized column names:", user_item_matrix.columns)

# Validate column names in the user-item matrix
try:
    valid_item_ids = set(map(int, user_item_matrix.columns))  # Convert column names to integers
except ValueError as e:
    print(f"Error: Invalid column names in user-item matrix: {user_item_matrix.columns}")  # Debugging log
    raise ValueError("All column names in the user-item matrix must be valid integers.") from e

# Debugging log: Check valid item IDs
print("Valid item IDs:", valid_item_ids)

# Use the standalone Flask app context for database queries
with app.app_context():
    # Verify that all items in the matrix have interactions
    interactions = Interaction.query.all()
    interaction_item_ids = {i.item_id for i in interactions}

    # Log items with interactions but missing from the matrix
    missing_from_matrix = interaction_item_ids - valid_item_ids
    if missing_from_matrix:
        print(f"Warning: The following item_ids have interactions but are missing from the user-item matrix: {missing_from_matrix}")

    # Log items in the matrix but without interactions
    missing_interactions = valid_item_ids - interaction_item_ids
    if missing_interactions:
        print(f"Warning: The following item_ids are in the user-item matrix but have no interactions: {missing_interactions}")

    # Synchronize the matrix with the database interactions
    active_item_ids = interaction_item_ids & valid_item_ids
    if not active_item_ids:
        raise ValueError("No valid items with interactions found. Ensure the user-item matrix and interactions are consistent.")

    # Filter the matrix to include only items with interactions
    user_item_matrix = user_item_matrix.loc[:, user_item_matrix.columns.isin(active_item_ids)]
    print(f"Filtered user-item matrix shape: {user_item_matrix.shape}")  # Debugging log

# Stop if the matrix is empty after filtering
if user_item_matrix.empty:
    raise ValueError("Filtered user-item matrix is empty. Ensure that interactions are logged and consistent with the matrix.")

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
