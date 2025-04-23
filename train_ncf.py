import os
import pandas as pd
from database import db, Interaction
from neural_collaborative_filtering import train_and_save_model, evaluate_model, load_ncf_model
from flask import Flask

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://recommender_user:securepassword@localhost/health_insurance_recommender'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)

def rebuild_user_item_matrix():
    print("Rebuilding user-item matrix from Interactions table...")
    with app.app_context():
        interactions = Interaction.query.all()
        if not interactions:
            print("No interactions found in the database. Returning an empty matrix.")
            return pd.DataFrame()

        data = [(i.user_id, int(i.item_id), i.rating, i.timestamp) for i in interactions]
        df = pd.DataFrame(data, columns=["user_id", "item_id", "rating", "timestamp"])
        df = df.sort_values(by="timestamp").drop_duplicates(subset=["user_id", "item_id"], keep="last")
        user_item_matrix = df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
        print(f"Rebuilt user-item matrix shape: {user_item_matrix.shape}")
        return user_item_matrix

user_item_matrix = rebuild_user_item_matrix()

if user_item_matrix.empty:
    print("The user-item matrix is empty. Skipping model training.")
    exit(0)

user_item_matrix.to_csv("user_item_matrix.csv")
print("User-item matrix saved to user_item_matrix.csv")

user_item_matrix = user_item_matrix.values
num_users, num_items = user_item_matrix.shape

if not os.path.exists("ncf_model.pth"):
    print("ncf_model.pth not found. Retraining the model from scratch...")
    NCF_MODEL = train_and_save_model(
        user_item_matrix,
        latent_dim=50,
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
        print("NCF model loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading NCF model: {e}. Retraining the model from scratch.")
        NCF_MODEL = train_and_save_model(
            user_item_matrix,
            latent_dim=50,
            hidden_dim=128,
            epochs=20,
            lr=0.001,
            model_path="ncf_model.pth"
        )

try:
    mse, f1 = evaluate_model(NCF_MODEL, user_item_matrix)
    print(f"Model Evaluation - Mean Squared Error (MSE): {mse:.4f}, F1-score: {f1:.4f}")
except ValueError as e:
    print(f"Error during model evaluation: {e}")
