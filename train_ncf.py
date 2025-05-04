import os
import pandas as pd
from database import db, Interaction
from neural_collaborative_filtering import train_and_save_model, load_ncf_model
from evaluation_metrics import evaluate_model_metrics # Import from evaluation_metrics.py
from flask import Flask
from plans import PLANS
from dotenv import load_dotenv
import requests

load_dotenv()

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
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
    # Dummy user input for evaluation
    user_input = {
        "age": "adult",
        "smoker": "no",
        "bmi": "",
        "income": "",
        "family_size": "",
        "chronic_condition": "no",
        "medical_care_frequency": "Low",
        "preferred_plan_type": "",
        "priority": "",
        "gender": "",
        "ethnicity": ""
    }
    # Calculate metrics
    k_percentage = 0.5  # Adjust k value as a percentage of the number of items
    num_items = user_item_matrix.shape[1]
    k_value = int(num_items * k_percentage)
    metrics = evaluate_model_metrics(NCF_MODEL, user_item_matrix, k=k_value, user_inputs=user_input, plans=PLANS)
    print(f"Precision@{k_value}: {metrics['Precision@K']:.4f}")
    print(f"Recall@{k_value}: {metrics['Recall@K']:.4f}")
    print(f"NDCG@{k_value}: {metrics['NDCG@K']:.4f}")
    print(f"Hit Rate@{k_value}: {metrics['Hit Rate']:.4f}")
except ValueError as e:
    print(f"Error during model evaluation: {e}")

def upload_to_supabase(filename, bucket, supabase_url, api_key):
    storage_url = f"{supabase_url}/storage/v1/object/{bucket}/{filename}"
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/octet-stream"
    }

    with open(filename, "rb") as f:
        response = requests.put(storage_url, headers=headers, data=f)
        if response.status_code in [200, 201]:
            print(f"Successfully uploaded {filename} to Supabase bucket '{bucket}'.")
        else:
            print(f"Failed to upload {filename}: {response.status_code}, {response.text}")

# Upload files to Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_API_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
BUCKET_NAME = "models"

if SUPABASE_URL and SUPABASE_API_KEY:
    upload_to_supabase("ncf_model.pth", BUCKET_NAME, SUPABASE_URL, SUPABASE_API_KEY)
    upload_to_supabase("user_item_matrix.csv", BUCKET_NAME, SUPABASE_URL, SUPABASE_API_KEY)
else:
    print("Supabase credentials not set. Skipping upload.")
