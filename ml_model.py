import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from models import DeepAutoencoder
from thresholds import unified_thresholds
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Define hyperparameters used during training
n_components = 9
num_hidden_1 = 200
num_hidden_2 = 100
num_hidden_3 = 50
num_latent = 25

def load_trained_objects():
    # Create an instance of the model with the same architecture as used in training
    final_rbm = DeepAutoencoder(num_visible=n_components,
                                num_hidden_1=num_hidden_1,
                                num_hidden_2=num_hidden_2,
                                num_hidden_3=num_hidden_3,
                                num_latent=num_latent)
    final_rbm.load_state_dict(torch.load('final_autoencoder.pth'))
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    return final_rbm, scaler, pca

# Load the objects once when the module is imported
FINAL_RBM, SCALER, PCA_MODEL = load_trained_objects()

# Load thresholds dynamically
THRESHOLDS = unified_thresholds(
    "state_level_insights.csv",
    keys=[
        "TOT_MDCR_STDZD_PYMT_PC", 
        "TOT_MDCR_PYMT_PC", 
        "BENE_AVG_RISK_SCRE", 
        "IP_CVRD_STAYS_PER_1000_BENES", 
        "ER_VISITS_PER_1000_BENES", 
        "MA_PRTCPTN_RATE",
        'BENE_DUAL_PCT'
    ]
)

def classify_spending(value, key):
    """
    Classify spending level based on dynamic thresholds.
    """
    if key not in THRESHOLDS:
        return "Unknown"
    thresholds = THRESHOLDS[key]
    if value < thresholds["low"]:
        return "Low"
    elif value > thresholds["high"]:
        return "High"
    return "Moderate"

def predict_medicare_spending(state_name):
    """
    Given a state name, this function loads the original processed dataset,
    extracts the row for that state, scales and transforms it using the same
    scaler and PCA used for training, then predicts via the trained model.
    Finally, it inverses both transformations to return predictions in the original scale.
    """
    # Load original data (for column names and later inverse transformation)
    user_item_matrix = pd.read_csv('state_level_insights.csv', index_col=0)
    
    # Debugging log: Check if the state exists in the dataset
    if state_name not in user_item_matrix.index:
        print(f"State '{state_name}' not found in the dataset.")
        return pd.DataFrame()  # Return an empty DataFrame if the state is not found

    # Select the row corresponding to the given state
    numeric_aggregated_state = user_item_matrix.loc[[state_name]]
    print(f"State data for '{state_name}':\n{numeric_aggregated_state}")  # Debugging log

    # Scale using the saved scaler
    sample_scaled = SCALER.transform(numeric_aggregated_state)
    
    # Transform with the saved PCA
    sample_pca = PCA_MODEL.transform(sample_scaled)
    
    sample_input = torch.tensor(sample_pca, dtype=torch.float32)
    
    # Set the model to evaluation mode and get prediction in PCA space
    FINAL_RBM.eval()
    with torch.no_grad():
        predicted_pca = FINAL_RBM(sample_input).squeeze(0)
    predicted_pca_np = predicted_pca.detach().numpy().reshape(1, -1)
    
    # Inverse transform: first from PCA space back to scaled space...
    reconstructed_scaled = PCA_MODEL.inverse_transform(predicted_pca_np)
    
    # ... then from scaled space back to the original feature space.
    reconstructed_output = SCALER.inverse_transform(reconstructed_scaled)

    friendly_names = {
        'TOT_MDCR_STDZD_PYMT_PC': "Standardized Medicare Payment per Capita",
        'TOT_MDCR_PYMT_PC': "Actual Medicare Payment per Capita",
        'BENE_AVG_RISK_SCRE': "Average Health Risk Score",
        'IP_CVRD_STAYS_PER_1000_BENES': "Inpatient Stay Rate (per 1,000 beneficiaries)",
        'ER_VISITS_PER_1000_BENES': "Emergency Department Visit Rate (per 1,000 beneficiaries)",
        'MA_PRTCPTN_RATE': "Medicare Advantage Participation Rate",
        'BENE_DUAL_PCT': "Medicaid Eligibility Percentage",
        'BENE_FEML_PCT': "Percent Female",
        'BENE_MALE_PCT': "Percent Male",
        'BENE_RACE_WHT_PCT': "Percent Non-Hispanic White",
        'BENE_RACE_BLACK_PCT': "Percent African American",
        'BENE_RACE_HSPNC_PCT': "Percent Hispanic"
    }

    # Convert to DataFrame with original column names
    predicted_df = pd.DataFrame(reconstructed_output, columns=user_item_matrix.columns)
    predicted_df = predicted_df.rename(columns=friendly_names)

    # Debugging log: Check the predicted DataFrame
    print(f"Predicted DataFrame for '{state_name}':\n{predicted_df}")  # Debugging log

    # Add spending level classifications
    for key, friendly_name in friendly_names.items():
        if key in predicted_df.columns:
            predicted_df[f"{friendly_name} Level"] = predicted_df[friendly_name].apply(
                lambda x: classify_spending(x, key)
            )
    return predicted_df

# Load the pre-trained transformer model for embeddings
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using a pre-trained transformer model.
    """
    embeddings = EMBEDDING_MODEL.encode(texts, convert_to_tensor=True)
    return embeddings.cpu().numpy()  # Move to CPU and convert to NumPy

def content_based_filtering(user_input, plans, item_scores=None):
    """
    Perform content-based filtering by calculating similarity between user inputs and plan descriptions.

    Parameters:
        user_input (dict): User inputs containing preferences.
        plans (list): List of plans with descriptions.
        item_scores (dict): Dictionary of collaborative filtering scores for each item.

    Returns:
        list: Ranked plans with similarity scores.
    """
    # Combine user inputs into a single string
    user_text = " ".join([f"{key}: {value}" for key, value in user_input.items() if value])

    # Generate embeddings for user input and plan descriptions
    user_embedding = generate_embeddings([user_text])
    plan_descriptions = [plan['description'] for plan in plans]
    plan_embeddings = generate_embeddings(plan_descriptions)

    # Compute cosine similarity between user input and plan descriptions
    similarities = cosine_similarity(user_embedding, plan_embeddings).flatten()

    # Add similarity scores to plans
    for i, plan in enumerate(plans):
        plan['similarity_score'] = similarities[i]

        # Combine collaborative filtering and content-based filtering scores
        item_id = plan["id"]
        if item_scores and item_id in item_scores:
            cf_score = item_scores[item_id]
            cb_score = plan['similarity_score']
            # Use a weighted average of the two scores
            plan['combined_score'] = 0.7 * cf_score + 0.3 * cb_score  # Adjust weights as needed
        else:
            plan['combined_score'] = plan['similarity_score']  # Use only content-based score if no CF score

    # Apply hard constraints to filter out irrelevant plans
    def violates_constraints(plan, user_input):
        """
        Check if a plan violates hard constraints based on user inputs.
        """
        # Exclude smoker plans for non-smokers
        if user_input.get("smoker") == "no" and "smoker" in plan["description"].lower():
            return True

        # Exclude catastrophic plans for users not in the 18-29 age group
        if user_input.get("age") != "young_adult" and "catastrophic" in plan["description"].lower():
            return True

        # Exclude chronic condition plans for users without chronic conditions
        if user_input.get("chronic_condition") == "no" and "chronic" in plan["description"].lower():
            return True

        # Exclude demographic-based plans for users not in the specified demographic
        ethnicity = user_input.get("ethnicity", "").lower()
        if ethnicity != "black" and "african american" in plan["description"].lower():
            return True
        if ethnicity != "hispanic" and "hispanic" in plan["description"].lower():
            return True
        if user_input.get("gender") != "female" and "women" in plan["description"].lower():
            return True

        return False

    # Filter out plans that violate constraints
    filtered_plans = [plan for plan in plans if not violates_constraints(plan, user_input)]

    # Sort plans by combined score in descending order
    ranked_plans = sorted(filtered_plans, key=lambda x: x['combined_score'], reverse=True)
    return ranked_plans