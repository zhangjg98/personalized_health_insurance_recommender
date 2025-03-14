import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # using PCA instead of SVD
import pickle
from models import DeepAutoencoder
from thresholds import unified_thresholds

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
    "processed_user_item_matrix.csv",
    keys=["TOT_MDCR_STDZD_PYMT_PC", "TOT_MDCR_PYMT_PC", "BENE_AVG_RISK_SCRE"]
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
    user_item_matrix = pd.read_csv('processed_user_item_matrix.csv', index_col=0)
    
    # Select the row corresponding to the given state. Since state is now the index,
    # this returns a DataFrame.
    numeric_aggregated_state = user_item_matrix.loc[[state_name]]
    
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
        'BENES_TOTAL_CNT': "Total Beneficiaries",
        'BENES_FFS_CNT': "Fee-for-Service Beneficiaries",
        'BENE_FEML_PCT': "Percent Female",
        'BENE_MALE_PCT': "Percent Male",
        'BENE_RACE_WHT_PCT': "Percent Non-Hispanic White",
        'BENE_RACE_BLACK_PCT': "Percent African American",
        'BENE_RACE_HSPNC_PCT': "Percent Hispanic"
    }

    # Convert to DataFrame with original column names
    predicted_df = pd.DataFrame(reconstructed_output, columns=user_item_matrix.columns)
    predicted_df = predicted_df.rename(columns=friendly_names)

    # Add spending level classifications
    for key, friendly_name in friendly_names.items():
        if key in predicted_df.columns:
            predicted_df[f"{friendly_name} Level"] = predicted_df[friendly_name].apply(
                lambda x: classify_spending(x, key)
            )
    return predicted_df
