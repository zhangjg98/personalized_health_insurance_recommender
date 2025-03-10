import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import pickle
from models import HybridRBM_SVD

# Define hyperparameters used during training
n_components = 4
num_hidden_1 = 50
num_hidden_2 = 30
num_latent = 10

def load_trained_objects():
    # Create an instance of the model with the same architecture as used in training.
    # (Ensure you use the same hyperparameters here.)
    final_rbm = HybridRBM_SVD(num_visible=n_components,  # from your SVD settings
                              num_hidden_1=num_hidden_1,
                              num_hidden_2=num_hidden_2,
                              num_latent=num_latent)
    final_rbm.load_state_dict(torch.load('final_rbm.pth'))
    
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('svd.pkl', 'rb') as f:
        svd = pickle.load(f)
    return final_rbm, scaler, svd

# Load the objects once when the module is imported
FINAL_RBM, SCALER, SVD = load_trained_objects()

def predict_medicare_spending(state_name):
    """
    Given a state name, this function loads the original processed dataset,
    extracts the row for that state, scales and transforms it using the same
    scaler and SVD used for training, then predicts via the trained model.
    Finally, it inverses both transformations to return predictions in the original scale.
    """
    # Load original data (for column names and later inverse transformation)
    user_item_matrix = pd.read_csv('processed_user_item_matrix.csv', index_col=0)
    
    # Get the row for the given state
    numeric_aggregated_state = user_item_matrix.loc[state_name].to_frame().T
    
    # Scale using the saved scaler
    sample_scaled = SCALER.transform(numeric_aggregated_state)
    
    # Transform with the saved SVD
    sample_svd = SVD.transform(sample_scaled)
    sample_input = torch.tensor(sample_svd, dtype=torch.float32)
    
    # Set the model to evaluation mode and get prediction in SVD space
    FINAL_RBM.eval()
    with torch.no_grad():
        predicted_svd = FINAL_RBM(sample_input).squeeze(0)
    predicted_svd_np = predicted_svd.detach().numpy().reshape(1, -1)
    
    # Inverse transform: first from SVD space back to scaled space...
    reconstructed_scaled = SVD.inverse_transform(predicted_svd_np)
    # ... then from scaled space back to original feature space.
    reconstructed_output = SCALER.inverse_transform(reconstructed_scaled)
    
    # Convert to DataFrame with original column names
    predicted_df = pd.DataFrame(reconstructed_output, columns=user_item_matrix.columns)
    return predicted_df
