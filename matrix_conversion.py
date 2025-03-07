import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_medicare_data.csv')

# Selecting relevant columns for the user-item matrix
features = [
    'BENE_GEO_DESC',  # State Name (User Index)
    'TOT_MDCR_STDZD_PYMT_PC', 'BENE_AVG_RISK_SCRE', 
    'IP_CVRD_STAYS_PER_1000_BENES', 'ER_VISITS_PER_1000_BENES', 
    'MA_PRTCPTN_RATE', 'BENE_DUAL_PCT'
]

df = df[features]

# **Aggregate by state to get a single row per state**
df = df.groupby('BENE_GEO_DESC', as_index=False).mean()

# Normalize numeric values for RBM input
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Convert DataFrame into a user-item matrix (State as index)
user_item_matrix = df.set_index('BENE_GEO_DESC')

# Save the transformed matrix
user_item_matrix.to_csv('processed_user_item_matrix.csv')

# Convert to PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix.values, dtype=torch.float32)

print(f"User-Item Matrix Shape: {user_item_tensor.shape}")  # Example: (50 states, 6 features)
