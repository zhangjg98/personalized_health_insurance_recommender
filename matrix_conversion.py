import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_medicare_data.csv')

# Selecting a richer set of relevant columns for the user-item matrix
features = [
    'BENE_GEO_DESC',              # State Name (User Index)
    'TOT_MDCR_STDZD_PYMT_PC',       # Standardized Per Capita Medicare Payment
    'TOT_MDCR_PYMT_PC',           # Actual Per Capita Medicare Payment
    'BENE_AVG_RISK_SCRE',         # Average HCC Score
    'IP_CVRD_STAYS_PER_1000_BENES', # Inpatient Covered Stays per 1,000 Beneficiaries
    'ER_VISITS_PER_1000_BENES',     # Emergency Department Visits per 1,000 Beneficiaries
    'MA_PRTCPTN_RATE',            # Medicare Advantage Participation Rate
    'BENE_DUAL_PCT',              # Percent Eligible for Medicaid
    'BENES_TOTAL_CNT',            # Total Medicare Beneficiaries
    'BENES_FFS_CNT'               # Fee-for-Service Beneficiaries
]

df = df[features]

# **Aggregate by state to get a single row per state**
df = df.groupby('BENE_GEO_DESC', as_index=False).mean()

# Normalize numeric values for ML input (all columns except the index)
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Convert DataFrame into a user-item matrix with state as the index
user_item_matrix = df.set_index('BENE_GEO_DESC')

# Save the transformed matrix for later use (e.g., in ML model training and prediction)
user_item_matrix.to_csv('processed_user_item_matrix.csv')

# Convert to PyTorch tensor (this tensor isn’t used for the final model, but useful for debugging/training)
user_item_tensor = torch.tensor(user_item_matrix.values, dtype=torch.float32)

print(f"User-Item Matrix Shape: {user_item_tensor.shape}")  # For example: (50 states, 9 features)
