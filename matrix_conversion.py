import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

# Load the preprocessed dataset
df = pd.read_csv('preprocessed_medicare_data.csv')

# Selecting relevant columns for the user-item matrix (exclude 'YEAR')
features = [
    'BENE_GEO_DESC',             # State Name (User Index)
    'TOT_MDCR_STDZD_PYMT_PC',      # Standardized Per Capita Medicare Payment
    'TOT_MDCR_PYMT_PC',           # Actual Per Capita Medicare Payment
    'BENE_AVG_RISK_SCRE',         # Average HCC Score
    'IP_CVRD_STAYS_PER_1000_BENES',# Inpatient Covered Stays per 1,000 Beneficiaries
    'ER_VISITS_PER_1000_BENES',     # Emergency Department Visits per 1,000 Beneficiaries
    'MA_PRTCPTN_RATE',            # Medicare Advantage Participation Rate
    'BENE_DUAL_PCT',              # Percent Eligible for Medicaid
    'BENES_TOTAL_CNT',            # Total Medicare Beneficiaries
    'BENES_FFS_CNT',              # Fee-for-Service Beneficiaries
    'BENE_FEML_PCT',              # Percent Female
    'BENE_MALE_PCT',              # Percent Male
    'BENE_RACE_WHT_PCT',          # Percent Non-Hispanic White
    'BENE_RACE_BLACK_PCT',        # Percent African American
    'BENE_RACE_HSPNC_PCT'         # Percent Hispanic
]

df = df[features]

# Aggregate by state (or state-year if you want temporal info, but then ensure consistency)
df = df.groupby('BENE_GEO_DESC', as_index=False).mean()

# Normalize numeric values for ML input
scaler = MinMaxScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])

# Convert DataFrame into a user-item matrix (State as index)
user_item_matrix = df.set_index('BENE_GEO_DESC')

# Save the transformed matrix
user_item_matrix.to_csv('processed_user_item_matrix.csv')

# Convert to PyTorch tensor (for training)
user_item_tensor = torch.tensor(user_item_matrix.values, dtype=torch.float32)
print(f"User-Item Matrix Shape: {user_item_tensor.shape}")
