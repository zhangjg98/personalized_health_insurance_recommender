import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Load the CSV into a DataFrame.
# Replace 'your_file.csv' with the actual filename.
df = pd.read_csv('2014-2022 Medicare FFS Geographic Variation Public Use File.csv')

# Step 2: Define the columns of interest.
columns_to_keep = [
    'YEAR',                      # Year to track trends over time.
    'BENE_GEO_LVL',              # Geographic level (National, State, County).
    'BENE_GEO_DESC',             # Name of State/County.
    'BENE_GEO_CD',               # State/County FIPS code.
    'BENE_AGE_LVL',              # Age Level: All, <65, or 65+.
    'BENES_TOTAL_CNT',           # Total Medicare beneficiaries.
    'BENES_FFS_CNT',             # Fee-for-Service beneficiaries.
    'MA_PRTCPTN_RATE',           # Medicare Advantage Participation Rate.
    'BENE_FEML_PCT',             # Percent Female.
    'BENE_MALE_PCT',             # Percent Male.
    'BENE_RACE_WHT_PCT',         # Percent Non-Hispanic White.
    'BENE_RACE_BLACK_PCT',       # Percent African American.
    'BENE_RACE_HSPNC_PCT',       # Percent Hispanic.
    'BENE_DUAL_PCT',             # Percent Eligible for Medicaid.
    'TOT_MDCR_PYMT_AMT',         # Total Actual Medicare Payment.
    'TOT_MDCR_STDZD_PYMT_AMT',   # Total Standardized Medicare Payment.
    'TOT_MDCR_PYMT_PC',          # Actual Per Capita Medicare Payment.
    'TOT_MDCR_STDZD_PYMT_PC',    # Standardized Per Capita Medicare Payment.
    'IP_CVRD_STAYS_PER_1000_BENES',  # Inpatient Covered Stays per 1,000 Beneficiaries.
    'BENES_ER_VISITS_CNT',       # Total count of Emergency Department Visits.
    'ER_VISITS_PER_1000_BENES',    # Emergency Department Visits per 1,000 Beneficiaries.
    'BENE_AVG_RISK_SCRE'         # Average HCC Score.
]

# Step 3: Filter the DataFrame to keep only these columns.
df = df[columns_to_keep]

# Step 4: Explore the data
print("First few rows:")
print(df.head())

print("\nData summary:")
print(df.info())

# Step 5: Convert columns to appropriate data types.
# For example, convert numeric columns that might be read as strings.
numeric_columns = [
    'YEAR', 'BENES_TOTAL_CNT', 'BENES_FFS_CNT', 'MA_PRTCPTN_RATE',
    'BENE_FEML_PCT', 'BENE_MALE_PCT', 'BENE_RACE_WHT_PCT', 'BENE_RACE_BLACK_PCT',
    'BENE_RACE_HSPNC_PCT', 'BENE_DUAL_PCT', 'TOT_MDCR_PYMT_AMT',
    'TOT_MDCR_STDZD_PYMT_AMT', 'TOT_MDCR_PYMT_PC', 'TOT_MDCR_STDZD_PYMT_PC',
    'IP_CVRD_STAYS_PER_1000_BENES', 'BENES_ER_VISITS_CNT', 'ER_VISITS_PER_1000_BENES',
    'BENE_AVG_RISK_SCRE'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Step 6: Handle missing values.
# Example: Fill missing numeric values with the column's mean.
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Step 7: (Optional) Normalize key numeric features.
scaler = MinMaxScaler()
df['normalized_per_capita_payment'] = scaler.fit_transform(df[['TOT_MDCR_STDZD_PYMT_PC']])

# Step 8: Aggregate data by a key dimension (e.g., state-level analysis)
# For example, compute average standardized per capita payment by state.
state_summary = df[df['BENE_GEO_LVL'] == 'State'].groupby('BENE_GEO_DESC')['TOT_MDCR_STDZD_PYMT_PC'].mean().reset_index()

print("\nState-level Summary (Average Standardized Per Capita Payment):")
print(state_summary.head())

# You can now export the preprocessed data if needed:
df.to_csv('preprocessed_medicare_data.csv', index=False)
