import pandas as pd
from ml_model import predict_medicare_spending
from thresholds import unified_thresholds

# Define the mapping from technical to friendly names
friendly_names = {
    "TOT_MDCR_STDZD_PYMT_PC": "Standardized Medicare Payment per Capita",
    "TOT_MDCR_PYMT_PC": "Actual Medicare Payment per Capita",
    "BENE_AVG_RISK_SCRE": "Average Health Risk Score",
    "IP_CVRD_STAYS_PER_1000_BENES": "Inpatient Stay Rate (per 1,000 beneficiaries)",
    "ER_VISITS_PER_1000_BENES": "Emergency Department Visit Rate (per 1,000 beneficiaries)",
    "MA_PRTCPTN_RATE": "Medicare Advantage Participation Rate",
    "BENE_DUAL_PCT": "Medicaid Eligibility Percentage"
}

# Load thresholds dynamically
thresholds = unified_thresholds(
    "processed_user_item_matrix.csv",
    keys=list(friendly_names.keys())
)

def classify_value(value, key):
    """
    Classify a value as 'Low', 'Moderate', or 'High' based on thresholds.
    """
    if key not in thresholds:
        return "Unknown"
    low, high = thresholds[key]["low"], thresholds[key]["high"]
    mid = (low + high) / 2
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    elif abs(value - mid) <= (high - low) * 0.25:
        return "Moderate"
    return "Low" if value < mid else "High"

# Load the processed user-item matrix to get the list of states
csv_path = "processed_user_item_matrix.csv"
df = pd.read_csv(csv_path, index_col=0)
states = df.index.tolist()

# Load the "National" row for comparison
national_data = pd.read_csv(csv_path, index_col=0).loc["National"]

# Classify each state using the trained data
classification_results = []
for state in states:
    # Use the trained model to predict Medicare spending for the state
    predicted_df = predict_medicare_spending(state)
    if predicted_df.empty:
        continue

    state_classification = {"State": state}

    # Classify each key using the trained data
    for key, friendly_name in friendly_names.items():
        if friendly_name in predicted_df.columns:
            value = predicted_df[friendly_name].iloc[0]
            classification = classify_value(value, key)
            state_classification[friendly_name] = classification

    # Add comparisons to national averages
    for key, friendly_name in friendly_names.items():
        if friendly_name in predicted_df.columns:
            state_value = predicted_df[friendly_name].iloc[0]
            national_value = national_data[key]
            if state_value > national_value:
                comparison = "above the national average"
            elif state_value < national_value:
                comparison = "below the national average"
            else:
                comparison = "on par with the national average"
            state_classification[f"{friendly_name} Comparison"] = comparison

    classification_results.append(state_classification)

# Convert results to a DataFrame
classification_df = pd.DataFrame(classification_results)

# Save the classification results to a CSV file for review
classification_df.to_csv("state_classifications_trained.csv", index=False)

# Generate a summary of states with multiple "High" or "Low" classifications
summary = []
for _, row in classification_df.iterrows():
    high_count = sum(1 for value in row[1:] if value == "High")
    low_count = sum(1 for value in row[1:] if value == "Low")
    if high_count > 1 or low_count > 1:
        summary.append({
            "State": row["State"],
            "High Count": high_count,
            "Low Count": low_count
        })

# Convert the summary to a DataFrame
summary_df = pd.DataFrame(summary)

# Check if the summary DataFrame is not empty before adding the "Outlier" column
if not summary_df.empty:
    summary_df["Outlier"] = summary_df.apply(
        lambda row: "Yes" if row["High Count"] > 1 or row["Low Count"] > 1 else "No", axis=1
    )

# Save the updated summary to a CSV file if it is not empty
if not summary_df.empty:
    summary_df.to_csv("state_summary_trained.csv", index=False)
