import pandas as pd
from thresholds import unified_thresholds

# Load the processed user-item matrix
csv_path = "processed_user_item_matrix.csv"
df = pd.read_csv(csv_path, index_col=0)

# Define the keys for classification
keys = [
    "TOT_MDCR_STDZD_PYMT_PC", 
    "TOT_MDCR_PYMT_PC",
    "BENE_AVG_RISK_SCRE", 
    "IP_CVRD_STAYS_PER_1000_BENES",
    "ER_VISITS_PER_1000_BENES",
    "MA_PRTCPTN_RATE",
    'BENE_DUAL_PCT'
]

# Compute thresholds dynamically using the same function as the backend
thresholds = unified_thresholds(csv_path, keys)

def classify_value(value, key):
    """
    Classify a value as 'Low', 'Moderate', or 'High' based on thresholds.
    """
    if key not in thresholds:
        return "Unknown"
    low, high = thresholds[key]["low"], thresholds[key]["high"]
    mid = (low + high) / 2  # Calculate the midpoint

    # Debugging logs
    print(f"Classifying value: {value} for key: {key} with thresholds: low={low}, high={high}, mid={mid}")

    if value < low:
        return "Low"
    elif value > high:
        return "High"
    elif abs(value - mid) <= (high - low) * 0.25:  # Stricter "Moderate" range
        return "Moderate"
    return "Low" if value < mid else "High"

# Classify each state for the selected keys
classification_results = []
for state, row in df.iterrows():
    state_classification = {"State": state}
    for key in keys:
        if key in row:
            value = row[key]
            classification = classify_value(value, key)
            print(f"Processing {state} - {key}: Value={value}, Classification={classification}")  # Debugging log
            state_classification[key] = classification
    classification_results.append(state_classification)

# Convert results to a DataFrame
classification_df = pd.DataFrame(classification_results)

# Save the classification results to a CSV file for review
classification_df.to_csv("state_classifications.csv", index=False)

# Print the classification results for quick reference
print(classification_df)

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

# Convert the summary to a DataFrame and save it
summary_df = pd.DataFrame(summary)

# Add an "Outlier" column to the summary
summary_df["Outlier"] = summary_df.apply(
    lambda row: "Yes" if row["High Count"] > 1 or row["Low Count"] > 1 else "No", axis=1
)

# Save the updated summary to a CSV file
summary_df.to_csv("state_summary.csv", index=False)

# Print the updated summary for quick reference
print("\nUpdated Summary of States with Multiple 'High' or 'Low' Classifications:")
print(summary_df)
