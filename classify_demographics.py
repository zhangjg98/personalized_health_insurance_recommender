import pandas as pd
from ml_model import predict_medicare_spending
from thresholds import unified_thresholds

def classify_demographics(states, output_file="demographic_classifications_trained.csv"):
    """
    Classify states based on demographic thresholds using trained data.

    Parameters:
        states (list): List of state names to evaluate.
        output_file (str): Path to save the classification results.
    """
    # Define demographic keys and their friendly names
    demographic_keys = {
        "BENE_FEML_PCT": "Percent Female",
        "BENE_MALE_PCT": "Percent Male",
        "BENE_RACE_WHT_PCT": "Percent Non-Hispanic White",
        "BENE_RACE_BLACK_PCT": "Percent African American",
        "BENE_RACE_HSPNC_PCT": "Percent Hispanic"
    }

    # Load thresholds for demographic variables with tighter quantiles
    thresholds = unified_thresholds(
        "processed_user_item_matrix.csv",
        demographic_keys.keys(),
        lower_quantile=0.2,  # Narrow the lower quantile
        upper_quantile=0.8,  # Narrow the upper quantile
        scale_factor=1.1     # Reduce the scale factor for tighter thresholds
    )

    # Initialize a list to store classification results
    classification_results = []

    # Classify each state using the trained model
    for state in states:
        # Predict demographic values for the state
        predicted_df = predict_medicare_spending(state)
        if predicted_df.empty:
            continue

        state_data = {"State": state}
        for key, friendly_name in demographic_keys.items():
            if friendly_name in predicted_df.columns:
                value = predicted_df[friendly_name].iloc[0]
                low, high = thresholds[key]["low"], thresholds[key]["high"]
                if value < low:
                    classification = "Below Threshold"
                elif value > high:
                    classification = "Above Threshold"
                else:
                    classification = "Within Threshold"
                state_data[friendly_name] = classification  # Use friendly name for clarity
        classification_results.append(state_data)

    # Convert results to a DataFrame
    classification_df = pd.DataFrame(classification_results)

    # Save the classification results to a CSV file
    classification_df.to_csv(output_file, index=False)
    print(f"Demographic classifications saved to {output_file}")

if __name__ == "__main__":
    # Load the processed dataset to get the list of states
    csv_path = "processed_user_item_matrix.csv"
    df = pd.read_csv(csv_path, index_col=0)
    states = df.index.tolist()

    # Classify states based on demographic thresholds using trained data
    classify_demographics(states)
