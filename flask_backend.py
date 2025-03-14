from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending
from thresholds import compute_dynamic_thresholds, unified_thresholds
import pandas as pd

app = Flask(__name__)
CORS(app)

# Define a mapping from technical to friendly names (for key metrics)
friendly_names = {
    'TOT_MDCR_STDZD_PYMT_PC': "Standardized Medicare Payment per Capita",
    'TOT_MDCR_PYMT_PC': "Actual Medicare Payment per Capita",
    'BENE_AVG_RISK_SCRE': "Average Health Risk Score",
    'IP_CVRD_STAYS_PER_1000_BENES': "Inpatient Stay Rate (per 1,000 beneficiaries)", 
    'ER_VISITS_PER_1000_BENES': "Emergency Department Visit Rate (per 1,000 beneficiaries)",
    'MA_PRTCPTN_RATE': "Medicare Advantage Participation Rate",
    'BENE_DUAL_PCT': "Medicaid Eligibility Percentage",
}

# Compute dynamic thresholds from the processed CSV
keys_for_thresholds = [
    "TOT_MDCR_STDZD_PYMT_PC", 
    "TOT_MDCR_PYMT_PC",
    "BENE_AVG_RISK_SCRE", 
    "IP_CVRD_STAYS_PER_1000_BENES",
    "ER_VISITS_PER_1000_BENES",
    "MA_PRTCPTN_RATE",
    'BENE_DUAL_PCT'
]
dynamic_thresholds = unified_thresholds("processed_user_item_matrix.csv", keys_for_thresholds)

# Load the summary of states with multiple "High" or "Low" classifications
state_summary = pd.read_csv("state_summary.csv")

@app.route('/')
def home():
    return render_template('index.html')

def classify_value(value, thresholds):
    """
    Classify a value as 'Low', 'Moderate', or 'High' based on thresholds.
    """
    low, high = thresholds["low"], thresholds["high"]
    mid = (low + high) / 2  # Calculate the midpoint
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    elif abs(value - mid) <= (high - low) * 0.25:  # Stricter "Moderate" range
        return "Moderate"
    return "Low" if value < mid else "High"

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json  # Use JSON input instead of form data
        print("Received user input:", user_input)  # Debugging log

        rule_recommendation = recommend_plan(user_input)
        state = user_input.get('state', '').strip()
        
        # Initialize variables with default values
        ml_output_json = []  # Default to an empty list
        ml_summary = ""
        outlier_message = ""

        # Validate input
        if not state:
            return jsonify({
                "recommendation": rule_recommendation,
                "ml_prediction": ml_output_json,
                "ml_summary": "No state provided. Unable to generate state-level analysis.",
                "outlier_message": "",
            })

        # Load raw values from processed_user_item_matrix.csv
        raw_data = pd.read_csv("processed_user_item_matrix.csv", index_col=0)
        if state not in raw_data.index:
            return jsonify({
                "recommendation": rule_recommendation,
                "ml_prediction": ml_output_json,
                "ml_summary": f"No data available for state: {state}.",
                "outlier_message": "",
            })

        state_data = raw_data.loc[state]
        print(f"Raw data for {state}:\n{state_data}")  # Debugging log

        # Classify raw values for outlier information
        classifications = {}
        for key, friendly_name in friendly_names.items():
            if key in dynamic_thresholds and key in state_data:
                value = state_data[key]
                classification = classify_value(value, dynamic_thresholds[key])
                classifications[friendly_name] = classification
                print(f"Processing {friendly_name}: Value={value}, Classification={classification}")  # Debugging log

        # Generate ML predictions
        ml_prediction_df = predict_medicare_spending(state)
        print("ML Prediction DataFrame:\n", ml_prediction_df)  # Debugging log
        ml_prediction_df = ml_prediction_df.rename(columns=friendly_names)

        # Convert ML predictions to JSON for frontend display
        ml_output_json = ml_prediction_df.to_dict(orient="records")

        # Build summary messages based on ML predictions
        spending = ml_prediction_df["Standardized Medicare Payment per Capita"].iloc[0]
        risk = ml_prediction_df["Average Health Risk Score"].iloc[0]
        er_rate = ml_prediction_df["Emergency Department Visit Rate (per 1,000 beneficiaries)"].iloc[0]

        spending_classification = classify_value(spending, dynamic_thresholds["TOT_MDCR_STDZD_PYMT_PC"])
        risk_classification = classify_value(risk, dynamic_thresholds["BENE_AVG_RISK_SCRE"])
        er_rate_classification = classify_value(er_rate, dynamic_thresholds["ER_VISITS_PER_1000_BENES"])

        if spending_classification == "High":
            spending_text = "high spending"
            rule_recommendation["plan"] += " (Given high state spending, consider comprehensive coverage.)"
        elif spending_classification == "Low":
            spending_text = "low spending"
            rule_recommendation["plan"] += " (Given low state spending, consider plans with lower premiums.)"
        else:
            spending_text = "moderate spending"
            rule_recommendation["plan"] += " (State spending levels appear moderate.)"
        
        if risk_classification == "High":
            risk_text = "a higher-than-average risk profile"
        elif risk_classification == "Low":
            risk_text = "a lower-than-average risk profile"
        else:
            risk_text = "an average risk profile"
        
        if er_rate_classification == "High":
            er_text = "a high rate of emergency visits"
        elif er_rate_classification == "Low":
            er_text = "a low rate of emergency visits"
        else:
            er_text = "a moderate rate of emergency visits"
        
        ml_summary = (
            f"State-level analysis indicates {spending_text}, with {risk_text} and {er_text}. "
            "These factors suggest that you should consider plans that balance cost and benefits accordingly."
        )
        
        # Generate outlier information based on raw data
        outlier_row = state_summary[state_summary["State"] == state]
        if not outlier_row.empty:
            high_count = outlier_row["High Count"].iloc[0]
            low_count = outlier_row["Low Count"].iloc[0]
            
            # Identify specific metrics classified as "High" or "Low"
            high_metrics = [k for k, v in classifications.items() if v == "High"]
            low_metrics = [k for k, v in classifications.items() if v == "Low"]

            # Construct a detailed outlier message
            outlier_message = f"Note: {state} has {high_count} 'High' classifications and {low_count} 'Low' classifications. "
            if high_metrics:
                outlier_message += f"Metrics classified as 'High': {', '.join(high_metrics)}. "
            if low_metrics:
                outlier_message += f"Metrics classified as 'Low': {', '.join(low_metrics)}."
            print(f"Outlier message for {state}: {outlier_message}")  # Debugging log

    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error during ML prediction: {e}")
        ml_output_json = [{"error": f"Error in generating ML prediction: {str(e)}"}]
        ml_summary = "An error occurred while generating state-level analysis."

    return jsonify({
        "recommendation": rule_recommendation,
        "ml_prediction": ml_output_json,
        "ml_summary": ml_summary,
        "outlier_message": outlier_message,
    })

if __name__ == '__main__':
    app.run(debug=True)
