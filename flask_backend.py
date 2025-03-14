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
    'BENE_AVG_RISK_SCRE': "Average Health Risk Score",
    'ER_VISITS_PER_1000_BENES': "Emergency Department Visit Rate (per 1,000 beneficiaries)"
}

# Compute dynamic thresholds from the processed CSV
keys_for_thresholds = ["TOT_MDCR_STDZD_PYMT_PC", "BENE_AVG_RISK_SCRE", "ER_VISITS_PER_1000_BENES"]
dynamic_thresholds = unified_thresholds("processed_user_item_matrix.csv", keys_for_thresholds)

# Load the summary of states with multiple "High" or "Low" classifications
state_summary = pd.read_csv("state_summary.csv")

@app.route('/')
def home():
    return render_template('index.html')

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

        # Generate ML predictions
        ml_prediction_df = predict_medicare_spending(state)
        print("ML Prediction DataFrame:\n", ml_prediction_df)  # Debugging log
        ml_prediction_df = ml_prediction_df.rename(columns=friendly_names)
        
        # Extract key metrics
        spending = ml_prediction_df["Standardized Medicare Payment per Capita"].iloc[0]
        risk = ml_prediction_df["Average Health Risk Score"].iloc[0]
        er_rate = ml_prediction_df["Emergency Department Visit Rate (per 1,000 beneficiaries)"].iloc[0]
        
        # Use dynamic thresholds (computed earlier)
        spending_thresholds = dynamic_thresholds["TOT_MDCR_STDZD_PYMT_PC"]
        risk_thresholds = dynamic_thresholds["BENE_AVG_RISK_SCRE"]
        er_thresholds = dynamic_thresholds["ER_VISITS_PER_1000_BENES"]
        
        # Classify metrics
        spending_classification = (
            "High" if spending > spending_thresholds["high"] else
            "Low" if spending < spending_thresholds["low"] else
            "Moderate"
        )
        risk_classification = (
            "High" if risk > risk_thresholds["high"] else
            "Low" if risk < risk_thresholds["low"] else
            "Moderate"
        )
        er_rate_classification = (
            "High" if er_rate > er_thresholds["high"] else
            "Low" if er_rate < er_thresholds["low"] else
            "Moderate"
        )
        
        # Build summary messages
        if spending_classification == "High":
            spending_text = "high spending"
            rule_recommendation["plan"] += " (Given high spending, consider comprehensive coverage.)"
        elif spending_classification == "Low":
            spending_text = "low spending"
            rule_recommendation["plan"] += " (Given low spending, consider plans with lower premiums.)"
        else:
            spending_text = "moderate spending"
            rule_recommendation["plan"] += " (Spending levels appear moderate.)"
        
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
        
        # Check for outlier information
        outlier_row = state_summary[state_summary["State"] == state]
        if not outlier_row.empty:
            high_count = outlier_row["High Count"].iloc[0]
            low_count = outlier_row["Low Count"].iloc[0]
            outlier_message = (
                f"Note: {state} has {high_count} 'High' classifications and {low_count} 'Low' classifications."
            )
        
        # Convert predictions to JSON
        key_metrics_df = ml_prediction_df[[
            "Standardized Medicare Payment per Capita",
            "Average Health Risk Score",
            "Emergency Department Visit Rate (per 1,000 beneficiaries)"
        ]]
        ml_output_json = key_metrics_df.to_dict(orient="records")

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
