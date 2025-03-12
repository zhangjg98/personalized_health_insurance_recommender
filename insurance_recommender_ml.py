from flask import Flask, request, jsonify
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

# Define friendly names mapping
friendly_names = {
        'TOT_MDCR_STDZD_PYMT_PC': "Standardized Medicare Payment per Capita",
        'TOT_MDCR_PYMT_PC': "Actual Medicare Payment per Capita",
        'BENE_AVG_RISK_SCRE': "Average Health Risk Score",
        'IP_CVRD_STAYS_PER_1000_BENES': "Inpatient Stay Rate (per 1,000 beneficiaries)",
        'ER_VISITS_PER_1000_BENES': "Emergency Department Visit Rate (per 1,000 beneficiaries)",
        'MA_PRTCPTN_RATE': "Medicare Advantage Participation Rate",
        'BENE_DUAL_PCT': "Medicaid Eligibility Percentage",
        'BENES_TOTAL_CNT': "Total Beneficiaries",
        'BENES_FFS_CNT': "Fee-for-Service Beneficiaries",
        'BENE_FEML_PCT': "Percent Female",
        'BENE_MALE_PCT': "Percent Male",
        'BENE_RACE_WHT_PCT': "Percent Non-Hispanic White",
        'BENE_RACE_BLACK_PCT': "Percent African American",
        'BENE_RACE_HSPNC_PCT': "Percent Hispanic"
    }

@app.route('/api/recommend', methods=['POST'])
def recommend():
    # Expect JSON input from React
    data = request.get_json()
    
    # Assume data is a dict with keys like age, smoker, state, etc.
    rule_recommendation = recommend_plan(data)
    state = data.get('state', '').strip()
    
    ml_summary = ""
    ml_data = {}
    
    if state:
        try:
            ml_prediction_df = predict_medicare_spending(state)
            # Rename columns to friendly names for a user‑friendly response
            ml_prediction_df = ml_prediction_df.rename(columns=friendly_names)
            
            # Extract key metrics (adjust as needed)
            spending = ml_prediction_df["Standardized Medicare Payment per Capita"].iloc[0]
            risk_score = ml_prediction_df["Average Health Risk Score"].iloc[0]
            spending_threshold = 50000
            if spending > spending_threshold:
                rule_recommendation["plan"] += " (Given high state-level spending, consider comprehensive coverage.)"
                spending_comment = "high spending levels"
            else:
                rule_recommendation["plan"] += " (State-level spending appears moderate.)"
                spending_comment = "moderate spending levels"
            risk_comment = "a higher-than-average risk profile" if risk_score > 1.0 else "an average risk profile"
            
            ml_summary = (f"The state shows {spending_comment} with a risk profile of {risk_score:.2f}.")
            # Convert DataFrame to dictionary for JSON response
            ml_data = ml_prediction_df.to_dict(orient='records')[0]
        except Exception as e:
            ml_summary = f"Error in ML prediction: {str(e)}"
    else:
        ml_summary = "No state was selected; state-level insights are unavailable."
    
    # Build the final response as JSON
    response = {
        "rule_recommendation": rule_recommendation,
        "ml_summary": ml_summary,
        "ml_data": ml_data
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
