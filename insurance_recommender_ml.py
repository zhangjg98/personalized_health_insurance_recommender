from flask import Flask, render_template, request
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending

app = Flask(__name__)

# Friendly name mapping for ML model outputs
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Retrieve user inputs from the form
    user_input = request.form
    # Get the rule-based recommendation
    rule_recommendation = recommend_plan(user_input)
    
    # Retrieve the selected state from the form (if provided)
    state = user_input.get('state', '').strip()
    # Retrieve additional demographic inputs (if you added these in your form)
    income_input = user_input.get('income', '')
    family_size_input = user_input.get('family_size', '')
    # ... add more if needed

    ml_output_text = ""
    ml_summary = ""
    
    if state:
        try:
            ml_prediction_df = predict_medicare_spending(state)
            ml_prediction_df = ml_prediction_df.rename(columns=friendly_names)
            
            # Extract key metrics
            spending = ml_prediction_df["Standardized Medicare Payment per Capita"].iloc[0]
            risk_score = ml_prediction_df["Average Health Risk Score"].iloc[0]
            
            # Define thresholds (adjust these based on domain knowledge)
            spending_threshold = 50000
            risk_threshold = 1.0
            
            # Build a plain language summary from ML outputs
            if spending > spending_threshold:
                spending_comment = "high spending levels"
                rule_recommendation["plan"] += " (Given high state-level spending, consider comprehensive coverage.)"
            else:
                spending_comment = "moderate spending levels"
                rule_recommendation["plan"] += " (State-level spending appears moderate.)"
            
            if risk_score > risk_threshold:
                risk_comment = "the beneficiary population is at higher risk"
            else:
                risk_comment = "the beneficiary population has average risk levels"
            
            ml_summary = (f"The state shows {spending_comment} with a risk score of {risk_score:.2f}. "
                          f"Based on your demographic inputs, you may want to consider options that "
                          f"balance cost and comprehensive coverage accordingly.")
            
            ml_output_text = ml_prediction_df.to_html(classes='table table-striped')
        except Exception as e:
            ml_output_text = f"Error in generating ML prediction: {str(e)}"
    else:
        ml_output_text = "No state was selected; state-level insights are unavailable."
    
    return render_template('result.html', 
                           recommendation=rule_recommendation, 
                           ml_prediction=ml_output_text,
                           ml_summary=ml_summary)

if __name__ == '__main__':
    app.run(debug=True)
