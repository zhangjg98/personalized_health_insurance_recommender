from flask import Flask, render_template, request
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending

app = Flask(__name__)

# Mapping of technical names to friendly names (only for key variables)
friendly_names = {
    'TOT_MDCR_STDZD_PYMT_PC': "Standardized Medicare Payment per Capita",
    'BENE_AVG_RISK_SCRE': "Average Health Risk Score",
    'ER_VISITS_PER_1000_BENES': "Emergency Department Visit Rate (per 1,000 beneficiaries)"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form
    rule_recommendation = recommend_plan(user_input)
    state = user_input.get('state', '').strip()
    
    ml_output_text = ""
    ml_summary = ""
    
    if state:
        try:
            # Get ML predictions
            ml_prediction_df = predict_medicare_spending(state)
            # Rename columns
            ml_prediction_df = ml_prediction_df.rename(columns=friendly_names)
            
            # Extract key metrics
            spending = ml_prediction_df["Standardized Medicare Payment per Capita"].iloc[0]
            risk = ml_prediction_df["Average Health Risk Score"].iloc[0]
            er_rate = ml_prediction_df["Emergency Department Visit Rate (per 1,000 beneficiaries)"].iloc[0]
            
            # Build a summary based on thresholds (adjust thresholds as needed)
            if spending > 50000:
                spending_text = "high spending"
                rule_recommendation["plan"] += " (Given high spending, consider comprehensive coverage.)"
            else:
                spending_text = "moderate spending"
                rule_recommendation["plan"] += " (Spending levels appear moderate.)"
            
            if risk > 1.0:
                risk_text = "a higher-than-average risk profile"
            else:
                risk_text = "an average risk profile"
            
            if er_rate > 800:  # example threshold, adjust accordingly
                er_text = "a high rate of emergency visits"
            else:
                er_text = "a moderate rate of emergency visits"
            
            ml_summary = (
                f"State-level analysis indicates {spending_text}, with {risk_text} and {er_text}. "
                "These factors suggest that you should consider plans that offer a balance of cost efficiency "
                "and comprehensive benefits."
            )
            
            # Optionally, if you want to show a table, limit to key metrics:
            key_metrics_df = ml_prediction_df[["Standardized Medicare Payment per Capita",
                                               "Average Health Risk Score",
                                               "Emergency Department Visit Rate (per 1,000 beneficiaries)"]]
            ml_output_text = key_metrics_df.to_html(classes='table table-striped')
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
