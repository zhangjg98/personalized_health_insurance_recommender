from flask import Flask, render_template, request
from insurance_recommender import recommend_plan
from ml_model import predict_medicare_spending
from models import DeepAutoencoder

app = Flask(__name__)

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
    
    # Initialize a variable for ML output text
    ml_output_text = ""
    
    if state:
        try:
            ml_prediction_df = predict_medicare_spending(state)
            # For example, get the predicted per capita spending
            spending = ml_prediction_df['TOT_MDCR_STDZD_PYMT_PC'].iloc[0]
            # Define a threshold (this is arbitrary; will adjust later)
            spending_threshold = 50000
            
            # Adjust the rule-based recommendation if spending is high
            if spending > spending_threshold:
                rule_recommendation["plan"] += " (Given high state-level spending, consider plans with comprehensive coverage.)"
            else:
                rule_recommendation["plan"] += " (State-level data suggests moderate spending levels.)"
            
            ml_output_text = ml_prediction_df.to_html(classes='table table-striped')
        except Exception as e:
            ml_output_text = f"Error in generating ML prediction: {str(e)}"
    else:
        ml_output_text = "No state was selected; state-level insights are unavailable."
    
    # Render the results page with both recommendations
    return render_template('result.html', 
                           recommendation=rule_recommendation, 
                           ml_prediction=ml_output_text)

if __name__ == '__main__':
    app.run(debug=True)
