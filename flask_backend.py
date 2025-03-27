from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from database import db, User, Item, Interaction, UserItemMatrix  # Import models from database.py
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending
from thresholds import compute_dynamic_thresholds, unified_thresholds
from neural_collaborative_filtering import load_ncf_model, predict_user_item_interactions
import pandas as pd
import numpy as np
import hashlib
import json  # Ensure JSON encoding/decoding for user_inputs

app = Flask(__name__)
CORS(app)

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://recommender_user:securepassword@localhost/health_insurance_recommender'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)  # Initialize the database with the Flask app

def create_tables():
    with app.app_context():
        db.create_all()

create_tables()

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
    "BENE_DUAL_PCT"
]
dynamic_thresholds = unified_thresholds("processed_user_item_matrix.csv", keys_for_thresholds)

# Load the NeuralCollaborativeFiltering model and user-item matrix
NCF_MODEL, USER_ITEM_MATRIX = load_ncf_model()

@app.route('/')
def home():
    return render_template('index.html')

def classify_value(value, thresholds):
    """
    Classify a value as 'Low', 'Moderate', or 'High' based on thresholds.
    """
    low, high = thresholds["low"], thresholds["high"]
    mid = (low + high) / 2
    if value < low:
        return "Low"
    elif value > high:
        return "High"
    elif abs(value - mid) <= (high - low) * 0.25:
        return "Moderate"
    return "Low" if value < mid else "High"

def compute_composite_ml_score(spending, risk, er_rate, thresholds):
    """
    Compute a composite score using normalized distances from the threshold midpoints.
    The score is an average of the normalized distances (between 0 and 1).
    """
    scores = {}
    for key, value in zip(["TOT_MDCR_STDZD_PYMT_PC", "BENE_AVG_RISK_SCRE", "ER_VISITS_PER_1000_BENES"],
                           [spending, risk, er_rate]):
        thresh = thresholds[key]
        midpoint = (thresh["low"] + thresh["high"]) / 2.0
        # Compute absolute distance normalized by the range
        norm_distance = abs(value - midpoint) / (thresh["high"] - thresh["low"])
        # Clip between 0 and 1 for safety
        norm_distance = np.clip(norm_distance, 0, 1)
        scores[key] = norm_distance
    # Composite score is the average of these distances.
    composite_score = np.mean(list(scores.values()))
    return composite_score, scores

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json
        print("Received user input:", user_input)

        user_id = int(user_input.get("user_id", -1))
        if user_id == -1:
            return jsonify({"error": "Invalid user_id. Please ensure you are logged in or registered."}), 400

        priority = user_input.get("priority", "")  # Get a single priority
        state = user_input.get('state', '').strip()

        # Generate ML predictions and insights only if a state is provided
        ml_prediction_df = None
        if state:
            ml_prediction_df = predict_medicare_spending(state)
            print("ML Prediction DataFrame:\n", ml_prediction_df)

        # Pass ML predictions to the recommendation logic
        recommendations = recommend_plan(user_input, priority, ml_prediction_df)

        # Initialize variables with default values
        ml_output_json = []
        ml_summary = ""
        outlier_message = ""
        ncf_recommendations = []

        # Generate ML predictions and insights only if a state is provided
        if state:
            # Load the "National" row for comparison
            national_data = pd.read_csv("processed_user_item_matrix.csv", index_col=0).loc["National"]

            # Add classification labels to ML predictions
            for key, friendly_name in friendly_names.items():
                if friendly_name in ml_prediction_df.columns:
                    ml_prediction_df[f"{friendly_name} Level"] = ml_prediction_df[friendly_name].apply(
                        lambda x: classify_value(x, dynamic_thresholds[key])
                    )

            # Add comparisons to national averages
            comparisons = {}
            for key, friendly_name in friendly_names.items():
                if friendly_name in ml_prediction_df.columns:
                    state_value = ml_prediction_df[friendly_name].iloc[0]
                    national_value = national_data[key]
                    if state_value > national_value:
                        comparison = "It is above the national average."
                    elif state_value < national_value:
                        comparison = "It is below the national average."
                    else:
                        comparison = "It is on par with the national average."
                    comparisons[friendly_name] = comparison

            # Include comparisons in the response
            ml_output_json = ml_prediction_df.to_dict(orient="records")
            for metric, comparison in comparisons.items():
                ml_output_json[0][f"{metric} Comparison"] = comparison

            # Add clarification message with percentage thresholds
            clarification_message = (
                "The classification (e.g., 'Moderate') is based on thresholds derived from state-level data, "
                "specifically the 10th to 90th percentiles. Values within this range are classified as 'Moderate'. "
                "The comparison (e.g., 'above the national average') is relative to the national average, "
                "calculated as the mean of all state values."
            )

            # Add clarification message for predicted values
            prediction_context_message = (
                "The values shown in 'Predicted Medicare Spending Details' are predictions based on historical data "
                "and trained machine learning models. These predictions aim to provide insights into state-level trends "
                "and are not exact measurements."
            )

            # Build summary messages based on ML predictions
            spending = ml_prediction_df["Standardized Medicare Payment per Capita"].iloc[0]
            risk = ml_prediction_df["Average Health Risk Score"].iloc[0]
            er_rate = ml_prediction_df["Emergency Department Visit Rate (per 1,000 beneficiaries)"].iloc[0]

            spending_classification = classify_value(spending, dynamic_thresholds["TOT_MDCR_STDZD_PYMT_PC"])
            risk_classification = classify_value(risk, dynamic_thresholds["BENE_AVG_RISK_SCRE"])
            er_rate_classification = classify_value(er_rate, dynamic_thresholds["ER_VISITS_PER_1000_BENES"])

            # Compute Composite ML Score
            composite_score, _ = compute_composite_ml_score(
                spending, risk, er_rate, {
                    "TOT_MDCR_STDZD_PYMT_PC": dynamic_thresholds["TOT_MDCR_STDZD_PYMT_PC"],
                    "BENE_AVG_RISK_SCRE": dynamic_thresholds["BENE_AVG_RISK_SCRE"],
                    "ER_VISITS_PER_1000_BENES": dynamic_thresholds["ER_VISITS_PER_1000_BENES"],
                }
            )

            # Build dynamic summary messages based on classifications
            messages = []

            # Spending message
            if spending_classification == "High":
                messages.append("high spending levels")
                if recommendations and recommendations[0]["plan"] is not None:
                    recommendations[0]["plan"] += " (Given high state spending, consider comprehensive coverage.)"
            elif spending_classification == "Low":
                messages.append("low spending levels")
                if recommendations and recommendations[0]["plan"] is not None:
                    recommendations[0]["plan"] += " (Given low state spending, consider plans with lower premiums.)"
            else:
                messages.append("moderate spending levels")
                if recommendations and recommendations[0]["plan"] is not None:
                    recommendations[0]["plan"] += " (State spending levels appear moderate.)"

            # Risk message
            if risk_classification == "High":
                messages.append("a higher-than-average risk profile")
            elif risk_classification == "Low":
                messages.append("a lower-than-average risk profile")
            else:
                messages.append("an average risk profile")

            # Emergency department visit rate message
            if er_rate_classification == "High":
                messages.append("a high rate of emergency visits")
            elif er_rate_classification == "Low":
                messages.append("a low rate of emergency visits")
            else:
                messages.append("a moderate rate of emergency visits")

            # Create the primary summary sentence
            primary_summary = "State-level analysis indicates " + ", ".join(messages[:-1]) + ", and " + messages[-1] + "."

            # Create a secondary sentence based on the combination of classifications
            if spending_classification == "High" or risk_classification == "High" or er_rate_classification == "High":
                secondary_summary = "These indicators suggest significant healthcare demand, so you might benefit from a plan with comprehensive coverage and lower deductibles."
            elif spending_classification == "Low" and risk_classification == "Low" and er_rate_classification == "Low":
                secondary_summary = "These indicators imply modest healthcare demand, so a plan with lower premiums and minimal coverage may be sufficient."
            else:
                secondary_summary = "Overall, the indicators are balanced, suggesting that a moderately comprehensive plan could be the most cost-effective choice."

            ml_summary = f"{primary_summary} {secondary_summary}"

            # Generate Outlier Information using trained data
            classifications = {}
            for key, friendly_name in friendly_names.items():
                if friendly_name in ml_prediction_df.columns:
                    value = ml_prediction_df[friendly_name].iloc[0]
                    classification = classify_value(value, dynamic_thresholds[key])
                    classifications[friendly_name] = classification

            high_count = sum(1 for v in classifications.values() if v == "High")
            low_count = sum(1 for v in classifications.values() if v == "Low")

            # Construct a detailed outlier message
            outlier_message = f"Note: {state} has {high_count} 'High' classifications and {low_count} 'Low' classifications. "
            if low_count > 0:
                low_metrics = [k for k, v in classifications.items() if v == "Low"]
                outlier_message += f"Metrics classified as 'Low': {', '.join(low_metrics)}. "
            if high_count > 0:
                high_metrics = [k for k, v in classifications.items() if v == "High"]
                outlier_message += f"Metrics classified as 'High': {', '.join(high_metrics)}."

        # Generate NeuralCollaborativeFiltering recommendations
        if 0 <= user_id < USER_ITEM_MATRIX.shape[0]:
            ncf_recommendations = predict_user_item_interactions(NCF_MODEL, USER_ITEM_MATRIX, user_id)
        else:
            ncf_recommendations = []

        # Ensure "Was this recommendation helpful?" does not appear if no valid plan is recommended
        valid_recommendations = [
            rec for rec in recommendations if rec.get("priority") != "insufficient_criteria"
        ]

        # If no valid recommendations exist, add a fallback warning message
        if not valid_recommendations:
            valid_recommendations = [{
                "plan": "No plan available",
                "justification": "Insufficient inputs provided. Please provide more information to generate meaningful recommendations.",
                "priority": "warning"
            }]

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "recommendations": valid_recommendations,  # Return recommendations, including fallback
        "ml_prediction": ml_output_json,
        "ml_summary": ml_summary,
        "outlier_message": outlier_message,  # Include the outlier message
        "clarification_message": clarification_message if state else "",
        "prediction_context_message": prediction_context_message if state else "",
        "ncf_recommendations": ncf_recommendations,
    })

def hash_user_id(user_id):
    """Hash the user ID using SHA-256."""
    return hashlib.sha256(str(user_id).encode('utf-8')).hexdigest()

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    user = User(**data)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered successfully", "user_id": user.id})

@app.route('/log_interaction', methods=['POST'])
def log_interaction():
    try:
        data = request.json
        user_id = data.get('user_id')
        item_name = data.get('item_id')  # Use the selected recommendation's plan name
        rating = data.get('rating')
        user_inputs = data.get('user_inputs', {})  # Get user inputs from the request

        # Validate user_id
        if not user_id or not isinstance(user_id, int):
            return jsonify({"error": "Invalid or missing user_id. Ensure it is an integer."}), 400

        # Validate rating
        if rating is None or not isinstance(rating, (int, float)):
            return jsonify({"error": "Invalid or missing rating. Ensure it is a number."}), 400

        # Validate item_name
        if not item_name or not isinstance(item_name, str):
            return jsonify({"error": "Invalid or missing item_id. Ensure it is a string."}), 400

        # Check if the user exists in the `users` table
        user = db.session.get(User, user_id)  # Use Session.get() instead of Query.get()
        if not user:
            return jsonify({"error": f"User with id {user_id} does not exist."}), 400

        # Handle fallback case where no recommendation is generated
        if item_name == "General Feedback":
            item_name = "No Recommendation Available"
            item_description = "This feedback was provided when no specific recommendation was generated."
        else:
            item_description = "Recommended plan"

        # Ensure the item exists in the `items` table
        item = Item.query.filter_by(name=item_name).first()
        if not item:
            item = Item(name=item_name, description=item_description)
            db.session.add(item)
            db.session.commit()

        # Serialize user_inputs to JSON string before encryption
        user_inputs_json = json.dumps(user_inputs)

        # Log the interaction
        interaction = Interaction(
            user_id=user_id,  # Use the actual user ID
            item_id=item.id,
            rating=rating
        )
        interaction.set_user_inputs(user_inputs_json)  # Encrypt and store user inputs

        db.session.add(interaction)
        db.session.commit()

        return jsonify({"message": f"Feedback logged for item: {item_name}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_interactions', methods=['GET'])
def get_interactions():
    interactions = Interaction.query.all()
    data = [
        {
            "id": i.id,
            "user_id": i.user_id,
            "item_id": i.item_id,
            "rating": i.rating,
            "timestamp": i.timestamp
        }
        for i in interactions
    ]
    return jsonify(data)

if __name__ == "__main__":
    with app.app_context():
        create_tables()
    app.run(debug=True)
