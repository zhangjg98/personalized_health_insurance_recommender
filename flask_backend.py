from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending
from thresholds import compute_dynamic_thresholds, unified_thresholds
from neural_collaborative_filtering import load_ncf_model, predict_user_item_interactions
import pandas as pd

app = Flask(__name__)
CORS(app)

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://recommender_user:securepassword@localhost/health_insurance_recommender'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Define database models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    age_group = db.Column(db.String(50))
    smoker = db.Column(db.Boolean)
    bmi_category = db.Column(db.String(50))
    income = db.Column(db.String(50))
    family_size = db.Column(db.String(50))
    chronic_condition = db.Column(db.Boolean)
    medical_care_frequency = db.Column(db.String(50))
    preferred_plan_type = db.Column(db.String(50))

class Item(db.Model):
    __tablename__ = 'items'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    description = db.Column(db.Text)

class Interaction(db.Model):
    __tablename__ = 'interactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'))
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

class UserItemMatrix(db.Model):
    __tablename__ = 'user_item_matrix'
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'), primary_key=True)
    rating = db.Column(db.Float)

def create_tables():
    db.create_all()

with app.app_context():
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

# Load the summary of states with multiple "High" or "Low" classifications
state_summary = pd.read_csv("state_summary.csv")

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

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_input = request.json
        print("Received user input:", user_input)

        user_id = int(user_input.get("user_id", -1))
        if user_id == -1:
            print("Error: Invalid user_id")
            return jsonify({
                "error": "Invalid user_id. Please ensure you are logged in or registered."
            }), 400

        rule_recommendation = recommend_plan(user_input)
        state = user_input.get('state', '').strip()

        # Initialize variables with default values
        ml_output_json = []
        ml_summary = ""
        outlier_message = ""
        ncf_recommendations = []

        # Validate input
        if not state:
            return jsonify({
                "recommendation": rule_recommendation,
                "ml_prediction": ml_output_json,
                "ml_summary": "No state provided. Unable to generate state-level analysis.",
                "outlier_message": "",
                "ncf_recommendations": ncf_recommendations,
            })

        # Load raw values from processed_user_item_matrix.csv
        raw_data = pd.read_csv("processed_user_item_matrix.csv", index_col=0)
        if state not in raw_data.index:
            return jsonify({
                "recommendation": rule_recommendation,
                "ml_prediction": ml_output_json,
                "ml_summary": f"No data available for state: {state}.",
                "outlier_message": "",
                "ncf_recommendations": ncf_recommendations,
            })

        state_data = raw_data.loc[state]
        print(f"Raw data for {state}:\n{state_data}")

        # Classify raw values for outlier information
        classifications = {}
        for key, friendly_name in friendly_names.items():
            if key in dynamic_thresholds and key in state_data:
                value = state_data[key]
                classification = classify_value(value, dynamic_thresholds[key])
                classifications[friendly_name] = classification
                print(f"Processing {friendly_name}: Value={value}, Classification={classification}")

        # Generate ML predictions
        ml_prediction_df = predict_medicare_spending(state)
        print("ML Prediction DataFrame:\n", ml_prediction_df)
        ml_prediction_df = ml_prediction_df.rename(columns=friendly_names)

        # Add classification labels to ML predictions
        for key, friendly_name in friendly_names.items():
            if friendly_name in ml_prediction_df.columns:
                ml_prediction_df[f"{friendly_name} Level"] = ml_prediction_df[friendly_name].apply(
                    lambda x: classify_value(x, dynamic_thresholds[key])
                )

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
            print(f"Outlier message for {state}: {outlier_message}")

        # Generate NeuralCollaborativeFiltering recommendations
        if 0 <= user_id < USER_ITEM_MATRIX.shape[0]:
            ncf_recommendations = predict_user_item_interactions(NCF_MODEL, USER_ITEM_MATRIX, user_id)
        else:
            print(f"Invalid user_id: {user_id}")
            ncf_recommendations = []

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "recommendation": rule_recommendation,
        "ml_prediction": ml_output_json,
        "ml_summary": ml_summary,
        "outlier_message": outlier_message,
        "ncf_recommendations": ncf_recommendations,
    })

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    user = User(**data)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered successfully", "user_id": user.id})

@app.route('/log_interaction', methods=['POST'])
def log_interaction():
    data = request.json
    user_id = data.get('user_id')
    item_name = data.get('item_id')
    rating = data.get('rating')

    # Ensure the item exists in the `items` table
    item = Item.query.filter_by(name=item_name).first()
    if not item:
        item = Item(name=item_name, description="Recommended plan")
        db.session.add(item)
        db.session.commit()

    # Log the interaction
    interaction = Interaction(
        user_id=user_id,
        item_id=item.id,
        rating=rating
    )
    db.session.add(interaction)
    db.session.commit()

    # Update the user-item matrix in the database
    user_item_entry = UserItemMatrix.query.filter_by(user_id=user_id, item_id=item.id).first()
    if user_item_entry:
        user_item_entry.rating = rating  # Update existing entry
    else:
        user_item_entry = UserItemMatrix(user_id=user_id, item_id=item.id, rating=rating)
        db.session.add(user_item_entry)  # Add new entry
    db.session.commit()

    return jsonify({"message": f"Feedback logged for item: {item_name}"})

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

if __name__ == '__main__':
    app.run(debug=True)
