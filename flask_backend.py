from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import os  # Import the os module
import hashlib
import json  # Ensure JSON encoding/decoding for user_inputs
import multiprocessing
import atexit
import warnings
from plans import PLANS  # Import the PLANS dictionary
from database import db, User, Item, Interaction, UserItemMatrix  # Import models from database.py
from propositional_logic import recommend_plan
from ml_model import predict_medicare_spending
from thresholds import unified_thresholds
from neural_collaborative_filtering import load_ncf_model
from evaluation_metrics import evaluate_model_metrics # Import from evaluation_metrics.py
from dotenv import load_dotenv
import psutil  # Import psutil for memory and CPU monitoring
from supabase_storage import download_file_if_needed

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app, origins=["https://zhangjg98.github.io"], methods=["GET", "POST", "OPTIONS"])

print("Flask app initialized.")  # Debugging log

# Add this line to disable CSRF protection for testing
app.config['WTF_CSRF_ENABLED'] = False

# Configure PostgreSQL database
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')  # Set the database URI from the environment variable
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)  # Initialize the database with the Flask app

def create_tables():
    with app.app_context():
        db.create_all()

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
try:
    # Check if there are any interactions in the database
    with app.app_context():
        interaction_count = db.session.query(Interaction).count()
        if interaction_count == 0:
            print("No interactions found in the database. Skipping NCF model loading.")  # Debugging log
            NCF_MODEL = None
            USER_ITEM_MATRIX = None
        else:
            # Download user-item matrix and model from Supabase
            user_item_matrix_path = download_file_if_needed("user_item_matrix.csv")
            model_path = download_file_if_needed("ncf_model.pth")

            # Load the user-item matrix to determine dimensions
            USER_ITEM_MATRIX = pd.read_csv(user_item_matrix_path, index_col=0)
            if USER_ITEM_MATRIX.empty or (USER_ITEM_MATRIX.shape[0] == 1 and USER_ITEM_MATRIX.shape[1] == 1):
                print("User-item matrix is not meaningful. Skipping NCF model loading.")  # Debugging log
                NCF_MODEL = None
                USER_ITEM_MATRIX = None
            else:
                num_users, num_items = USER_ITEM_MATRIX.shape
                NCF_MODEL = load_ncf_model(
                    model_path=model_path,
                    num_users=num_users,
                    num_items=num_items,
                    latent_dim=50,
                    hidden_dim=128
                )
                print("NCF model and user-item matrix loaded successfully.")  # Debugging log
except FileNotFoundError:
    print("The user_item_matrix.csv file was not found. Skipping NCF model loading.")  # Debugging log
    NCF_MODEL = None
    USER_ITEM_MATRIX = None
except ValueError as e:
    print(f"Error loading NCF model: {e}. Skipping NCF model loading.")  # Debugging log
    NCF_MODEL = None
    USER_ITEM_MATRIX = None

@app.before_request
def log_request_info():
    print("Received a request.")
    print("Request method:", request.method)
    print("Request URL:", request.url)
    print("Request headers:", request.headers)
    if request.method == "POST":
        print("Request body:", request.get_json())

@app.route('/')
def home():
    return render_template('index.html')

def classify_value(value, thresholds):
    """
    Classify a value as 'Low', 'Moderate', or 'High' based on thresholds.
    """
    low = thresholds.get("low")
    high = thresholds.get("high")

    # Ensure thresholds are not None
    if low is None or high is None:
        print(f"Invalid thresholds: low={low}, high={high}")  # Debugging log
        return "Unknown"

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
        print("Starting /recommend endpoint.")
        print("Request Headers:", request.headers)  # Log request headers

        # Log memory usage at the start of the request
        process = psutil.Process(os.getpid())
        print(f"Memory usage at start: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        try:
            user_input = request.get_json()
            print("Received user input:", user_input)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            user_input = None

        # Generate ML predictions and insights only if a state is provided
        ml_prediction_df = None
        state = user_input.get('state', '').strip()
        if state:
            print(f"Fetching ML predictions for state: {state}")  # Debugging log
            try:
                ml_prediction_df = predict_medicare_spending(state)
                print("ML Prediction DataFrame:\n", ml_prediction_df)  # Debugging log
            except Exception as e:
                print(f"Error during ML prediction for state '{state}': {e}")  # Debugging log
                return jsonify({"error": f"Error during ML prediction for state '{state}': {e}"}), 500

        # Log memory usage before calling recommend_plan
        print(f"Memory usage before recommend_plan: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Pass ML predictions to the recommendation logic
        print("Calling recommend_plan with user input and ML predictions...")  # Debugging log
        try:
            recommendations = recommend_plan(user_input, ml_prediction_df=ml_prediction_df)
            print("Generated recommendations:", recommendations)  # Debugging log
        except Exception as e:
            print(f"Error during recommend_plan execution: {e}")  # Debugging log
            return jsonify({"error": f"Error during recommend_plan execution: {e}"}), 500

        # Log memory usage after recommend_plan
        print(f"Memory usage after recommend_plan: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        # Remove disclaimer from backend response
        recommendations = [
            rec for rec in recommendations if rec["priority"] != "disclaimer"
        ]

        # Remove fallback recommendations from being treated as content-based
        recommendations = [
            rec for rec in recommendations if rec["priority"] != "fallback"
        ] + [
            rec for rec in recommendations if rec["priority"] == "fallback"
        ]

        # Ensure state-level messages are only generated when a state is provided
        ml_output_json = []
        ml_summary = ""
        outlier_message = ""
        if state and ml_prediction_df is not None:
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
            elif spending_classification == "Low":
                messages.append("low spending levels")
            else:
                messages.append("moderate spending levels")

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

        # Ensure all recommendations and insights are JSON-serializable
        try:
            print("Validating JSON serialization compatibility...")  # Debugging log
            for rec in recommendations:
                rec["plan"] = str(rec["plan"]) if rec.get("plan") else "No plan available"
                rec["justification"] = str(rec["justification"])
                rec["priority"] = str(rec["priority"])
                rec["score"] = float(rec["score"]) if rec.get("score") is not None else 0.0
                rec["similarity_score"] = float(rec["similarity_score"]) if rec.get("similarity_score") is not None else 0.0
            for record in ml_output_json:
                for key, value in record.items():
                    if isinstance(value, (np.float32, np.float64)):
                        record[key] = float(value)
                    elif isinstance(value, (np.int32, np.int64)):
                        record[key] = int(value)
            print("Recommendations and insights after ensuring JSON compatibility:", recommendations, ml_output_json)  # Debugging log
        except Exception as e:
            print(f"Error ensuring JSON serialization compatibility: {e}")  # Debugging log
            return jsonify({"error": f"Error ensuring JSON serialization compatibility: {e}"}), 500

        return jsonify({
            "recommendations": recommendations,
            "ml_prediction": ml_output_json,
            "ml_summary": ml_summary,
            "outlier_message": outlier_message,
        })
    except Exception as e:
        print(f"Unhandled exception in /recommend endpoint: {e}")  # Debugging log
        return jsonify({"error": f"Unhandled exception: {e}"}), 500

def hash_user_id(user_id):
    """Hash the user ID using SHA-256."""
    return hashlib.sha256(str(user_id).encode('utf-8')).hexdigest()

def get_or_create_item(plan_name, plan_description="Recommended plan"):
    """
    Retrieve the item_id for a plan from the database, or create a new entry if it doesn't exist.
    """
    item = Item.query.filter_by(name=plan_name).first()
    if not item:
        item = Item(name=plan_name, description=plan_description)
        db.session.add(item)
        db.session.commit()
    return item.id

@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    # Remove the 'id' field if it exists in the request data
    data.pop('id', None)
    user = User(**data)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message": "User registered successfully", "user_id": user.id})

@app.route('/log_interaction', methods=['POST'])
def log_interaction():
    try:
        data = request.json
        print("Received interaction data:", data)  # Debugging log

        user_id = str(data.get('user_id'))  # Convert user_id to string
        item_id = data.get('item_id')  # This could be an integer, a string, or a plan_name
        rating = data.get('rating')
        user_inputs = data.get('user_inputs', {})  # Get user inputs from the request

        # Validate user_id
        if not user_id:
            return jsonify({"error": "Invalid or missing user_id. Ensure it is a string."}), 400

        # Validate rating
        if rating is None or not isinstance(rating, (int, float)):
            return jsonify({"error": "Invalid or missing rating. Ensure it is a number."}), 400

        # Handle item_id as a string that can be converted to an integer
        if isinstance(item_id, str) and item_id.isdigit():
            item_id = int(item_id)

        # Check if item_id is a valid integer; if not, treat it as a plan_name
        if isinstance(item_id, str):
            # Attempt to retrieve the item_id from the database using the plan_name
            item = Item.query.filter_by(name=item_id).first()
            if not item:
                return jsonify({"error": f"Plan '{item_id}' does not exist in the database."}), 400
            item_id = item.id
        elif not isinstance(item_id, int):
            return jsonify({"error": "Invalid or missing item_id. Ensure it is an integer or a valid plan name."}), 400

        # Check if the user exists in the `users` table
        user = db.session.get(User, user_id)  # Use Session.get() instead of Query.get()
        if not user:
            return jsonify({"error": f"User with id {user_id} does not exist."}), 400

        # Check if the item exists in the `items` table
        item = db.session.get(Item, item_id)
        if not item:
            return jsonify({"error": f"Item with id {item_id} does not exist."}), 400

        # Serialize user_inputs to JSON string before encryption
        user_inputs_json = json.dumps(user_inputs)

        # Log the interaction
        interaction = Interaction(
            user_id=user_id,  # Use the actual user ID
            item_id=item_id,  # Use the correct item ID from the request
            rating=rating
        )
        interaction.set_user_inputs(user_inputs_json)  # Encrypt and store user inputs

        db.session.add(interaction)
        db.session.commit()

        print(f"Interaction logged: user_id={user_id}, item_id={item_id}, rating={rating}")  # Debugging log
        return jsonify({"message": f"Feedback logged for item: {item.name}"})
    except Exception as e:
        print(f"Error logging interaction: {e}")  # Debugging log
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

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        if NCF_MODEL is None or USER_ITEM_MATRIX is None:
            return jsonify({"error": "Model or user-item matrix not loaded."}), 500

        # More permissive dummy user input for evaluation
        user_input = {
            "age": "adult",
            "smoker": "",  # Allow both smokers and non-smokers
            "bmi": "",
            "income": "",
            "family_size": "",
            "chronic_condition": "",  # Allow both with and without chronic conditions
            "medical_care_frequency": "",
            "preferred_plan_type": "",
            "priority": "",
            "gender": "",
            "ethnicity": "",
            "state": ""
        }

        # Pass user_input and PLANS to evaluate_model_metrics
        print("Calculating evaluation metrics for the NCF model...")  # Debugging log
        k_percentage = 0.5  # Adjust k value as a percentage of the number of items
        num_items = USER_ITEM_MATRIX.shape[1]
        k_value = int(num_items * k_percentage)
        metrics = evaluate_model_metrics(NCF_MODEL, USER_ITEM_MATRIX.values, k=k_value, user_inputs=user_input, plans=PLANS)
        print(f"Evaluation Metrics: Precision@{k_value}={metrics['Precision@K']:.4f}, Recall@{k_value}={metrics['Recall@K']:.4f}, "
              f"NDCG@{k_value}={metrics['NDCG@K']:.4f}, Hit Rate@{k_value}={metrics['Hit Rate']:.4f}")  # Debugging log

        return jsonify(metrics)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return jsonify({"error": str(e)}), 500

# Function to clean up multiprocessing resources
def cleanup_resources():
    print("Cleaning up resources...")
    # Terminate all active child processes
    for child in multiprocessing.active_children():
        print(f"Terminating child process: {child.pid}")
        child.terminate()
        child.join()  # Ensure the process is fully terminated

    # Explicitly clean up semaphore objects
    try:
        print("Cleaning up leaked semaphore objects...")
        from multiprocessing import resource_tracker
        resource_tracker.unregister("/mp-semaphore", "semaphore")  # Unregister semaphore objects
        print("Semaphore cleanup completed.")
    except Exception as e:
        print(f"Error during semaphore cleanup: {e}")

    # Suppress warnings about leaked semaphores
    warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")

    print("All resources cleaned up.")

# Register the cleanup function to run at application exit
atexit.register(cleanup_resources)

if __name__ == "__main__":
    if os.environ.get("FLASK_ENV") != "production":
        with app.app_context():
            create_tables()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
