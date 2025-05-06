import pandas as pd
import os
import numpy as np  # Import numpy for array operations
from database import db, Item, Interaction  # Import from database.py
from thresholds import unified_thresholds  # Import dynamic thresholds
from ml_model import content_based_filtering  # Use trained data for thresholds
from neural_collaborative_filtering import predict_user_item_interactions
from plans import PLANS  # Import plans from plans dictionary
from utils import filter_irrelevant_plans  # Import the filtering function
from resource_manager import get_user_item_matrix, get_ncf_model  # Use resource manager for shared resources
import torch

# Description: This file contains the propositional logic for the insurance recommender system.

def get_or_create_item(plan_name, plan_description):
    """
    Retrieve the item_id for a plan from the database, or create a new entry if it doesn't exist.
    """
    # Prevent adding the fallback recommendation to the database
    if (plan_name == "No Recommendation Available"):
        return -1  # Use a placeholder item_id for fallback recommendations

    item = Item.query.filter_by(name=plan_name).first()
    if (not item):
        print(f"Plan '{plan_name}' not found in the database. Adding it now.")  # Debugging log
        item = Item(name=plan_name, description=plan_description)
        db.session.add(item)
        db.session.commit()
    return item.id

def batch_predict_scores(model, user_idx, item_indices):
    """
    Predict scores for all items for a given user in a single batch.
    Always returns a 1D numpy array.
    """
    model.eval()
    with torch.no_grad():
        user_tensor = torch.LongTensor([user_idx] * len(item_indices))
        item_tensor = torch.LongTensor(item_indices)
        scores_tensor = model(user_tensor, item_tensor)
        scores = scores_tensor.cpu().numpy()  # Ensure it's a NumPy array
        if scores.ndim == 0:
            scores = np.array([scores.item()])
        return scores

# Recommendation function
def recommend_plan(user_input, priority="", ml_prediction_df=None):
    print("Starting recommend_plan function...")
    print(f"User input: {user_input}")

    user_item_matrix_df = get_user_item_matrix()
    ncf_model = get_ncf_model()

    if user_item_matrix_df is None or ncf_model is None:
        print("Error: User-item matrix or NCF model is not loaded.")
        return [{"priority": "error", "justification": "Model or matrix not available."}]

    user_item_matrix = user_item_matrix_df.to_numpy()
    print(f"User-item matrix loaded with shape: {user_item_matrix.shape}")
    print(f"Priority: {priority}")

    # Avoid printing large DataFrames or objects
    if ml_prediction_df is not None:
        print("ML Prediction DataFrame is available.")
    else:
        print("ML Prediction DataFrame is None.")

    # Use a static user ID for all users
    user_id = 1
    user_input["user_id"] = user_id

    # Handle the case where there are no interactions
    interaction_count = db.session.query(Interaction).count()
    if (interaction_count == 0):
        print("No interactions found in the database. Skipping collaborative and content-based filtering.")  # Debugging log

    # Proceed with rule-based recommendations regardless of interactions or matrix availability
    recommendations = []

    # If user_item_matrix_df is available, proceed with matrix-related logic
    if (user_item_matrix_df is not None):
        print("user_item_matrix_df is available. Proceeding with matrix-related logic.")  # Debugging log
        try:
            # Ensure item_id values are integers
            user_item_matrix_df.columns = user_item_matrix_df.columns.astype(int)

            # Generate the mapping directly from the user-item matrix columns
            matrix_index_to_item_id = {index: item_id for index, item_id in enumerate(user_item_matrix_df.columns)}
            item_id_to_matrix_index = {item_id: index for index, item_id in matrix_index_to_item_id.items()}

            # Debugging log: Check mappings
            print("Matrix index to item_id mapping:", matrix_index_to_item_id)
            print("Item_id to matrix index mapping:", item_id_to_matrix_index)

            # Verify that all items in the matrix have interactions
            valid_item_ids = set(user_item_matrix_df.columns)
            interactions = Interaction.query.all()
            interaction_item_ids = {int(i.item_id) for i in interactions}  # Ensure item_id is an integer

            # Log items with interactions but missing from the matrix
            missing_from_matrix = interaction_item_ids - valid_item_ids
            if (missing_from_matrix):
                print(f"Warning: The following item_ids have interactions but are missing from the user-item matrix: {missing_from_matrix}")

            # Log items in the matrix but without interactions
            missing_interactions = valid_item_ids - interaction_item_ids
            if (missing_interactions):
                print(f"Warning: The following item_ids are in the user-item matrix but have no interactions: {missing_interactions}")

            # Extract user inputs
            user_id = user_input.get("user_id", -1)
            if (user_id == -1):
                print("Invalid user_id provided.")  # Debugging log
                return [{
                    "plan": "No plan available",
                    "justification": "Invalid user ID provided.",
                    "priority": "error"
                }]

            # Map the actual user_id to the zero-based index in the matrix
            user_index = None  # Initialize user_index to None
            if user_id in user_item_matrix_df.index:
                user_index = user_item_matrix_df.index.tolist().index(user_id)
                print(f"Mapped user_id {user_id} to user_index {user_index}.")  # Debugging log
            else:
                print(f"User ID {user_id} not found in the user-item matrix.")  # Debugging log

                user_index = None  # Skip collaborative filtering for this user

            # Ensure the user-item matrix is a NumPy array
            if user_item_matrix is not None:
                print("Converting user-item matrix to NumPy array...")  # Debugging log
                user_item_matrix = user_item_matrix_df.values

        except Exception as e:
            print(f"Error during matrix-related logic: {e}")  # Debugging log
    else:
        print("user_item_matrix_df is not available. Skipping matrix-related logic.")  # Debugging log

    # Extract additional user inputs
    age_group = user_input.get('age', '18-29')
    smoker = user_input.get('smoker', 'no')
    bmi_category = user_input.get('bmi', '')
    income = user_input.get('income', '')
    family_size = user_input.get('family_size', '')
    chronic_condition = user_input.get('chronic_condition', 'no')
    medical_care_frequency = user_input.get('medical_care_frequency', 'Low')
    preferred_plan_type = user_input.get('preferred_plan_type', '')
    priority = user_input.get('priority', '')
    gender = user_input.get('gender', '')
    ethnicity = user_input.get('ethnicity', '').lower()

    recommendations = []

    # Load dynamic thresholds for demographic fields based on predicted values
    demographic_keys = ["BENE_FEML_PCT", "BENE_RACE_BLACK_PCT", "BENE_RACE_HSPNC_PCT"]
    try:
        demographic_thresholds = unified_thresholds(
            "processed_user_item_matrix.csv",
            demographic_keys,
            lower_quantile=0.2,
            upper_quantile=0.8,
            scale_factor=1.1
        )

        # Validate that demographic_thresholds is a dictionary
        if (not isinstance(demographic_thresholds, dict)):
            raise TypeError("Expected demographic_thresholds to be a dictionary.")
    except Exception as e:
        print(f"Error loading demographic thresholds: {e}")
        demographic_thresholds = {}

    # High-priority rules (evaluated first)
    if (smoker == "yes"):
        plan = PLANS["high_deductible_smokers"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if (not priority and not preferred_plan_type):
            return recommendations

    if (bmi_category == "underweight"):
        plan = PLANS["nutritional_support_underweight"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if (not priority and not preferred_plan_type):
            return recommendations
    elif (bmi_category in ["overweight", "obese"]):
        plan = PLANS["wellness_programs_overweight"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if (not priority and not preferred_plan_type):
            return recommendations

    if (chronic_condition == "yes"):
        if (medical_care_frequency == "Low"):
            plan = PLANS["chronic_care_low_frequency"]
        else:
            plan = PLANS["chronic_care_high_frequency"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if (not priority and not preferred_plan_type):
            return recommendations

    # Demographic-based recommendations (evaluated independently)
    predicted_female = predicted_black = predicted_hispanic = None
    if ml_prediction_df is not None and not ml_prediction_df.empty:
        try:
            if "Percent Female" in ml_prediction_df.columns:
                predicted_female = float(ml_prediction_df["Percent Female"].iloc[0])
            if "Percent African American" in ml_prediction_df.columns:
                predicted_black = float(ml_prediction_df["Percent African American"].iloc[0])
            if "Percent Hispanic" in ml_prediction_df.columns:
                predicted_hispanic = float(ml_prediction_df["Percent Hispanic"].iloc[0])
        except Exception as e:
            print(f"Error processing demographic predictions: {e}")
            predicted_female = predicted_black = predicted_hispanic = None

    # Debugging logs for predicted values
    print(f"Predicted Female: {predicted_female}, Predicted Black: {predicted_black}, Predicted Hispanic: {predicted_hispanic}")

    def process_recommendation(demographic, predicted_value, thresholds, plan_name, plan_description):
        print(f"Processing recommendation for {demographic}...")  # Debugging log
        print(f"{demographic} Predicted Value: {predicted_value}, {demographic} Thresholds: {thresholds}")  # Debugging log

        # Ensure predicted_value and thresholds["high"] are not None
        if predicted_value is None or thresholds.get("high") is None or thresholds.get("low") is None:
            print(f"Skipping {demographic} recommendation due to missing predicted value or thresholds.")
            return

        try:
            # Ensure predicted_value is a scalar and not an array
            if isinstance(predicted_value, (list, np.ndarray)):
                if len(predicted_value) == 0:
                    print(f"Skipping {demographic} recommendation due to empty predicted value array.")
                    return
                predicted_value = np.array(predicted_value).reshape(-1)[0]  # Convert to scalar if necessary

            if predicted_value > thresholds["high"]:
                print(f"{demographic} predicted value ({predicted_value}) is above the high threshold ({thresholds['high']}).")
                item_id = get_or_create_item(plan_name, plan_description)
                recommendations.append({
                    "item_id": item_id,
                    "plan": plan_name,
                    "justification": plan_description,
                    "priority": "strongly recommended"
                })
            else:
                print(f"{demographic} predicted value ({predicted_value}) is not above the high threshold ({thresholds['high']}).")
        except TypeError as e:
            print(f"TypeError during comparison for {demographic}: {e}")
        except Exception as e:
            print(f"Unexpected error during processing for {demographic}: {e}")

    # Process each demographic category only if ml_prediction_df is valid
    if ml_prediction_df is not None and not ml_prediction_df.empty:
        if gender == "female":
            plan = PLANS["enhanced_womens_health"]
            process_recommendation(
                "Female",
                predicted_female,
                demographic_thresholds.get("BENE_FEML_PCT", {}),
                plan["name"],
                plan["description"]
            )

        if ethnicity == "black":
            plan = PLANS["preventive_care_chronic_conditions_african_americans"]
            process_recommendation(
                "Black",
                predicted_black,
                demographic_thresholds.get("BENE_RACE_BLACK_PCT", {}),
                plan["name"],
                plan["description"]
            )

        if ethnicity == "hispanic":
            plan = PLANS["culturally_relevant_healthcare_hispanics"]
            process_recommendation(
                "Hispanic",
                predicted_hispanic,
                demographic_thresholds.get("BENE_RACE_HSPNC_PCT", {}),
                plan["name"],
                plan["description"]
            )
    else:
        print("Skipping demographic-based recommendations due to invalid or missing ML predictions.")  # Debugging log

    # Age-based recommendations
    if (age_group == "young_adult" and not recommendations):
        if (income == "below_30000"):
            if (chronic_condition == "yes"):
                plan = PLANS["medicaid_or_subsidized_young_adults"]
            else:
                plan = PLANS["catastrophic_young_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif (bmi_category == "normal" and chronic_condition == "no" and medical_care_frequency == "Low"):
            plan = PLANS["high_deductible_low_premium_young_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    if (age_group == "adult" and not recommendations):
        if (income == "below_30000"):
            plan = PLANS["affordable_preventive_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif (medical_care_frequency == "High"):
            plan = PLANS["moderate_deductible_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    if (age_group == "senior" and not recommendations):
        if (chronic_condition == "yes" or medical_care_frequency == "High"):
            plan = PLANS["low_deductible_seniors"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    # Family size adjustments
    if (family_size == "4_plus" and not recommendations):
        if (income == "above_100000"):
            plan = PLANS["ppo_family_coverage_high_income"]
        else:
            plan = PLANS["family_coverage_pediatric_maternity"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
    elif (family_size == "2_to_3" and not recommendations):
        plan = PLANS["family_plans_preventive_moderate_deductibles"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })

    # Gender-based recommendations
    if (gender == "female" and not recommendations):
        plan = PLANS["maternity_preventive_care_females"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })

    # Generalized Income-based recommendations
    if (not recommendations):
        if (income == "below_30000" and age_group != "young_adult"):
            plan = PLANS["low_cost_subsidized_coverage"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif (income in ["75000_to_99999", "30000_to_74999"]):
            plan = PLANS["hdhp_with_hsa"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif (income == "above_100000"):
            plan = PLANS["ppo_flexibility_high_income"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    # Include priority-based recommendations alongside other recommendations
    if (priority):
        plan_key = priority.lower().replace(" ", "_")  # Convert priority to match the plan key format
        plan = PLANS.get(plan_key)
        if (plan):
            print(f"Generating priority-based recommendation for priority: {priority}")  # Debugging log
            item_id = get_or_create_item(plan["name"], plan["description"])
            priority_recommendation = {
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "user-selected priority"
            }
            recommendations.append(priority_recommendation)  # Add priority-based plan to the recommendations
        else:
            print(f"Error: No plan found for priority: {priority}")  # Debugging log

    # Rules based on Preferred Plan Type
    if (preferred_plan_type):
        plan = PLANS.get(f"{preferred_plan_type.lower()}_plan")
        if (plan):
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "user-selected"
            })

    def filter_preferred_plan_type_recommendations(recommendations, preferred_plan_type):
        """
        Filter recommendations to ensure:
        1. Strongly recommended plans for the preferred plan type are prioritized.
        2. Priority-based plans for the preferred plan type are considered next.
        3. A generic preferred plan type recommendation is included if no specific matches exist.
        """
        if (not preferred_plan_type):
            return recommendations

        # Separate recommendations by priority
        strongly_recommended = [
            rec for rec in recommendations
            if (rec["priority"] == "strongly recommended" and preferred_plan_type.lower() in rec["plan"].lower())
        ]
        priority_based = [
            rec for rec in recommendations
            if (rec["priority"] == "user-selected priority" and preferred_plan_type.lower() in rec["plan"].lower())
        ]
        generic_preferred = [
            rec for rec in recommendations
            if (rec["priority"] == "user-selected" and preferred_plan_type.lower() in rec["plan"].lower())
        ]

        # Combine recommendations in the correct priority order
        filtered_recommendations = strongly_recommended + priority_based

        # Add one generic preferred plan type recommendation if no matches exist in the first two categories
        if (not filtered_recommendations and generic_preferred):
            filtered_recommendations.append(generic_preferred[0])

        # Add other recommendations that do not match the preferred plan type
        other_recommendations = [
            rec for rec in recommendations
            if (rec not in strongly_recommended + priority_based + generic_preferred)
        ]
        return filtered_recommendations + other_recommendations

    def remove_duplicates(recommendations):
        """
        Remove duplicate recommendations based on `item_id` while preserving order.
        """
        seen_item_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            # Ensure `item_id` exists before processing
            if ("item_id" not in rec):
                print(f"Error: Missing `item_id` in recommendation: {rec}")  # Debugging log
                continue
            if (rec["item_id"] not in seen_item_ids):
                unique_recommendations.append(rec)
                seen_item_ids.add(rec["item_id"])
        return unique_recommendations

    # Filter recommendations based on the preferred plan type
    recommendations = filter_preferred_plan_type_recommendations(recommendations, preferred_plan_type)

    # Combine all relevant recommendations
    def remove_duplicates(recommendations):
        """
        Remove duplicate recommendations based on `item_id` while preserving order.
        """
        seen_item_ids = set()
        unique_recommendations = []
        for rec in recommendations:
            # Ensure `item_id` exists before processing
            if ("item_id" not in rec):
                print(f"Error: Missing `item_id` in recommendation: {rec}")  # Debugging log
                continue
            if (rec["item_id"] not in seen_item_ids):
                unique_recommendations.append(rec)
                seen_item_ids.add(rec["item_id"])
        return unique_recommendations

    # Ensure content-based and collaborative filtering are only applied when no strongly recommended plans exist
    strongly_recommended_plans = [
        rec for rec in recommendations if (rec.get("priority") == "strongly recommended")
    ]

    if (not strongly_recommended_plans and user_item_matrix is not None):
        # Step 1: Filter irrelevant plans during content-based filtering
        plans = [{"id": item.id, "name": item.name, "description": item.description} for item in Item.query.all()]
        filtered_plans = filter_irrelevant_plans(plans, user_input)

        # Step 2.1: Validate plans against the user-item matrix
        valid_item_ids = set(user_item_matrix_df.columns)
        filtered_plans = [plan for plan in filtered_plans if (plan["id"] in valid_item_ids)]

        # Step 2.2: Log warnings for plans with no interactions but do not exclude them
        plans_with_no_interactions = [plan for plan in filtered_plans if (plan["id"] not in interaction_item_ids)]
        if (plans_with_no_interactions):
            print(f"Warning: The following plans exist in the user-item matrix but have no interactions: {[plan['id'] for plan in plans_with_no_interactions]}")

        # Ensure all plans in the matrix are considered valid, even if they have no interactions
        filtered_plans = [plan for plan in filtered_plans if (plan["id"] in valid_item_ids)]

        # Step 2.3: Perform content-based filtering
        if (not filtered_plans):
            print("No valid plans found after filtering against the user-item matrix.")  # Debugging log
        else:
            # Debugging: Print item scores before content-based filtering
            print("Item scores before content-based filtering:")
            item_scores = {}
            for plan in filtered_plans:
                item_id = plan["id"]
                if item_id in item_id_to_matrix_index:
                    matrix_index = item_id_to_matrix_index[item_id]
                    # Get the collaborative filtering score for the item
                    cf_score = batch_predict_scores(
                        ncf_model,
                        user_index,
                        [matrix_index]
                    )[0]  # Default to 0 if not found
                    item_scores[item_id] = cf_score
                    print(f"Item ID: {item_id}, Matrix Index: {matrix_index}, CF Score: {cf_score}")
                else:
                    print(f"Item ID: {item_id} not found in item_id_to_matrix_index")

            ranked_plans = content_based_filtering(user_input, filtered_plans, item_scores=item_scores)

            # Limit to the top-ranked content-based recommendation
            if (ranked_plans):
                top_plan = ranked_plans[0]
                item_id = top_plan.get("id")  # Ensure item_id is set
                if (item_id is not None):
                    recommendations.append({
                        "item_id": item_id,
                        "plan": top_plan.get("name"),
                        "justification": top_plan.get("description"),
                        "similarity_score": top_plan.get("similarity_score"),
                        "priority": "content-based",
                        "disclaimer_note": "These plans are generated using advanced filtering techniques to provide additional insights."
                    })

        # Step 3: Rank filtered plans using collaborative filtering
        def rank_with_collaborative_filtering(recommendations, user_index):
            if user_item_matrix is None:
                print("user_item_matrix is None. Skipping collaborative filtering.")  # Debugging log
                return recommendations
            try:
                if user_item_matrix.shape[1] < 2:
                    raise ValueError("Insufficient items in the user-item matrix for collaborative filtering.")

                # Ensure top_k is an integer
                top_k = int(user_item_matrix.shape[1] // 2)

                item_indices = list(range(user_item_matrix.shape[1]))
                predicted_scores = batch_predict_scores(
                    ncf_model,
                    user_index,
                    item_indices
                )

                for rec in recommendations:
                    item_id = rec.get("item_id")
                    if item_id not in matrix_index_to_item_id.values():
                        print(f"Warning: item_id {item_id} is not in the predicted scores. Skipping.")
                        rec["score"] = 0
                        continue

                    rec["score"] = predicted_scores[item_id_to_matrix_index[item_id]]
                    # Ensure only one disclaimer is added
                    if "disclaimer_note" not in rec:
                        rec["disclaimer_note"] = "These plans are generated using advanced filtering techniques to provide additional insights."

                recommendations.sort(key=lambda x: x["score"], reverse=True)

                return recommendations
            except Exception as e:
                print(f"Error during collaborative filtering: {e}")
                return recommendations

        recommendations = rank_with_collaborative_filtering(recommendations, user_index)

    # Filter recommendations to include only items present in the matrix for collaborative/content-based filtering
    if user_item_matrix_df is not None:  # Use user_item_matrix_df for metadata
        valid_item_ids = set(user_item_matrix_df.columns)
        filtered_recommendations = []
        for rec in recommendations:
            if rec.get("priority") in ["content-based", "collaborative filtering"]:
                # Only filter collaborative/content-based recommendations
                if rec.get("item_id") in valid_item_ids:
                    filtered_recommendations.append(rec)
                else:
                    print(f"Skipping recommendation for item_id {rec.get('item_id')} as it is not in the user-item matrix.")
            else:
                # Include rule-based recommendations without filtering
                filtered_recommendations.append(rec)
    else:
        print("user_item_matrix_df is None. Skipping filtering based on matrix.")  # Debugging log
        filtered_recommendations = recommendations

    # Limit the number of recommendations displayed to the user
    MAX_RECOMMENDATIONS = 3
    final_recommendations = remove_duplicates(filtered_recommendations)[:MAX_RECOMMENDATIONS] if (user_item_matrix is not None) else filtered_recommendations[:MAX_RECOMMENDATIONS]

    # Fallback if no recommendations exist
    if (not final_recommendations):
        fallback_plan = {
            "plan": "No Recommendation Available",
            "justification": "This feedback was provided when no specific recommendation was generated.",
            "priority": "fallback",  # Assign a distinct priority for fallback recommendations
            "item_id": -1,  # Use a placeholder `item_id` for fallback
            "score": 0.0,  # Ensure score is set to 0 for fallback
            "similarity_score": 0.0,  # Ensure similarity_score is set to 0 for fallback
            "disclaimer_note": None  # No disclaimer for fallback recommendations
        }
        final_recommendations.append(fallback_plan)

    # Ensure fallback recommendations are excluded from content-based filtering
    filtered_recommendations = [
        rec for rec in recommendations if (rec["priority"] != "fallback")
    ] + [
        rec for rec in recommendations if (rec["priority"] == "fallback")
    ]

    # Ensure disclaimers are removed for user-selected plans
    def remove_disclaimer_for_user_selected(recommendations):
        """
        Remove disclaimers from user-selected recommendations.
        """
        for rec in recommendations:
            if (rec.get("priority") == "user-selected"):
                rec.pop("disclaimer_note", None)

    # Ensure disclaimers are removed for user-selected plans
    remove_disclaimer_for_user_selected(recommendations)

    # Ensure all recommendations are JSON-serializable
    for rec in final_recommendations:
        try:
            # Ensure `item_id` exists
            if ("item_id" not in rec):
                print(f"Error: Missing `item_id` in recommendation before serialization: {rec}")  # Debugging log
                rec["item_id"] = -1  # Add a placeholder `item_id` to prevent errors
            rec["score"] = float(rec["score"]) if (rec.get("score") is not None) else 0.0
            rec["similarity_score"] = float(rec["similarity_score"]) if (rec.get("similarity_score") is not None) else 0.0
            if ("explanation" in rec and isinstance(rec["explanation"], list)):
                rec["explanation"] = [
                    {"feature": str(entry["feature"]), "impact": float(entry["impact"])}
                    for entry in rec["explanation"]
                ]
        except Exception as e:
            print(f"Error serializing recommendation: {rec}, Error: {e}")  # Debugging log
            rec["explanation"] = "Error serializing explanation."

    # Debugging log: Check recommendations before returning
    print("Final recommendations before returning:", final_recommendations)

    # Debugging log: Check the priorities of all recommendations
    print("Generated recommendations and their priorities:")
    for rec in recommendations:
        print(f"Plan: {rec.get('plan')}, Priority: {rec.get('priority')}")

    return final_recommendations
