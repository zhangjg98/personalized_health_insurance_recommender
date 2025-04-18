import pandas as pd
import os
import numpy as np  # Import numpy for array operations
from database import db, Item, Interaction  # Import from database.py
from thresholds import unified_thresholds  # Import dynamic thresholds
from ml_model import predict_medicare_spending, content_based_filtering  # Use trained data for thresholds
from neural_collaborative_filtering import predict_user_item_interactions, load_ncf_model, explain_ncf_predictions
from plans import PLANS  # Import plans from plans dictionary

# Description: This file contains the propositional logic for the insurance recommender system.

# Lazy loading of the NCF model and user-item matrix
NCF_MODEL = None
USER_ITEM_MATRIX = None

def load_ncf_resources():
    global NCF_MODEL, USER_ITEM_MATRIX
    try:
        # Check if there are any interactions in the database
        interaction_count = db.session.query(Interaction).count()
        if interaction_count == 0:
            print("No interactions found in the database. Skipping NCF model loading.")  # Debugging log
            NCF_MODEL = None
            USER_ITEM_MATRIX = None
            return

        # Load the user-item matrix
        USER_ITEM_MATRIX = pd.read_csv("user_item_matrix.csv", index_col=0)
        if USER_ITEM_MATRIX.empty or (USER_ITEM_MATRIX.shape[0] == 1 and USER_ITEM_MATRIX.shape[1] == 1):
            print("User-item matrix is not meaningful. Skipping NCF model loading.")  # Debugging log
            NCF_MODEL = None
            USER_ITEM_MATRIX = None
            return

        # Debugging log: Check the shape of the loaded matrix
        print(f"Loaded USER_ITEM_MATRIX with shape: {USER_ITEM_MATRIX.shape}")

        # Load the NCF model
        num_users, num_items = USER_ITEM_MATRIX.shape
        NCF_MODEL = load_ncf_model(
            model_path="ncf_model.pth",
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
    except Exception as e:
        print(f"Error loading NCF resources: {e}. Skipping NCF model loading.")  # Debugging log
        NCF_MODEL = None
        USER_ITEM_MATRIX = None

def get_or_create_item(plan_name, plan_description):
    """
    Retrieve the item_id for a plan from the database, or create a new entry if it doesn't exist.
    """
    # Prevent adding the fallback recommendation to the database
    if plan_name == "No Recommendation Available":
        return -1  # Use a placeholder item_id for fallback recommendations

    item = Item.query.filter_by(name=plan_name).first()
    if not item:
        print(f"Plan '{plan_name}' not found in the database. Adding it now.")  # Debugging log
        item = Item(name=plan_name, description=plan_description)
        db.session.add(item)
        db.session.commit()
    return item.id

# Recommendation function
def recommend_plan(user_input, priority="", ml_prediction_df=None):
    print("Starting recommend_plan function...")  # Debugging log
    print("User input:", user_input)  # Debugging log
    print("Priority:", priority)  # Debugging log
    print("ML Prediction DataFrame:\n", ml_prediction_df)  # Debugging log

    # Ensure the NCF model and matrix are loaded
    load_ncf_resources()
    if NCF_MODEL is None or USER_ITEM_MATRIX is None:
        print("Collaborative filtering skipped due to missing NCF model or user-item matrix.")  # Debugging log

    # Handle the case where there are no interactions
    interaction_count = db.session.query(Interaction).count()
    if interaction_count == 0:
        print("No interactions found in the database. Skipping collaborative and content-based filtering.")  # Debugging log

    # Proceed with rule-based recommendations regardless of interactions or matrix availability
    recommendations = []

    # If USER_ITEM_MATRIX is available, proceed with matrix-related logic
    if USER_ITEM_MATRIX is not None:
        print("USER_ITEM_MATRIX is available. Proceeding with matrix-related logic.")  # Debugging log
        try:
            # Ensure item_id values are integers
            USER_ITEM_MATRIX.columns = USER_ITEM_MATRIX.columns.astype(int)

            # Generate the mapping directly from the user-item matrix columns
            matrix_index_to_item_id = {index: item_id for index, item_id in enumerate(USER_ITEM_MATRIX.columns)}
            item_id_to_matrix_index = {item_id: index for index, item_id in matrix_index_to_item_id.items()}

            # Debugging log: Check mappings
            print("Matrix index to item_id mapping:", matrix_index_to_item_id)
            print("Item_id to matrix index mapping:", item_id_to_matrix_index)

            # Verify that all items in the matrix have interactions
            valid_item_ids = set(USER_ITEM_MATRIX.columns)
            interactions = Interaction.query.all()
            interaction_item_ids = {int(i.item_id) for i in interactions}  # Ensure item_id is an integer

            # Log items with interactions but missing from the matrix
            missing_from_matrix = interaction_item_ids - valid_item_ids
            if missing_from_matrix:
                print(f"Warning: The following item_ids have interactions but are missing from the user-item matrix: {missing_from_matrix}")

            # Log items in the matrix but without interactions
            missing_interactions = valid_item_ids - interaction_item_ids
            if missing_interactions:
                print(f"Warning: The following item_ids are in the user-item matrix but have no interactions: {missing_interactions}")

            # Extract user inputs
            user_id = user_input.get("user_id", -1)
            if user_id == -1:
                print("Invalid user_id provided.")  # Debugging log
                return [{
                    "plan": "No plan available",
                    "justification": "Invalid user ID provided.",
                    "priority": "error"
                }]

            # Map the actual user_id to the zero-based index in the matrix
            try:
                user_index = USER_ITEM_MATRIX.index.tolist().index(user_id)  # Map to zero-based index
                print(f"Mapped user_id {user_id} to user_index {user_index}.")  # Debugging log
            except ValueError:
                print(f"User ID {user_id} not found in the user-item matrix.")  # Debugging log
                return [{
                    "plan": "No Recommendation Available",
                    "justification": "User ID not found in the user-item matrix. Please provide feedback to improve recommendations.",
                    "priority": "fallback"
                }]
        except Exception as e:
            print(f"Error during matrix-related logic: {e}")  # Debugging log
    else:
        print("USER_ITEM_MATRIX is not available. Skipping matrix-related logic.")  # Debugging log

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
        if not isinstance(demographic_thresholds, dict):
            raise TypeError("Expected demographic_thresholds to be a dictionary.")
    except Exception as e:
        print(f"Error loading demographic thresholds: {e}")
        demographic_thresholds = {}

    # High-priority rules (evaluated first)
    if smoker == "yes":
        plan = PLANS["high_deductible_smokers"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if not priority and not preferred_plan_type:
            return recommendations

    if bmi_category == "underweight":
        plan = PLANS["nutritional_support_underweight"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if not priority and not preferred_plan_type:
            return recommendations
    elif bmi_category in ["overweight", "obese"]:
        plan = PLANS["wellness_programs_overweight"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })
        if not priority and not preferred_plan_type:
            return recommendations

    if chronic_condition == "yes":
        if medical_care_frequency == "Low":
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
        if not priority and not preferred_plan_type:
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

    # Process each demographic category
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

    # Age-based recommendations
    if age_group == "young_adult" and not recommendations:
        if income == "below_30000":
            if chronic_condition == "yes":
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
        elif bmi_category == "normal" and chronic_condition == "no" and medical_care_frequency == "Low":
            plan = PLANS["high_deductible_low_premium_young_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    if age_group == "adult" and not recommendations:
        if income == "below_30000":
            plan = PLANS["affordable_preventive_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif medical_care_frequency == "High":
            plan = PLANS["moderate_deductible_adults"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    if age_group == "senior" and not recommendations:
        if chronic_condition == "yes" or medical_care_frequency == "High":
            plan = PLANS["low_deductible_seniors"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    # Family size adjustments
    if family_size == "4_plus" and not recommendations:
        if income == "above_100000":
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
    elif family_size == "2_to_3" and not recommendations:
        plan = PLANS["family_plans_preventive_moderate_deductibles"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })

    # Gender-based recommendations
    if gender == "female" and not recommendations:
        plan = PLANS["maternity_preventive_care_females"]
        item_id = get_or_create_item(plan["name"], plan["description"])
        recommendations.append({
            "item_id": item_id,
            "plan": plan["name"],
            "justification": plan["description"],
            "priority": "strongly recommended"
        })

    # Generalized Income-based recommendations
    if not recommendations:
        if income == "below_30000" and age_group != "young_adult":
            plan = PLANS["low_cost_subsidized_coverage"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif income in ["75000_to_99999", "30000_to_74999"]:
            plan = PLANS["hdhp_with_hsa"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })
        elif income == "above_100000":
            plan = PLANS["ppo_flexibility_high_income"]
            item_id = get_or_create_item(plan["name"], plan["description"])
            recommendations.append({
                "item_id": item_id,
                "plan": plan["name"],
                "justification": plan["description"],
                "priority": "strongly recommended"
            })

    # Include priority-based recommendations alongside other recommendations
    if priority:
        plan_key = priority.lower().replace(" ", "_")  # Convert priority to match the plan key format
        plan = PLANS.get(plan_key)
        if plan:
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
    if preferred_plan_type:
        plan = PLANS.get(f"{preferred_plan_type.lower()}_plan")
        if plan:
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
        if not preferred_plan_type:
            return recommendations

        # Separate recommendations by priority
        strongly_recommended = [
            rec for rec in recommendations
            if rec["priority"] == "strongly recommended" and preferred_plan_type.lower() in rec["plan"].lower()
        ]
        priority_based = [
            rec for rec in recommendations
            if rec["priority"] == "user-selected priority" and preferred_plan_type.lower() in rec["plan"].lower()
        ]
        generic_preferred = [
            rec for rec in recommendations
            if rec["priority"] == "user-selected" and preferred_plan_type.lower() in rec["plan"].lower()
        ]

        # Combine recommendations in the correct priority order
        filtered_recommendations = strongly_recommended + priority_based

        # Add one generic preferred plan type recommendation if no matches exist in the first two categories
        if not filtered_recommendations and generic_preferred:
            filtered_recommendations.append(generic_preferred[0])

        # Add other recommendations that do not match the preferred plan type
        other_recommendations = [
            rec for rec in recommendations
            if rec not in strongly_recommended + priority_based + generic_preferred
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
            if "item_id" not in rec:
                print(f"Error: Missing `item_id` in recommendation: {rec}")  # Debugging log
                continue
            if rec["item_id"] not in seen_item_ids:
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
            if "item_id" not in rec:
                print(f"Error: Missing `item_id` in recommendation: {rec}")  # Debugging log
                continue
            if rec["item_id"] not in seen_item_ids:
                unique_recommendations.append(rec)
                seen_item_ids.add(rec["item_id"])
        return unique_recommendations

    # Ensure content-based and collaborative filtering are only applied when no strongly recommended plans exist
    strongly_recommended_plans = [
        rec for rec in recommendations if rec.get("priority") == "strongly recommended"
    ]

    if not strongly_recommended_plans and USER_ITEM_MATRIX is not None:
        # Step 1: Filter irrelevant plans during content-based filtering
        def filter_irrelevant_plans(user_input, plans):
            filtered_plans = []
            for plan in plans:
                if user_input.get("smoker") == "no" and "smoker" in plan["description"].lower():
                    continue
                if user_input.get("chronic_condition") == "no" and "chronic" in plan["description"].lower():
                    continue
                if user_input.get("family_size") not in ["2_to_3", "4_plus"] and "family" in plan["description"].lower():
                    continue
                if user_input.get("age") == "young_adult" and "senior" in plan["description"].lower():
                    continue
                if user_input.get("preferred_plan_type") and user_input["preferred_plan_type"] not in plan["name"]:
                    continue
                filtered_plans.append(plan)
            return filtered_plans

        plans = [{"id": item.id, "name": item.name, "description": item.description} for item in Item.query.all()]
        filtered_plans = filter_irrelevant_plans(user_input, plans)

        # Debugging log: Check filtered plans
        print("Filtered plans:", filtered_plans)

        # Step 2: Perform content-based filtering with hard constraints
        plans = [{"id": item.id, "name": item.name, "description": item.description} for item in Item.query.all()]
        filtered_plans = filter_irrelevant_plans(user_input, plans)

        # Debugging log: Check filtered plans
        print("Filtered plans:", filtered_plans)

        # Step 2.1: Validate plans against the user-item matrix
        valid_item_ids = set(USER_ITEM_MATRIX.columns)
        filtered_plans = [plan for plan in filtered_plans if plan["id"] in valid_item_ids]

        # Debugging log: Check valid plans after filtering against the matrix
        print("Valid plans after filtering against the user-item matrix:", filtered_plans)

        # Step 2.2: Log warnings for plans with no interactions but do not exclude them
        plans_with_no_interactions = [plan for plan in filtered_plans if plan["id"] not in interaction_item_ids]
        if plans_with_no_interactions:
            print(f"Warning: The following plans exist in the user-item matrix but have no interactions: {[plan['id'] for plan in plans_with_no_interactions]}")

        # Ensure all plans in the matrix are considered valid, even if they have no interactions
        filtered_plans = [plan for plan in filtered_plans if plan["id"] in valid_item_ids]

        # Debugging log: Final filtered plans
        print("Final filtered plans after relaxing interaction-based validation:", filtered_plans)

        # Step 2.2: Perform content-based filtering
        if not filtered_plans:
            print("No valid plans found after filtering against the user-item matrix.")  # Debugging log
        else:
            ranked_plans = content_based_filtering(user_input, filtered_plans)

            # Limit to the top-ranked content-based recommendation
            if ranked_plans:
                top_plan = ranked_plans[0]
                item_id = top_plan.get("id")  # Ensure item_id is set
                if item_id is not None:
                    recommendations.append({
                        "item_id": item_id,
                        "plan": top_plan.get("name"),
                        "justification": top_plan.get("description"),
                        "similarity_score": top_plan.get("similarity_score"),
                        "priority": "content-based",
                        "disclaimer_note": "These plans are generated using advanced filtering techniques to provide additional insights."
                    })

        # Debugging log: Check recommendations after limiting content-based filtering
        print("Recommendations after limiting content-based filtering:", recommendations)

        # Step 3: Rank filtered plans using collaborative filtering
        def rank_with_collaborative_filtering(recommendations, user_index):
            if USER_ITEM_MATRIX is None:
                print("USER_ITEM_MATRIX is None. Skipping collaborative filtering.")  # Debugging log
                return recommendations
            try:
                if USER_ITEM_MATRIX.shape[1] < 2:
                    raise ValueError("Insufficient items in the user-item matrix for collaborative filtering.")

                predicted_scores = predict_user_item_interactions(
                    NCF_MODEL,
                    USER_ITEM_MATRIX,
                    user_index,
                    top_k=None,
                    matrix_index_to_item_id=matrix_index_to_item_id  # Pass the corrected mapping
                )

                for rec in recommendations:
                    item_id = rec.get("item_id")
                    if item_id not in predicted_scores:
                        print(f"Warning: item_id {item_id} is not in the predicted scores. Skipping.")
                        rec["score"] = 0
                        continue

                    rec["score"] = predicted_scores[item_id]
                    # Ensure only one disclaimer is added
                    if "disclaimer_note" not in rec:
                        rec["disclaimer_note"] = "These plans are generated using advanced filtering techniques to provide additional insights."

                recommendations.sort(key=lambda x: x["score"], reverse=True)
                return recommendations
            except Exception as e:
                print(f"Error during collaborative filtering: {e}")
                return recommendations

        recommendations = rank_with_collaborative_filtering(recommendations, user_index)

        # Debugging log: Check recommendations after collaborative filtering
        print("Recommendations after collaborative filtering:", recommendations)

    # Filter recommendations to include only items present in the matrix for collaborative/content-based filtering
    if USER_ITEM_MATRIX is not None:
        valid_item_ids = set(USER_ITEM_MATRIX.columns)
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
        print("USER_ITEM_MATRIX is None. Skipping filtering based on matrix.")  # Debugging log
        filtered_recommendations = recommendations

    # Debugging log: Check filtered recommendations
    print("Filtered recommendations:", filtered_recommendations)

    # Limit the number of recommendations displayed to the user
    MAX_RECOMMENDATIONS = 3
    final_recommendations = remove_duplicates(filtered_recommendations)[:MAX_RECOMMENDATIONS] if USER_ITEM_MATRIX is not None else filtered_recommendations[:MAX_RECOMMENDATIONS]

    # Fallback if no recommendations exist
    if not final_recommendations:
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
        rec for rec in recommendations if rec["priority"] != "fallback"
    ] + [
        rec for rec in recommendations if rec["priority"] == "fallback"
    ]

    # Ensure disclaimers are removed for user-selected plans
    def remove_disclaimer_for_user_selected(recommendations):
        """
        Remove disclaimers from user-selected recommendations.
        """
        for rec in recommendations:
            if rec.get("priority") == "user-selected":
                rec.pop("disclaimer_note", None)

    # Ensure disclaimers are removed for user-selected plans
    remove_disclaimer_for_user_selected(recommendations)

    # Ensure all recommendations are JSON-serializable
    for rec in final_recommendations:
        try:
            # Ensure `item_id` exists
            if "item_id" not in rec:
                print(f"Error: Missing `item_id` in recommendation before serialization: {rec}")  # Debugging log
                rec["item_id"] = -1  # Add a placeholder `item_id` to prevent errors
            rec["score"] = float(rec["score"]) if rec.get("score") is not None else 0.0
            rec["similarity_score"] = float(rec["similarity_score"]) if rec.get("similarity_score") is not None else 0.0
            if "explanation" in rec and isinstance(rec["explanation"], list):
                rec["explanation"] = [
                    {"feature": str(entry["feature"]), "impact": float(entry["impact"])}
                    for entry in rec["explanation"]
                ]
        except Exception as e:
            print(f"Error serializing recommendation: {rec}, Error: {e}")  # Debugging log
            rec["explanation"] = "Error serializing explanation."

    # Debugging log: Check recommendations after serialization
    print("Recommendations after serialization:", final_recommendations)

    # Debugging log: Check recommendations before returning
    print("Final recommendations before returning:", final_recommendations)

    return final_recommendations
