import pandas as pd
import os
import numpy as np  # Import numpy for array operations
from database import db, Item, Interaction  # Import from database.py
from thresholds import unified_thresholds  # Import dynamic thresholds
from ml_model import predict_medicare_spending, content_based_filtering  # Use trained data for thresholds
from neural_collaborative_filtering import predict_user_item_interactions, load_ncf_model, explain_ncf_predictions

# Description: This file contains the propositional logic for the insurance recommender system.

# Lazy loading of the NCF model and user-item matrix
NCF_MODEL = None
USER_ITEM_MATRIX = None

def load_ncf_resources():
    global NCF_MODEL, USER_ITEM_MATRIX
    if NCF_MODEL is None or USER_ITEM_MATRIX is None:
        NCF_MODEL, USER_ITEM_MATRIX = load_ncf_model()

def get_or_create_item(plan_name, plan_description):
    """
    Retrieve the item_id for a plan from the database, or create a new entry if it doesn't exist.
    """
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
    try:
        load_ncf_resources()
        print("NCF model and user-item matrix loaded successfully.")  # Debugging log
    except Exception as e:
        print(f"Error loading NCF resources: {e}")  # Debugging log
        raise

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
            "plan": "No plan available",
            "justification": f"User ID {user_id} not found in the user-item matrix.",
            "priority": "error"
        }]

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
        plan_name = "Plan Recommendation: Prioritize Plans with High Deductible and Preventive Care for Smokers"
        plan_description = (
            "Smokers will have higher health insurance premiums, so a high deductible plan will help mitigate "
            "the monthly premium amount. Preventive care includes services that will help with health screenings, "
            "which could be important due to risks of lung cancer and other health problems for smokers."
        )
        item_id = get_or_create_item(plan_name, plan_description)
        recommendations.append({
            "item_id": item_id,
            "plan": plan_name,
            "justification": plan_description,
            "priority": "strongly recommended"
        })

    if bmi_category == "underweight" and not recommendations:
        plan_name = "Plan Recommendation: Prioritize Plans with Specialized Nutritional Support Coverage"
        plan_description = (
            "Strongly consider a plan that provides specialized nutritional support if you have issues with being underweight. "
            "Nutritional support is a therapy that can help for people who have difficulty getting enough nourishment through eating or drinking."
        )
        item_id = get_or_create_item(plan_name, plan_description)
        recommendations.append({
            "item_id": item_id,
            "plan": plan_name,
            "justification": plan_description,
            "priority": "strongly recommended"
        })
    elif bmi_category in ["overweight", "obese"] and not recommendations:
        plan_name = "Plan Recommendation: Look Into Plans that Provide Health and Wellness Programs"
        plan_description = (
            "For those struggling with weight issues, plans that can provide health and wellness programs should be strongly considered. "
            "Moving towards a healthy lifestyle is critical in ensuring that you do not have health issues further down the line."
        )
        item_id = get_or_create_item(plan_name, plan_description)
        recommendations.append({
            "item_id": item_id,
            "plan": plan_name,
            "justification": plan_description,
            "priority": "strongly recommended"
        })

    if chronic_condition == "yes" and not recommendations:
        if medical_care_frequency == "Low":
            plan_name = "Plan Recommendation: Medication Therapy Management Program for Chronic Conditions"
            plan_description = (
                "If you do not need to see a doctor frequently but have a chronic condition, a medication therapy management program might be suited for you. "
                "This plan makes sure that you take your medications correctly and safely, and basic services related to your condition come at no cost."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })
        else:
            plan_name = "Plan Recommendation: Chronic Care Coverage"
            plan_description = (
                "Chronic care coverage is a Medicare program that helps with chronic conditions. Services include a comprehensive care plan that lists your health problems and goals "
                "as well as provide needed medication and urgent care needs."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })

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
        process_recommendation(
            "Female", 
            predicted_female, 
            demographic_thresholds.get("BENE_FEML_PCT", {}), 
            "Plan Recommendation: Consider Plans with Enhanced Women's Health Coverage", 
            "The predicted percentage of female beneficiaries is high. Consider plans that include robust maternity and women’s health services."
        )

    if ethnicity == "black":
        process_recommendation(
            "Black", 
            predicted_black, 
            demographic_thresholds.get("BENE_RACE_BLACK_PCT", {}), 
            "Plan Recommendation: Consider Plans with Preventive Care for Chronic Conditions", 
            "A higher predicted percentage of African American beneficiaries may indicate elevated risk for chronic conditions such as hypertension and diabetes. Plans with strong preventive care and chronic disease management are recommended."
        )

    if ethnicity == "hispanic":
        process_recommendation(
            "Hispanic", 
            predicted_hispanic, 
            demographic_thresholds.get("BENE_RACE_HSPNC_PCT", {}), 
            "Plan Recommendation: Consider Plans Offering Culturally Relevant Healthcare Services", 
            "A significant predicted Hispanic population suggests that plans offering bilingual support and culturally tailored services could be beneficial."
        )

    # Age-based recommendations
    if age_group == "young_adult" and not recommendations:
        if income == "below_30000":
            if chronic_condition == "yes":
                plan_name = "Plan Recommendation: Strongly Consider Medicaid or Subsidized Marketplace Plan with Chronic Care Management Services"
                plan_description = (
                    "Young adults with low income and a chronic condition should prioritize Medicaid (if eligible) or subsidized marketplace plans. "
                    "Medicaid is a government program that provides comprehensive benefits for low-income individuals at little to no cost. "
                    "If you are not eligible for Medicaid, prioritize a low-cost plan or subsidized plan that still provides chronic care management services."
                )
                item_id = get_or_create_item(plan_name, plan_description)
                recommendations.append({
                    "item_id": item_id,
                    "plan": plan_name,
                    "justification": plan_description,
                    "priority": "strongly recommended"
                })
            else:
                plan_name = "Plan Recommendation: Catastrophic Health Plan"
                plan_description = (
                    "Young adults with low income can benefit from catastrophic policies. These policies cover numerous essential health benefits like other marketplace plans, "
                    "including preventive services at no cost. They have low monthly premiums and very high deductibles, making them an affordable option for low-income young adults."
                )
                item_id = get_or_create_item(plan_name, plan_description)
                recommendations.append({
                    "item_id": item_id,
                    "plan": plan_name,
                    "justification": plan_description,
                    "priority": "strongly recommended"
                })
        elif bmi_category == "normal" and chronic_condition == "no" and medical_care_frequency == "Low":
            plan_name = "Plan Recommendation: Prioritize Plans with High Deductible and Low Premium"
            plan_description = (
                "If you are a young, healthy adult, having a plan with high deductible and low premium can be well suited for you. "
                "These plans will keep you insured in case for getting sick or injured, and the low monthly premiums will ensure that you do not have to pay too much to stay insured."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })

    if age_group == "adult" and not recommendations:
        if income == "below_30000":
            plan_name = "Plan Recommendation: Prioritize Affordable Plans that Provide Preventive Care Options"
            plan_description = (
                "Adults with low income should prioritize affordability over all else to ensure that they are covered. "
                "These plans should still include preventive care options to ensure basic health needs and services."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })
        elif medical_care_frequency == "High":
            plan_name = "Plan Recommendation: Prioritize a Moderate Deductible Plan"
            plan_description = (
                "If you have frequent medical care visits, it is important to have a deductible that is at the very least moderate. "
                "A moderate deductible ensures a reasonable balance between monthly premiums and out of pocket costs incurred until the deductible amount is hit."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })

    if age_group == "senior" and not recommendations:
        if chronic_condition == "yes" or medical_care_frequency == "High":
            plan_name = "Plan Recommendation: Prioritize Low-Deductible Plans"
            plan_description = (
                "For seniors with chronic conditions or frequent medical visits, it is best to prioritize a low-deductible plan. "
                "These types of plans ensure that you will not have to pay too much out of pocket to reach your deductible amount. "
                "This in turn will allow you to have your plan cover frequent medical expenses after reaching the deductible amount."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })

    # Family size adjustments
    if family_size == "4_plus" and not recommendations:
        if income == "above_100000":
            plan_name = "Plan Recommendation: Consider Preferred Provider Organization (PPO) Plans with Family Coverage and Comprehensive Benefits"
            plan_description = (
                "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. "
                "You ensure that your family members will get the care they need while not restricting them."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })
        else:
            plan_name = "Plan Recommendation: Consider Family Coverage with Pediatric and Maternity Benefits"
            plan_description = (
                "Pediatric and maternity benefits will help cover medical care for children and mothers (for pregnancy and childbirth) respectively. "
                "Making sure that your family gets the benefits they need is important with a family."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })
    elif family_size == "2_to_3" and not recommendations:
        plan_name = "Plan Recommendation: Consider Family Plans with Preventive Care and Moderate Deductibles"
        plan_description = (
            "For smaller families, a family health plan that balances affordability with services like preventive care is ideal. "
            "Moderate deductibles ensure manageable out-of-pocket costs while providing coverage for routine check-ups and family-specific needs."
        )
        item_id = get_or_create_item(plan_name, plan_description)
        recommendations.append({
            "item_id": item_id,
            "plan": plan_name,
            "justification": plan_description,
            "priority": "strongly recommended"
        })

    # Gender-based recommendations
    if gender == "female" and not recommendations:
        plan_name = "Plan Recommendation: Prioritize Plans with Maternity and Preventive Care Benefits"
        plan_description = (
            "For females, plans with maternity benefits and preventive care services are highly recommended to ensure comprehensive coverage for women's health needs."
        )
        item_id = get_or_create_item(plan_name, plan_description)
        recommendations.append({
            "item_id": item_id,
            "plan": plan_name,
            "justification": plan_description,
            "priority": "strongly recommended"
        })

    # Generalized Income-based recommendations
    if not recommendations:
        if income == "below_30000" and age_group != "young_adult":
            plan_name = "Plan Recommendation: Consider Low Cost or Subsidized Coverage"
            plan_description = (
                "Low cost plans ensure that you stay covered for basic medical needs. Subsidized coverage is health coverage that comes at reduced or low costs for people below certain income thresholds. "
                "Both are affordable ways to ensure you still get sufficient medical coverage."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })
        elif income in ["75000_to_99999", "30000_to_74999"]:
            plan_name = "Plan Recommendation: High Deductible Health Plans (HDHPs) with Health Savings Account (HSA)"
            plan_description = (
                "HDHPs have high deductibles and low premiums. A HSA allows you to pay that deductible amount and other medical costs with money set aside in an account where you can contribute and withdraw tax-free. "
                "HDHPs and HSAs usually go hand in hand, making them a solid option for people in your income range."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })
        elif income == "above_100000":
            plan_name = "Plan Recommendation: Preferred Provider Organization (PPO) Plans that Ensure Flexibility"
            plan_description = (
                "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. "
                "You can get the care that you want at any provider you would like, though costs will be lower if you do choose to go with an in-network provider."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "strongly recommended"
            })

    # Rules based on Preferred Plan Type
    preferred_plan_recommendation = None
    if preferred_plan_type:
        if preferred_plan_type == "HMO":
            preferred_plan_recommendation = {
                "plan": "Plan Recommendation: Health Maintenance Organization (HMO) Plan",
                "justification": "HMO plans typically offer lower premiums and coordinated care within a network, making them ideal if you prefer managed care.",
                "priority": "user-selected"
            }
        elif preferred_plan_type == "PPO":
            preferred_plan_recommendation = {
                "plan": "Plan Recommendation: Preferred Provider Organization (PPO) Plan",
                "justification": "PPO plans provide more flexibility in choosing providers and generally offer comprehensive benefits.",
                "priority": "user-selected"
            }
        elif preferred_plan_type == "EPO":
            preferred_plan_recommendation = {
                "plan": "Plan Recommendation: Exclusive Provider Organization (EPO) Plan",
                "justification": "EPO plans require using a network of providers but often have lower premiums than PPOs. They’re a good option if you don’t need out-of-network coverage.",
                "priority": "user-selected"
            }
        elif preferred_plan_type == "POS":
            preferred_plan_recommendation = {
                "plan": "Plan Recommendation: Point of Service (POS) Plan",
                "justification": "POS plans combine features of HMOs and PPOs, offering more flexibility than HMOs while keeping costs relatively low.",
                "priority": "user-selected"
            }

    # Ensure preferred plan type recommendation is added if no conflicts exist
    if preferred_plan_type and preferred_plan_recommendation:
        item_id = get_or_create_item(preferred_plan_recommendation["plan"], preferred_plan_recommendation["justification"])
        preferred_plan_recommendation["item_id"] = item_id

        # Check if a strongly recommended plan matching the preferred plan type exists
        matching_preferred_plan_exists = any(
            rec.get("priority") == "strongly recommended" and preferred_plan_type in rec.get("plan", "")
            for rec in recommendations
        )

        # Filter out all generic recommendations for the preferred plan type if a strongly recommended match exists
        recommendations = [
            rec for rec in recommendations if not (
                rec.get("priority") == "user-selected" and preferred_plan_type in rec.get("plan", "")
            )
        ]

        # Add the preferred plan type recommendation only if no strongly recommended match exists
        if not matching_preferred_plan_exists:
            recommendations.append(preferred_plan_recommendation)

    # Validate recommendations before processing
    if not recommendations:
        print("No recommendations generated. Returning fallback recommendation.")  # Debugging log
        return [{
            "plan": "No plan available",
            "justification": "We could not generate a recommendation based on the provided inputs.",
            "priority": "insufficient_criteria"
        }]

    # Filter recommendations to prioritize "Strongly Recommended" plans
    strongly_recommended_plans = [
        rec for rec in recommendations if rec.get("priority") == "strongly recommended"
    ]

    # Include priority-based recommendations
    priority_recommendations = []
    if priority:  # Keep this block
        priority_plan_name = None
        priority_plan_description = None

        if priority == "Low Premiums":
            priority_plan_name = "Plan Recommendation: High Deductible, Low Premium Plan"
            priority_plan_description = (
                "Since you prioritize low premiums, a high deductible plan is recommended. These plans have lower monthly costs, making them more affordable."
            )
        elif priority == "Comprehensive Coverage":
            priority_plan_name = "Plan Recommendation: Comprehensive PPO Plan"
            priority_plan_description = (
                "Since you value comprehensive coverage, a PPO plan is recommended. These plans provide flexibility and access to a wide range of healthcare providers."
            )
        elif priority == "Preventive Care":
            priority_plan_name = "Plan Recommendation: Preventive Care-Focused Plan"
            priority_plan_description = (
                "Since you prioritize preventive care, this plan includes extensive preventive services to help you maintain good health."
            )
        elif priority == "Low Deductibles":
            priority_plan_name = "Plan Recommendation: Low Deductible Plan"
            priority_plan_description = (
                "Since you prefer low deductibles, this plan minimizes out-of-pocket costs before insurance coverage begins."
            )

        if priority_plan_name and priority_plan_description:
            item_id = get_or_create_item(priority_plan_name, priority_plan_description)
            priority_recommendations.append({
                "item_id": item_id,
                "plan": priority_plan_name,
                "justification": priority_plan_description,
                "priority": "user-selected priority"
            })

    # Include preferred plan type recommendations
    preferred_plan_recommendations = []
    if preferred_plan_type:
        preferred_plan_name = None
        preferred_plan_description = None

        if preferred_plan_type == "HMO":
            preferred_plan_name = "Plan Recommendation: Health Maintenance Organization (HMO) Plan"
            preferred_plan_description = (
                "HMO plans typically offer lower premiums and coordinated care within a network, making them ideal if you prefer managed care."
            )
        elif preferred_plan_type == "PPO":
            preferred_plan_name = "Plan Recommendation: Preferred Provider Organization (PPO) Plan"
            preferred_plan_description = (
                "PPO plans provide more flexibility in choosing providers and generally offer comprehensive benefits."
            )
        elif preferred_plan_type == "EPO":
            preferred_plan_name = "Plan Recommendation: Exclusive Provider Organization (EPO) Plan"
            preferred_plan_description = (
                "EPO plans require using a network of providers but often have lower premiums than PPOs. They’re a good option if you don’t need out-of-network coverage."
            )
        elif preferred_plan_type == "POS":
            preferred_plan_name = "Plan Recommendation: Point of Service (POS) Plan"
            preferred_plan_description = (
                "POS plans combine features of HMOs and PPOs, offering more flexibility than HMOs while keeping costs relatively low."
            )

        if preferred_plan_name and preferred_plan_description:
            # Always include the preferred plan type recommendation
            item_id = get_or_create_item(preferred_plan_name, preferred_plan_description)
            preferred_plan_recommendations.append({
                "item_id": item_id,
                "plan": preferred_plan_name,
                "justification": preferred_plan_description,
                "priority": "user-selected"
            })

    # Combine all relevant recommendations
    final_recommendations = strongly_recommended_plans + priority_recommendations + preferred_plan_recommendations

    # Separate recommendations into categories
    strongly_recommended_plans = [
        rec for rec in recommendations if rec.get("priority") == "strongly recommended"
    ]
    supplemental_recommendations = [
        rec for rec in recommendations if rec.get("priority") != "strongly recommended"
    ]

    # Combine recommendations while ensuring `Strongly Recommended` plans appear first
    final_recommendations = strongly_recommended_plans + supplemental_recommendations + priority_recommendations

    # Remove duplicates while preserving order
    seen_item_ids = set()
    final_recommendations = [
        rec for rec in final_recommendations if not (rec["item_id"] in seen_item_ids or seen_item_ids.add(rec["item_id"]))
    ]

    if final_recommendations:
        print("Returning combined recommendations: Strongly Recommended followed by supplemental recommendations.")  # Debugging log
        return final_recommendations

    # Remove duplicates while preserving order
    seen_item_ids = set()
    final_recommendations = [
        rec for rec in final_recommendations if not (rec["item_id"] in seen_item_ids or seen_item_ids.add(rec["item_id"]))
    ]

    if final_recommendations:
        print("Returning combined recommendations: Strongly Recommended, Priority, and Preferred Plan Type.")  # Debugging log
        return final_recommendations

    # User-selected priority (evaluated after high-priority rules)
    if priority:
        if priority == "Low Premiums":
            plan_name = "Plan Recommendation: High Deductible, Low Premium Plan"
            plan_description = (
                "Since you prioritize low premiums, a high deductible plan is recommended. These plans have lower monthly costs, making them more affordable."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "user-selected"
            })
        elif priority == "Comprehensive Coverage":
            plan_name = "Plan Recommendation: Comprehensive PPO Plan"
            plan_description = (
                "Since you value comprehensive coverage, a PPO plan is recommended. These plans provide flexibility and access to a wide range of healthcare providers."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "user-selected"
            })
        elif priority == "Preventive Care":
            plan_name = "Plan Recommendation: Preventive Care-Focused Plan"
            plan_description = (
                "Since you prioritize preventive care, this plan includes extensive preventive services to help you maintain good health."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "user-selected"
            })
        elif priority == "Low Deductibles":
            plan_name = "Plan Recommendation: Low Deductible Plan"
            plan_description = (
                "Since you prefer low deductibles, this plan minimizes out-of-pocket costs before insurance coverage begins."
            )
            item_id = get_or_create_item(plan_name, plan_description)
            recommendations.append({
                "item_id": item_id,
                "plan": plan_name,
                "justification": plan_description,
                "priority": "user-selected"
            })

    # Add justification for matching preferred plan type
    for rec in recommendations:
        if preferred_plan_type and preferred_plan_type in rec.get("plan", ""):
            rec["justification"] += f" This recommendation aligns with your preferred plan type: {preferred_plan_type}."

        # Ensure the item_id is consistent with the database
        item = Item.query.filter_by(name=rec["plan"]).first()
        if item:
            rec["item_id"] = item.id
        else:
            print(f"Warning: Plan '{rec['plan']}' not found in the database.")  # Debugging log
            rec["item_id"] = None  # Set to None if not found

    # Remove recommendations with invalid item_id
    recommendations = [rec for rec in recommendations if rec["item_id"] is not None]

    # Handle conflicts between primary recommendation and preferred plan type
    if preferred_plan_type:
        # Check if any existing recommendation already matches the preferred plan type
        preferred_plan_exists = any(
            preferred_plan_type in rec.get("plan", "") for rec in recommendations
        )
        if not preferred_plan_exists and preferred_plan_recommendation:
            # Add only the default PPO plan if the preferred plan type is PPO
            if preferred_plan_type == "PPO":
                recommendations = [
                    rec for rec in recommendations if "PPO" not in rec.get("plan", "")
                ]
            recommendations.append(preferred_plan_recommendation)

    # Step 1: Apply high-priority rules first
    if age_group == "young_adult" and income == "below_30000" and chronic_condition == "no":
        plan_name = "Plan Recommendation: Catastrophic Health Plan"
        plan_description = (
            "Young adults with low income can benefit from catastrophic policies. These policies cover numerous essential health benefits like other marketplace plans, "
            "including preventive services at no cost. They have low monthly premiums and very high deductibles, making them an affordable option for low-income young adults."
        )
        item_id = get_or_create_item(plan_name, plan_description)
        recommendations.append({
            "item_id": item_id,
            "plan": plan_name,
            "justification": plan_description,
            "priority": "strongly recommended"
        })

    # Step 2: Filter irrelevant plans during content-based filtering
    plans = [{"id": item.id, "name": item.name, "description": item.description} for item in Item.query.all()]

    filtered_plans = []
    for plan in plans:
        # Apply filtering logic to exclude irrelevant plans
        if user_input.get("smoker") == "no" and "smoker" in plan["description"].lower():
            continue  # Skip plans targeted at smokers
        if user_input.get("chronic_condition") == "no" and "chronic" in plan["description"].lower():
            continue  # Skip chronic care plans for users without chronic conditions
        if user_input.get("family_size") not in ["2_to_3", "4_plus"] and "family" in plan["description"].lower():
            continue  # Skip family plans for users without families
        if user_input.get("age") == "young_adult" and "senior" in plan["description"].lower():
            continue  # Skip senior plans for young adults
        if user_input.get("preferred_plan_type") and user_input["preferred_plan_type"] not in plan["name"]:
            continue  # Skip plans that don't match the preferred plan type
        filtered_plans.append(plan)

    # Perform content-based filtering on the filtered plans
    ranked_plans = content_based_filtering(user_input, filtered_plans)

    # Step 3: Add ranked plans to recommendations
    for plan in ranked_plans:
        recommendations.append({
            "item_id": plan.get("id"),
            "plan": plan.get("name"),
            "justification": plan.get("description"),
            "similarity_score": plan.get("similarity_score"),
            "priority": "content-based"
        })

    # Step 4: Rank filtered plans using collaborative filtering
    if recommendations:
        try:
            # Ensure the user-item matrix has sufficient data
            if USER_ITEM_MATRIX.shape[1] < 2:  # Less than 2 items
                print("Insufficient items in the user-item matrix for collaborative filtering.")  # Debugging log
                raise ValueError("Insufficient items in the user-item matrix for collaborative filtering.")

            # Predict scores for all plans using collaborative filtering
            predicted_scores = predict_user_item_interactions(NCF_MODEL, USER_ITEM_MATRIX, user_index, top_k=None)

            # Validate predicted_scores
            if not predicted_scores:
                print("No valid scores generated by collaborative filtering.")  # Debugging log
            else:
                # Map item indices in the matrix to item IDs in the database
                matrix_index_to_item_id = {index: item.id for index, item in enumerate(Item.query.all())}
                item_id_to_matrix_index = {v: k for k, v in matrix_index_to_item_id.items()}  # Reverse mapping

                # Debugging log: Check mappings
                print("Matrix index to item_id mapping:", matrix_index_to_item_id)
                print("Item_id to matrix index mapping:", item_id_to_matrix_index)

                # Rank the filtered recommendations based on collaborative filtering scores
                for rec in recommendations:
                    item_id = rec.get("item_id")
                    # Find the matrix index corresponding to the item_id
                    matrix_index = item_id_to_matrix_index.get(item_id)
                    if matrix_index is not None and matrix_index in predicted_scores:
                        rec["score"] = predicted_scores[matrix_index]
                    else:
                        rec["score"] = 0  # Default score if no prediction is available

                # Filter out recommendations with invalid matrix indices
                recommendations = [
                    rec for rec in recommendations if rec.get("item_id") in item_id_to_matrix_index
                ]

                # Sort recommendations by score (highest first)
                recommendations.sort(key=lambda x: x["score"], reverse=True)

                # Remove duplicate recommendations
                seen_item_ids = set()
                recommendations = [
                    rec for rec in recommendations if not (rec["item_id"] in seen_item_ids or seen_item_ids.add(rec["item_id"]))
                ]
        except Exception as e:
            print(f"Error during collaborative filtering: {e}")  # Debugging log

    # Validate that the USER_ITEM_MATRIX has sufficient columns
    num_items_in_matrix = USER_ITEM_MATRIX.shape[1]
    num_items_in_db = Item.query.count()
    if num_items_in_matrix < num_items_in_db:
        print(f"Mismatch: USER_ITEM_MATRIX has {num_items_in_matrix} columns, but the database has {num_items_in_db} items.")
        print("Ensure that the USER_ITEM_MATRIX is updated to include all items in the database.")
        return [{
            "plan": "No plan available",
            "justification": "The user-item matrix is not aligned with the database. Please update the matrix.",
            "priority": "error"
        }]

    def validate_shap_values(shap_values):
        """
        Validate SHAP values to ensure they are a 2D NumPy array with numeric values.
        """
        if shap_values is None:
            print("SHAP values are None.")  # Debugging log
            return False
        if not isinstance(shap_values, np.ndarray):
            print(f"SHAP values are not a NumPy array: {type(shap_values)}")  # Debugging log
            return False
        if shap_values.ndim != 2:
            print(f"SHAP values do not have 2 dimensions: {shap_values.ndim}")  # Debugging log
            return False
        if not np.issubdtype(shap_values.dtype, np.number):
            print(f"SHAP values contain non-numeric data: {shap_values}")  # Debugging log
            return False
        return True

    def validate_feature_names(feature_names):
        """
        Validate feature names to ensure they are a list of strings.
        """
        if not isinstance(feature_names, list):
            print(f"Feature names are not a list: {type(feature_names)}")  # Debugging log
            return False
        if not all(isinstance(name, str) for name in feature_names):
            print(f"Feature names contain non-string elements: {feature_names}")  # Debugging log
            return False
        return True

    def validate_shap_explanation(explanation):
        """
        Validate SHAP explanation to ensure it is a list of dictionaries with numeric 'impact' values.
        """
        if not isinstance(explanation, list):
            print(f"Invalid explanation format: {explanation} (not a list)")  # Debugging log
            return False
        for entry in explanation:
            if not isinstance(entry, dict):
                print(f"Invalid explanation entry: {entry} (not a dictionary)")  # Debugging log
                return False
            if "feature" not in entry or "impact" not in entry:
                print(f"Missing keys in explanation entry: {entry}")  # Debugging log
                return False
            if not isinstance(entry["feature"], str):
                print(f"Invalid feature type in explanation entry: {entry}")  # Debugging log
                return False
            if not isinstance(entry["impact"], (int, float)):
                print(f"Invalid impact type in explanation entry: {entry}")  # Debugging log
                return False
        return True

    # Step 5: SHAP Explanations for NCF Predictions
    valid_item_ids = set(item_id_to_matrix_index.keys())  # Get valid item IDs from the matrix
    for rec in recommendations:
        try:
            # Ensure the item_id corresponds to a valid matrix index
            if rec["item_id"] not in valid_item_ids:
                print(f"Skipping SHAP explanation for invalid item_id: {rec['item_id']}")  # Debugging log
                rec["explanation"] = "Invalid item_id for SHAP explanation."
                continue

            matrix_index = item_id_to_matrix_index[rec["item_id"]]
            if matrix_index >= USER_ITEM_MATRIX.shape[1]:  # Validate matrix index range
                print(f"Matrix index {matrix_index} out of range for item_id: {rec['item_id']}")  # Debugging log
                rec["explanation"] = (
                    f"Matrix index {matrix_index} is out of range for the user-item matrix. "
                    "This item may not be represented in the matrix."
                )
                continue

            # Generate SHAP values
            shap_values = explain_ncf_predictions(NCF_MODEL, USER_ITEM_MATRIX, user_index, matrix_index)
            print(f"SHAP values for item_id {rec['item_id']}: {shap_values}")  # Debugging log

            # Validate SHAP values
            if not isinstance(shap_values, np.ndarray) or shap_values.ndim != 2:
                print(f"Invalid SHAP values for item_id {rec['item_id']}: {shap_values}")  # Debugging log
                rec["explanation"] = "Invalid SHAP values."
                continue

            # Ensure feature names are valid
            feature_names = USER_ITEM_MATRIX.columns.tolist()
            if not validate_feature_names(feature_names):
                print(f"Invalid feature names: {feature_names}")  # Debugging log
                rec["explanation"] = "Invalid feature names."
                continue

            # Simplify SHAP explanations for users
            top_features = sorted(
                enumerate(shap_values[0]), key=lambda x: abs(x[1]), reverse=True
            )[:3]  # Top 3 features
            explanation = [
                {"feature": feature_names[i], "impact": round(float(impact), 4)}
                for i, impact in top_features if isinstance(impact, (int, float))
            ]

            # Validate SHAP explanation
            if validate_shap_explanation(explanation):
                rec["explanation"] = explanation
            else:
                rec["explanation"] = "Invalid SHAP explanation format."
            print(f"Generated SHAP explanation for item_id {rec['item_id']}: {rec['explanation']}")  # Debugging log
        except Exception as e:
            print(f"Error generating SHAP explanation for item {rec['item_id']}: {e}")  # Debugging log
            rec["explanation"] = "Error occurred while generating SHAP explanation."

    # Ensure all recommendations are JSON-serializable
    for rec in recommendations:
        try:
            print(f"Before serialization: {rec}")  # Debugging log
            rec["score"] = float(rec["score"]) if rec.get("score") is not None else 0.0  # Handle None values
            rec["similarity_score"] = float(rec["similarity_score"]) if rec.get("similarity_score") is not None else 0.0  # Handle None values
            if "explanation" in rec and isinstance(rec["explanation"], list):
                # Ensure explanation is JSON-serializable
                rec["explanation"] = [
                    {"feature": str(entry["feature"]), "impact": float(entry["impact"])}
                    for entry in rec["explanation"]
                ]
            print(f"After serialization: {rec}")  # Debugging log
        except Exception as e:
            print(f"Error serializing recommendation {rec}: {e}")  # Debugging log
            rec["explanation"] = "Error serializing explanation."

    # Validate the final recommendations list
    try:
        print("Validating final recommendations...")  # Debugging log
        for rec in recommendations:
            if not isinstance(rec, dict):
                raise ValueError(f"Invalid recommendation format: {rec}")
            if "plan" not in rec or "justification" not in rec or "priority" not in rec:
                raise ValueError(f"Missing keys in recommendation: {rec}")
            if "explanation" in rec and not isinstance(rec["explanation"], (list, str)):
                raise ValueError(f"Invalid explanation format in recommendation: {rec}")
        print("Final recommendations validated successfully.")  # Debugging log
    except Exception as e:
        print(f"Error validating recommendations: {e}")  # Debugging log
        return [{
            "plan": "No plan available",
            "justification": "An error occurred while validating recommendations.",
            "priority": "error"
        }]

    # Debugging log: Final recommendations
    print("Final recommendations:", recommendations)  # Debugging log
    return recommendations
