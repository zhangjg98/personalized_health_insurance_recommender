import pandas as pd
import os
from database import db, Item  # Import db and Item from database.py
from thresholds import unified_thresholds  # Import dynamic thresholds

# Description: This file contains the propositional logic for the insurance recommender system.

# Recommendation function
def recommend_plan(user_input, priority="", ml_prediction_df=None):
    age_group = user_input.get('age', '18-29')
    smoker = user_input.get('smoker', 'no')
    bmi_category = user_input.get('bmi', '')
    income = user_input.get('income', '')
    family_size = user_input.get('family_size', '')
    chronic_condition = user_input.get('chronic_condition', 'no')
    medical_care_frequency = user_input.get('medical_care_frequency', 'Low')
    preferred_plan_type = user_input.get('preferred_plan_type', '')
    gender = user_input.get('gender', '')
    ethnicity = user_input.get('ethnicity', '')

    recommendations = []

    # High-priority rules (evaluated first)
    if smoker == "yes":
        recommendations.append({
            "plan": "Plan Recommendation: Prioritize Plans with High Deductible and Preventive Care for Smokers",
            "justification": "Smokers will have higher health insurance premiums, so a high deductible plan will help mitigate the monthly premium amount. Preventive care includes services that will help with health screenings, which could be important due to risks of lung cancer and other health problems for smokers.",
            "priority": "strongly recommended"
        })

    if bmi_category == "underweight" and not recommendations:
        recommendations.append({
            "plan": "Plan Recommendation: Prioritize Plans with Specialized Nutritional Support Coverage",
            "justification": "Strongly consider a plan that provides specialized nutritional support if you have issues with being underweight. Nutritional support is a therapy that can help for people who have difficulty getting enough nourishment through eating or drinking.",
            "priority": "strongly recommended"
        })
    elif bmi_category in ["overweight", "obese"] and not recommendations:
        recommendations.append({
            "plan": "Plan Recommendation: Look Into Plans that Provide Health and Wellness Programs",
            "justification": "For those struggling with weight issues, plans that can provide health and wellness programs should be strongly considered. Moving towards a healthy lifestyle is critical in ensuring that you do not have health issues further down the line.",
            "priority": "strongly recommended"
        })

    if chronic_condition == "yes" and not recommendations:
        if medical_care_frequency == "Low":
            recommendations.append({
                "plan": "Plan Recommendation: Medication Therapy Management Program for Chronic Conditions",
                "justification": "If you do not need to see a doctor frequently but have a chronic condition, a medication therapy management program might be suited for you. This plan makes sure that you take your medications correctly and safely, and basic services related to your condition come at no cost.",
                "priority": "strongly recommended"
            })
        else:
            recommendations.append({
                "plan": "Plan Recommendation: Chronic Care Coverage",
                "justification": "Chronic care coverage is a Medicare program that helps with chronic conditions. Services include a comprehensive care plan that lists your health problems and goals as well as provide needed medication and urgent care needs.",
                "priority": "strongly recommended"
            })

    # Age-based recommendations
    if age_group == "young_adult" and not recommendations:
        if income == "below_30000":
            if chronic_condition == "yes":
                recommendations.append({
                    "plan": "Plan Recommendation: Strongly Consider Medicaid or Subsidized Marketplace Plan with Chronic Care Management Services",
                    "justification": "Young adults with low income and a chronic condition should prioritize Medicaid (if eligible) or subsidized marketplace plans. Medicaid is a government program that provides comprehensive benefits for low-income individuals at little to no cost. If you are not eligible for Medicaid, prioritize a low-cost plan or subsidized plan that still provides chronic care management services.",
                    "priority": "strongly recommended"
                })
            else:
                recommendations.append({
                    "plan": "Plan Recommendation: Catastrophic Health Plan",
                    "justification": "Young adults with low income can benefit from catastrophic policies. These policies cover numerous essential health benefits like other marketplace plans, including preventive services at no cost. They have low monthly premiums and very high deductibles, making them an affordable option for low-income young adults.",
                    "priority": "strongly recommended"
                })
        elif bmi_category == "normal" and chronic_condition == "no" and medical_care_frequency == "Low":
            recommendations.append({
                "plan": "Plan Recommendation: Prioritize Plans with High Deductible and Low Premium",
                "justification": "If you are a young, healthy adult, having a plan with high deductible and low premium can be well suited for you. These plans will keep you insured in case for getting sick or injured, and the low monthly premiums will ensure that you do not have to pay too much to stay insured.",
                "priority": "strongly recommended"
            })

    if age_group == "adult" and not recommendations:
        if income == "below_30000":
            recommendations.append({
                "plan": "Plan Recommendation: Prioritize Affordable Plans that Provide Preventive Care Options",
                "justification": "Adults with low income should prioritize affordability over all else to ensure that they are covered. These plans should still include preventive care options to ensure basic health needs and services.",
                "priority": "strongly recommended"
            })
        elif medical_care_frequency == "High":
            recommendations.append({
                "plan": "Plan Recommendation: Prioritize a Moderate Deductible Plan",
                "justification": "If you have frequent medical care visits, it is important to have a deductible that is at the very least moderate. A moderate deductible ensures a reasonable balance between monthly premiums and out of pocket costs incurred until the deductible amount is hit.",
                "priority": "strongly recommended"
            })

    if age_group == "senior" and not recommendations:
        if chronic_condition == "yes" or medical_care_frequency == "High":
            recommendations.append({
                "plan": "Plan Recommendation: Prioritize Low-Deductible Plans",
                "justification": "For seniors with chronic conditions or frequent medical visits, it is best to prioritize a low-deductible plan. These types of plans ensure that you will not have to pay too much out of pocket to reach your deductible amount. This in turn will allow you to have your plan cover frequent medical expenses after reaching the deductible amount.",
                "priority": "strongly recommended"
            })

    # Family size adjustments
    if family_size == "4_plus" and not recommendations:
        if income == "above_100000":
            recommendations.append({
                "plan": "Plan Recommendation: Consider Preferred Provider Organization (PPO) Plans with Family Coverage and Comprehensive Benefits",
                "justification": "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. You ensure that your family members will get the care they need while not restricting them.",
                "priority": "strongly recommended"
            })
        else:
            recommendations.append({
                "plan": "Plan Recommendation: Consider Family Coverage with Pediatric and Maternity Benefits",
                "justification": "Pediatric and maternity benefits will help cover medical care for children and mothers (for pregnancy and childbirth) respectively. Making sure that your family gets the benefits they need is important with a family.",
                "priority": "strongly recommended"
            })
    elif family_size == "2_to_3" and not recommendations:
        recommendations.append({
            "plan": "Plan Recommendation: Consider Family Plans with Preventive Care and Moderate Deductibles",
            "justification": "For smaller families, a family health plan that balances affordability with services like preventive care is ideal. Moderate deductibles ensure manageable out-of-pocket costs while providing coverage for routine check-ups and family-specific needs.",
            "priority": "strongly recommended"
        })

    # Gender-based recommendations
    if gender == "female" and not recommendations:
        recommendations.append({
            "plan": "Plan Recommendation: Prioritize Plans with Maternity and Preventive Care Benefits",
            "justification": "For females, plans with maternity benefits and preventive care services are highly recommended to ensure comprehensive coverage for women's health needs.",
            "priority": "strongly recommended"
        })

    # Generalized Income-based recommendations (only if no priority is selected)
    if not priority and not recommendations:
        if income == "below_30000" and age_group != "young_adult":
            recommendations.append({
                "plan": "Plan Recommendation: Consider Low Cost or Subsidized Coverage",
                "justification": "Low cost plans ensure that you stay covered for basic medical needs. Subsidized coverage is health coverage that comes at reduced or low costs for people below certain income thresholds. Both are affordable ways to ensure you still get sufficient medical coverage.",
                "priority": "strongly recommended"
            })
        elif income in ["75000_to_99999", "30000_to_74999"]:
            recommendations.append({
                "plan": "Plan Recommendation: High Deductible Health Plans (HDHPs) with Health Savings Account (HSA)",
                "justification": "HDHPs have high deductibles and low premiums. A HSA allows you to pay that deductible amount and other medical costs with money set aside in an account where you can contribute and withdraw tax-free. HDHPs and HSAs usually go hand in hand, making them a solid option for people in your income range.",
                "priority": "strongly recommended"
            })
        elif income == "above_100000":
            recommendations.append({
                "plan": "Plan Recommendation: Preferred Provider Organization (PPO) Plans that Ensure Flexibility",
                "justification": "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. You can get the care that you want at any provider you would like, though costs will be lower if you do choose to go with an in-network provider.",
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

    # User-selected priority (evaluated after high-priority rules)
    if priority:
        if priority == "Low Premiums":
            recommendations.append({
                "plan": "Plan Recommendation: High Deductible, Low Premium Plan",
                "justification": "Since you prioritize low premiums, a high deductible plan is recommended. These plans have lower monthly costs, making them more affordable.",
                "priority": "user-selected"
            })
        elif priority == "Comprehensive Coverage":
            recommendations.append({
                "plan": "Plan Recommendation: Comprehensive PPO Plan",
                "justification": "Since you value comprehensive coverage, a PPO plan is recommended. These plans provide flexibility and access to a wide range of healthcare providers.",
                "priority": "user-selected"
            })
        elif priority == "Preventive Care":
            recommendations.append({
                "plan": "Plan Recommendation: Preventive Care-Focused Plan",
                "justification": "Since you prioritize preventive care, this plan includes extensive preventive services to help you maintain good health.",
                "priority": "user-selected"
            })
        elif priority == "Low Deductibles":
            recommendations.append({
                "plan": "Plan Recommendation: Low Deductible Plan",
                "justification": "Since you prefer low deductibles, this plan minimizes out-of-pocket costs before insurance coverage begins.",
                "priority": "user-selected"
            })

    # Add demographic-based recommendations using ML-predicted values
    try:
        # Extract predicted demographic values
        predicted_female = float(ml_prediction_df["Percent Female"].iloc[0])
        predicted_male = float(ml_prediction_df["Percent Male"].iloc[0])
        predicted_white = float(ml_prediction_df["Percent Non-Hispanic White"].iloc[0])
        predicted_black = float(ml_prediction_df["Percent African American"].iloc[0])
        predicted_hispanic = float(ml_prediction_df["Percent Hispanic"].iloc[0])
    except Exception as e:
        # If demographic predictions are unavailable, skip these rules
        predicted_female = predicted_male = predicted_white = predicted_black = predicted_hispanic = None

    # Load dynamic thresholds for demographic fields
    demographic_keys = ["BENE_FEML_PCT", "BENE_RACE_BLACK_PCT", "BENE_RACE_HSPNC_PCT"]
    demographic_thresholds = unified_thresholds("processed_user_item_matrix.csv", demographic_keys)

    # Add recommendations based on demographic thresholds
    if predicted_female is not None:
        female_thresholds = demographic_thresholds.get("BENE_FEML_PCT", {})
        if female_thresholds and predicted_female > female_thresholds.get("high", 0.55):
            recommendations.append({
                "plan": "Plan Recommendation: Consider Plans with Enhanced Women's Health Coverage",
                "justification": "The predicted percentage of female beneficiaries is high. Consider plans that include robust maternity and women’s health services.",
                "priority": "additional"
            })

    if predicted_black is not None:
        black_thresholds = demographic_thresholds.get("BENE_RACE_BLACK_PCT", {})
        if black_thresholds and predicted_black > black_thresholds.get("high", 0.15):
            recommendations.append({
                "plan": "Plan Recommendation: Consider Plans with Preventive Care for Chronic Conditions",
                "justification": "A higher predicted percentage of African American beneficiaries may indicate elevated risk for chronic conditions such as hypertension and diabetes. Plans with strong preventive care and chronic disease management are recommended.",
                "priority": "additional"
            })

    if predicted_hispanic is not None:
        hispanic_thresholds = demographic_thresholds.get("BENE_RACE_HSPNC_PCT", {})
        if hispanic_thresholds and predicted_hispanic > hispanic_thresholds.get("high", 0.10):
            recommendations.append({
                "plan": "Plan Recommendation: Consider Plans Offering Culturally Relevant Healthcare Services",
                "justification": "A significant predicted Hispanic population suggests that plans offering bilingual support and culturally tailored services could be beneficial.",
                "priority": "additional"
            })

    # Add justification for matching preferred plan type
    for rec in recommendations:
        if preferred_plan_type and preferred_plan_type in rec.get("plan", ""):
            rec["justification"] += f" This recommendation aligns with your preferred plan type: {preferred_plan_type}."

    # Handle conflicts between primary recommendation and preferred plan type
    if preferred_plan_type:
        # Check if any existing recommendation already matches the preferred plan type
        preferred_plan_exists = any(
            preferred_plan_type in rec.get("plan", "") for rec in recommendations
        )
        if not preferred_plan_exists and preferred_plan_recommendation:
            recommendations.append(preferred_plan_recommendation)

    # Fallback: Recommend the highest-rated plan if no recommendations exist
    if not recommendations:
        recommendations.append({
            "plan": None,  # Use None to indicate no valid plan
            "justification": "Please select more criteria so that we can produce a meaningful recommendation.",
            "priority": "insufficient_criteria"
        })

    # Ensure recommendations is not empty
    if not recommendations:
        recommendations.append({
            "plan": "Plan Recommendation: Contact a representative for personalized advice",
            "justification": "Based on the information provided, a representative is more likely to help you identify the most suitable plan.",
            "priority": "fallback"
        })

    # Convert all recommendation values to standard Python types
    for rec in recommendations:
        rec["plan"] = str(rec["plan"]) if rec["plan"] else "No plan available"
        rec["justification"] = str(rec["justification"])
        rec["priority"] = str(rec["priority"])

    return recommendations
