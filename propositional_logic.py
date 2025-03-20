import pandas as pd
import os
# Description: This file contains the propositional logic for the insurance recommender system.

# Recommendation function
def recommend_plan(user_input, priority=""):
    age_group = user_input.get('age', '18-29')
    smoker = user_input.get('smoker', 'no')
    bmi_category = user_input.get('bmi', '')
    income = user_input.get('income', '')
    family_size = user_input.get('family_size', '')
    chronic_condition = user_input.get('chronic_condition', 'no')
    medical_care_frequency = user_input.get('medical_care_frequency', 'Low')

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
        if medical_care_frequency == "High":
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

    # Fallback: Recommend the highest-rated plan if no recommendations exist
    if not recommendations:
        if os.path.exists("plan_feedback_analysis.csv"):
            plan_feedback = pd.read_csv("plan_feedback_analysis.csv")
            highest_rated_plan = plan_feedback.sort_values(by="rating", ascending=False).iloc[0]
            recommendations.append({
                "plan": highest_rated_plan["plan"],
                "justification": "This plan is highly rated by other users and may suit your needs.",
                "priority": "fallback"
            })
        else:
            recommendations.append({
                "plan": None,  # No plan to recommend
                "justification": "Please select more criteria so that we can produce a meaningful recommendation.",
                "priority": "insufficient_criteria"
            })

    return recommendations
