# Description: This file contains the propositinoal logic for the insurance recommender system.

# Recommendation function
def recommend_plan(user_input):
    age_group = user_input.get('age', '18-29')
    smoker = user_input.get('smoker', 'no')
    bmi_category = user_input.get('bmi', '')
    income = user_input.get('income', '')
    family_size = user_input.get('family_size', '')
    chronic_condition = user_input.get('chronic_condition', 'no')
    medical_care_frequency = user_input.get('medical_care_frequency', 'Low')

    # Smoker recommendation
    if smoker == "yes":
        return {
            "plan": "Plan Recommendation: Prioritize Plans with High Deductible and Preventive Care for Smokers",
            "justification": "Smokers will have higher health insurance premiums, so a high deductible plan will help mitigate the monthly premium amount. Preventive care includes services that will help with health screenings, which could be important due to risks of lung cancer and other health problems for smokers."
        }
    
    # Weight Recommendations
    if bmi_category == "underweight":
        return {
            "plan": "Plan Recommendation: Prioritize Plans with Specialized Nutritional Support Coverage",
            "justification": "Strongly consider a plan that provides specialized nutritional support if you have issues with being underweight. Nutritional support is a therapy that can help for people who have difficulty getting enough nourishment through eating or drinking."
        }
    elif bmi_category == "normal" and medical_care_frequency == "High":
        return {
            "plan": "Plan Recommendation: Prioritize Plans with Preventive Screening Coverage",
            "justification": "For someone with frequent medical visits but normal BMI, it is best to strongly consider plans with preventive screening coverage. Preventive screening can help detect health problems early, which in turn can also prevent serious illnesses."
        }
    elif bmi_category in ["overweight", "obese"]:
        return {
            "plan": "Plan Recommendation: Look Into Plans that Provide Health and Wellness Programs",
            "justification": "For those struggling with weight issues, plans that can provide health and wellness programs should be strongly considered. Moving towards a healthy lifestyle is critical in ensuring that you do not have health issues further down the line."
        }

    # Age-based recommendations
    if age_group == "young_adult":
        if income == "below_30000":
            if chronic_condition == "yes":
                return {
                "plan": "Plan Recommendation: Strongly Consider Medicaid or Subsidized Marketplace Plan with Chronic Care Management Services",
                "justification": "Young adults with low income and a chronic condition should prioritize Medicaid (if eligible) or subsidized marketplace plans. Medicaid is a government program that provides comprehensive benefits for low-income individuals at little to no cost. If you are not eligible for Medicaid, prioritize a low-cost plan or subsidized plan that still provides chronic care management services."
                }

            return {
                "plan": "Plan Recommendation: Catastrophic Health Plan",
                "justification": "Young adults with low income can benefit from catastrophic policies. These policies cover numerous essential health benefits like other marketplace plans, including preventive services at no cost. They have low monthly premiums and very high deductibles, making them an affordable option for low-income young adults."
            }
        elif bmi_category == "normal" and chronic_condition == "no" and medical_care_frequency == "Low":
            return {
                "plan": "Plan Recommendation: Prioritize Plans with High Deductible and Low Premium",
                "justification": "If you are a young, healthy adult, having a plan with high deductible and low premium can be well suited for you. These plans will keep you insured in case for getting sick or injured, and the low monthly premiums will ensure that you do not have to pay too much to stay insured."
            }

    if age_group == "adult":
        if income == "below_30000":
            return {
                "plan": "Plan Recommendation: Prioritize Affordable Plans that Provide Preventive Care Options",
                "justification": "Adults with low income should prioritize affordability over all else to ensure that they are covered. These plans should still include preventive care options to ensure basic health needs and services."
            }
        if chronic_condition == "yes":
            return {
                "plan": "Plan Recommendation: Strongly Consider Plans that Provide Chronic Care Management Services",
                "justification": "For adults with a chronic condition, chronic care management services can be vital to help you better deal with your condition. These serices include a comprehensive care plan that lists your health problems and goals as well as provide needed medication and urgent care needs."
            }
        if medical_care_frequency == "High":
            return {
                "plan": "Plan Recommendation: Prioritize a Moderate Deductible Plan",
                "justification": "If you have frequent medical care visits, it is important to have a deductible that is at the very least moderate. A moderate deductible ensures a reasonable balance between monthly premiums and out of pocket costs incurred until the deductible amount is hit."
            }
        
    if age_group == "senior":
        if chronic_condition == "yes" or medical_care_frequency == "High":
            return {
                "plan": "Plan Recommendation: Prioritize Low-Deductible Plans",
                "justification": "For seniors with chronic conditions or frequent medical visits, it is best to prioritize a low-deductible plan. These types of plans ensure that you will not have to pay too much out of pocket to reach your deductible amount. This in turn will allow you to have your plan cover frequent medical expenses after reaching the deductible amount."
            }
                
    # Family size adjustments
    if family_size == "4_plus":
        if income == "above_100000":
            return {
                "plan": "Plan Recommendation: Consider Preferred Provider Organization (PPO) Plans with Family Coverage and Comprehensive Benefits",
                "justification": "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. You ensure that your family members will get the care they need while not restricting them "
            }
        return {
            "plan": "Plan Recommendation: Consider Family Coverage with Pediatric and Maternity Benefits",
            "justification": "Pediatric and maternity benefits will help cover medical care for children and mothers (for pregnancy and childbirth) respectively. Making sure that your family gets the benefits they need is important with a family."
        }
    elif family_size == "2_to_3":
        return {
            "plan": "Plan Recommendation: Consider Family Plans with Preventive Care and Moderate Deductibles",
            "justification": "For smaller families, a family health plan that balances affordability with services like preventive care is ideal. Moderate deductibles ensure manageable out-of-pocket costs while providing coverage for routine check-ups and family-specific needs."
        }


    # Generalized Income-based recommendations
    if income == "below_30000":
        return {
            "plan": "Plan Recommendation: Consider Low Cost or Subsidized Coverage",
            "justification": "Low cost plans ensure that you stay covered for basic medical needs. Subsidized coverage is health coverage that come at reduced or low costs for people below certain income thresholds. Both are affordable ways to ensure you still get sufficient medical coverage."
        }
    elif income in ["75000_to_99999", "30000_to_74999"]:
         return {
             "plan": "Plan Recommendation: High Deductible Health Plans (HDHPs) with Health Savings Account (HSA)",
             "justification": "HDHPs have high deductibles and low premiums. A HSA allows you to pay that deductible amount and other medical costs with money set aside in an account where you can contribute and withdraw tax-free. HDHPs and HSAs usually go hand in hand, making them a solid option for people in your income range."
         }
    elif income == "above_100000":
        return {
            "plan": "Plan Recommendation: Preferred Provider Organization (PPO) Plans that Ensure Flexibility",
            "justification": "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. You can get the care that you want at any provider you would like, though costs will be lower if you do choose to go with an in-network provider."
        }
        
    # General Chronic Condition and Medical Care Frequency Recommendations
    if chronic_condition == "yes" and medical_care_frequency == "Low":
         return {
             "plan": "Plan Recommendation: Medication Therapy Management Program for Chronic Conditions",
             "justification": "If you do not need to see a doctor frequently but have a chronic condition, a medication therapy management program might be suited for you. This plan makes sure that you take your medications correctly and safely, and basic services related to your condition come at no cost."
         }
    elif chronic_condition == "yes":
         return {
             "plan": "Plan Recommendation: Chronic Care Coverage",
             "justification": "Chronic care coverage is a Medicare program that helps with chronic conditions. Services include a comprehensive care plan that lists your health problems and goals as well as provide needed medication and urgent care needs."
         }
    elif medical_care_frequency == "High":
         return {
                "plan": "Plan Recommendation: Low-Deductible Plan",
                "justification": "If you need frequent medical care, having a plan with a low deductible would be most preferable for you. Low deductible plans ensure that your out of pocket costs will be low, as you only need to hit a low threshold before your insurance covers your medical costs."
            }
    elif medical_care_frequency == "Low":
         return {
                "plan": "Plan Recommendation: High Deductible, Low-Premium Plan",
                "justification": "If you are not in need of frequent medical care, having a plan with high deductible and low premium can be well suited for you. These plans will keep you insured in case for getting sick or injured, and the low monthly premiums will ensure that you do not have to pay too much to stay insured."
            }
    elif medical_care_frequency == "Moderate":
        return {
            "plan": "Plan Recommendation: Moderate Deductible Plan",
            "justification": "For someone who needs occasional medical care, a moderate deductible plan offers a good balance. It ensures manageable out-of-pocket costs while keeping premiums lower than low-deductible plans."
        }
    
    # General fallback
    return {
        "plan": "Plan Recommendation: Contact a representative for personalized advice",
        "justification": "Based on the information provided, a representative is more likely to help you identify the most suitable plan for you."
    }    
