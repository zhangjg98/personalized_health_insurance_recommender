from flask import Flask, render_template, request

app = Flask(__name__)

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
            return "Plan Recommendation: Consider Preferred Provider Organization (PPO) Plans with Family Coverage and Comprehensive Benefits"
        return "Plan Recommendation: Consider Family Coverage with Pediatric and Maternity Benefits"

    # Generalized Income-based recommendations
    if income == "below_30000":
        return "Plan Recommendation: Low Cost or Subsidized Options"
    elif income in ["75000_to_99999", "30000_to_74999"]:
         return "Plan Recommendation: High Deductible Health Plans (HDHPs) with Health Savings Account (HSA)"
    elif income == "above_100000":
        return "Plan Recommendation: Preferred Provider Organization (PPO) Plans that Ensure Flexibility"
        
    # General Chronic Condition and Medical Care Frequency Recommendations
    if chronic_condition == "yes" and medical_care_frequency == "Low":
         return "Plan Recommendation: Medication Management Plan for Chronic Conditions"
    elif chronic_condition == "yes":
         return "Plan Recommendation: Chronic Care Coverage"
    elif medical_care_frequency == "High":
         return "Plan Recommendation: Low-Deductible Plan"
    elif medical_care_frequency == "Low":
        return "Plan Recommendation: High-Deductible, Low Premium Plan"

    # General fallback
    return {
        "plan": "Plan Recommendation: Contact a representative for personalized advice",
        "justification": "Based on the information provided, a representative is more likely to help you identify the most suitable plan for you."
    }    
    
# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.form
    recommendation = recommend_plan(user_input)
    return render_template('result.html', recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)
