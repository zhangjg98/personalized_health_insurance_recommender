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
        return "Plan: High Deductible with Preventive Care for Smokers"
    
    # Weight Recommendations
    if bmi_category == "underweight":
        return "Plan: Specialized Nutritional Support Coverage"
    elif bmi_category == "normal" and medical_care_frequency == "High":
         return "Plan: Preventive Screening Coverage Plan"
    elif bmi_category in ["overweight", "obese"]:
         return "Plan: Wellness and Fitness Support Plan"

    # Age-based recommendations
    if age_group == "young_adult":
        if income == "below_30000":
            return "Plan: Affordable Catastrophic Plan for Young Adults"
        elif bmi_category == "normal" and chronic_condition == "no" and medical_care_frequency == "Low":
            return "Plan: High Deductible, Low Premium for Young and Healthy"

    if age_group == "adult":
        if income == "below_30000":
            return "Plan: Affordable Coverage with Preventive Wellness Options"
        if chronic_condition == "yes":
            return "Plan: Chronic Condition Management for Working Adults"
        if medical_care_frequency == "High":
            return "Plan: Moderate Deductible Plan with Specialist Access"

    if age_group == "senior":
        if chronic_condition == "yes":
            return "Plan: Comprehensive Low-Deductible with Specialist Access"
        elif medical_care_frequency == "High":
            return "Plan: Comprehensive Plan with Moderate Deductible"
        
    # Family size adjustments
    if family_size == "4_plus":
        if income == "above_100000":
            return "Plan: Premium Family Coverage with Comprehensive Benefits"
        return "Plan: Family Coverage with Pediatric and Maternity Benefits"

    # Generalized Income-based recommendations
    if income == "below_30000":
        return "Plan: Low Cost or Subsidized Options"
    elif income in ["75000_to_99999", "30000_to_74999"]:
         return "Plan: High Deductible with HSA"
    elif income == "above_100000":
        return "Plan: Premium Benefits"
        
    # General Chronic Condition and Medical Care Frequency Recommendations
    if chronic_condition == "yes" and medical_care_frequency == "Low":
         return "Plan: Medication Management Plan for Chronic Conditions"
    elif chronic_condition == "yes":
         return "Plan: Chronic Care Coverage"
    elif medical_care_frequency == "High":
         return "Plan: Low-Deductible Plan"
    elif medical_care_frequency == "Low":
        return "Plan: High-Deductible, Low Premium Plan"

    # General fallback
    return "Plan: Contact a representative for personalized advice"
    
    
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
