from flask import Flask, render_template, request

app = Flask(__name__)

# Recommendation function
def recommend_plan(user_input):
    age_group = user_input.get('age', '18-29')
    smoker = user_input.get('smoker', 'no')
    bmi_category = user_input.get('bmi', 'normal')
    income = user_input.get('income', 'below_30000')
    family_size = user_input.get('family_size', '1')
    chronic_condition = user_input.get('chronic_condition', 'no')
    medical_care_frequency = user_input.get('medical_care_frequency', 'Low')
    
    if age_group == "young_adult" and smoker == "yes":
        return "Plan: High Deductible with Preventive Care for Smokers"
    elif age_group == "young_adult" and income == "below_30000":
        return "Plan: Affordable Catastrophic Plan for Young Adults"
    elif family_size == "4_plus" and income == "below_30000":
        return "Plan: Subsidized Family Coverage"
    elif age_group == "senior" and chronic_condition == "yes":
        return "Plan: Comprehensive Low-Deductible with Specialist Access"
    elif age_group == "senior" and medical_care_frequency == "High" and chronic_condition == "no":
        return "Plan: Comprehensive Plan with Moderate Deductible"
    elif bmi_category == "underweight":
        return "Plan: Specialized Nutritional Support Coverage"
    elif age_group == "adult" and bmi_category in ["overweight", "obese"] and smoker == "yes":
        return "Plan: Wellness and Preventive Care Plan for Adults"
    elif bmi_category == "normal" and medical_care_frequency == "High":
        return "Plan: Preventive Screening Coverage Plan"
    elif bmi_category == "obese" and smoker == "no":
        return "Plan: Wellness and Fitness Support Plan"
    elif smoker == "yes":
        return "Plan: Preventive Care for Smokers"
    elif bmi_category == "obese":
        return "Plan: Wellness Program with Chronic Care Management"
    elif income == "below_30000":
        return "Plan: Low Cost or Subsidized Options"
    elif income == "above_100000":
        return "Plan: Premium Benefits"
    elif family_size == "4_plus":
        return "Plan: Family Coverage with Pediatric and Maternity Benefits"
    elif family_size == "4_plus" and income == "above_100000":
        return "Plan: Premium Family Coverage with Comprehensive Benefits"
    elif family_size == 1:
        return "Plan: Individual Coverage"
    elif chronic_condition == "yes" and medical_care_frequency == "Low":
        return "Plan: Medication Management Plan for Chronic Conditions"
    elif chronic_condition == "yes":
        return "Plan: Chronic Care Coverage"
    elif medical_care_frequency == "yes":
        return "Plan: Low-Deductible Plan"
    elif age_group == "young_adult" and bmi_category == "normal" and smoker == "no":
        return "Plan: High Deductible, Low Premium for Young and Healthy"
    elif income in ["75000_to_99999", "30000_to_74999"]:
        return "Plan: High Deductible with HSA"
    elif age_group == "young_adult":
        return "Plan: Low Premium, High Deductible"
    elif age_group == "senior":
        return "Plan: Comprehensive Coverage"
    else:
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
