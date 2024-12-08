from flask import Flask, render_template, request

app = Flask(__name__)

# Recommendation function
def recommend_plan(user_input):
    age_range = user_input.get('age', '18-29')
    smoker = user_input.get('smoker', 'no').lower() == 'yes'
    bmi_range = user_input.get('bmi', 'normal')
    income = float(user_input.get('income', 0))
    family_size = int(user_input.get('family_size', 1))
    chronic_condition = user_input.get('chronic_condition', 'no').lower() == 'yes'
    preventive_care_needed = user_input.get('preventive_care_needed', 'no').lower() == 'yes'
    frequent_medical_visits = user_input.get('frequent_medical_visits', 'no').lower() == 'yes'

    # Categorizing the age ranges
    if age_range == "18-29":
        age_group = "young_adult"
    elif age_range == "30-59":
        age_group = "adult"
    else:
        age_group = "senior"
    
    bmi_category = bmi_range.lower()  # Use BMI range directly from dropdown to assign to appropriate category

    if age_group == "young_adult" and smoker:
        return "Plan: High Deductible with Preventive Care for Smokers"
    elif family_size > 3 and income < 30000:
        return "Plan: Subsidized Family Coverage"
    elif age_range == "senior" and chronic_condition:
        return "Plan: Comprehensive Low-Deductible with Specialist Access"
    elif age_range == "young_adult":
        return "Plan: Low Premium, High Deductible"
    elif age_range == "senior":
        return "Plan: Comprehensive Coverage"
    elif smoker:
        return "Plan: Preventive Care for Smokers"
    elif bmi_category == "obese":
        return "Plan: Wellness Program with Chronic Care Management"
    elif income < 30000:
        return "Plan: Low Cost or Subsidized Options"
    elif income > 100000:
        return "Plan: Premium Benefits"
    elif family_size > 3:
        return "Plan: Family Coverage with Pediatric and Maternity Benefits"
    elif family_size == 1:
        return "Plan: Individual Coverage"
    elif chronic_condition:
        return "Plan: Chronic Care Coverage"
    elif preventive_care_needed:
        return "Plan: Preventive Care Coverage"
    elif frequent_medical_visits:
        return "Plan: Low-Deductible Plan"
    if age_group == "young_adult" and bmi_category == "normal" and not smoker:
        return "Plan: High Deductible, Low Premium for Young and Healthy"
    elif income > 75000:
        return "Plan: High Deductible with HSA"
    else:
        return "Plan: Basic Coverage"

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
