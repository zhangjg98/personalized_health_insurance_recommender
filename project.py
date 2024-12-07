def recommend_plan(user_input):
    # Extract user information
    age = user_input.get('age', 0)
    smoker = user_input.get('smoker', False)
    bmi = user_input.get('bmi', 0)
    income = user_input.get('income', 0)
    family_size = user_input.get('family_size', 1)
    chronic_condition = user_input.get('chronic_condition', False)
    preventive_care_needed = user_input.get('preventive_care_needed', False)
    frequent_medical_visits = user_input.get('frequent_medical_visits', False)
    healthy = user_input.get('healthy', True)  # Defaults to True if not specified

    # Plan recommendations based on rules
    if age < 30 and smoker:
        return "Plan: High Deductible with Preventive Care for Smokers"
    elif family_size > 3 and income < 30000:
        return "Plan: Subsidized Family Coverage"
    elif age >= 60 and chronic_condition:
        return "Plan: Comprehensive Low-Deductible with Specialist Access"
    elif age < 30:
        return "Plan: Low Premium, High Deductible"
    elif age >= 60:
        return "Plan: Comprehensive Coverage"
    elif smoker:
        return "Plan: Preventive Care for Smokers"
    elif bmi > 30:
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
    elif age < 35 and healthy:
        return "Plan: High Deductible, Low Premium"
    elif income > 75000:
        return "Plan: High Deductible with HSA"
    else:
        return "Plan: Basic Coverage"

# Example usage
user = {
    "age": 29,
    "smoker": True,
    "bmi": 28,
    "income": 40000,
    "family_size": 1,
    "chronic_condition": False,
    "preventive_care_needed": True,
    "frequent_medical_visits": False,
    "healthy": False
}

plan = recommend_plan(user)
print(plan)  # Outputs the recommended plan
