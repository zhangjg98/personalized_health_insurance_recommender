def filter_irrelevant_plans(plans, user_input):
    """
    Filter out plans that conflict with user inputs.

    Parameters:
        plans (list): List of plans to filter.
        user_input (dict): User inputs containing preferences.

    Returns:
        list: Filtered list of plans.
    """
    def violates_constraints(plan, user_input):
        """
        Check if a plan violates constraints based on user inputs.
        """
        description = plan["description"].lower()

        # Check for BMI-related plans
        bmi = user_input.get("bmi", "").lower()
        if "underweight" in description and (not bmi or bmi != "underweight"):
            return True
        if "overweight" in description and (not bmi or bmi != "overweight"):
            return True
        if "obese" in description and (not bmi or bmi != "obese"):
            return True
        if "weight" in description and (not bmi or bmi == "normal"):
            return True

        # Check for ethnicity-related plans
        ethnicity = user_input.get("ethnicity", "").lower()
        if "african american" in description and ethnicity != "black":
            return True
        if "hispanic" in description and ethnicity != "hispanic":
            return True

        # Check for gender-related plans
        gender = user_input.get("gender", "").lower()
        if "women" in description and gender != "female":
            return True
        if "maternity" in description and user_input.get("gender") != "female":
            return True

        # Check for low-income-related plans
        income = user_input.get("income", "").lower()
        if "low income" in description and income not in ["below_30000", ""]:
            return True
        if "high income" in description and user_input.get("income") == "below_30000":
            return True

        # Check for frequent medical care visits
        medical_care_frequency = user_input.get("medical_care_frequency", "").lower()
        if "frequent medical visits" in description and medical_care_frequency != "high":
            return True
        if "low frequency" in description and user_input.get("medical_care_frequency") == "high":
            return True
        if "high frequency" in description and user_input.get("medical_care_frequency") == "low":
            return True

        # Check for smoker-related plans
        if "smoker" in description and user_input.get("smoker") == "no":
            return True

        # Check for chronic condition-related plans
        if "chronic" in description and user_input.get("chronic_condition") == "no":
            return True

        # Check for family size-related plans
        family_size = user_input.get("family_size", "").lower()
        if "family" in description and family_size not in ["2_to_3", "4_plus", ""]:
            return True

        # Check for age-related plans
        age = user_input.get("age", "").lower()
        if "senior" in description and age == "young_adult":
            return True

        # Check for preferred plan type-related plans
        preferred_plan_type = user_input.get("preferred_plan_type", "").lower()
        if preferred_plan_type and preferred_plan_type not in description:
            return True

        return False

    # Filter out plans that violate constraints
    filtered_plans = [plan for plan in plans if not violates_constraints(plan, user_input)]
    return filtered_plans
