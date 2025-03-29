from propositional_logic import recommend_plan

def test_recommendation_logic():
    # Simulate user input
    user_input = {
        "age": "young_adult",
        "smoker": "yes",
        "bmi": "normal",
        "income": "below_30000",
        "family_size": "1",
        "chronic_condition": "no",
        "medical_care_frequency": "Low",
        "preferred_plan_type": "HMO",
        "gender": "male",
        "ethnicity": "hispanic",
        "user_id": 1
    }

    # Call the recommendation function
    recommendations = recommend_plan(user_input)
    print("Recommendations:")
    for rec in recommendations:
        print(f"Plan: {rec['plan']}, Justification: {rec['justification']}, Priority: {rec['priority']}")

if __name__ == "__main__":
    test_recommendation_logic()
