# Dictionary to store plan names, descriptions, and justifications
PLANS = {
    "high_deductible_smokers": {
        "name": "Plan Recommendation: Prioritize Plans with High Deductible and Preventive Care for Smokers",
        "description": (
            "Smokers will have higher health insurance premiums, so a high deductible plan will help mitigate "
            "the monthly premium amount. Preventive care includes services that will help with health screenings, "
            "which could be important due to risks of lung cancer and other health problems for smokers."
        ),
    },
    "nutritional_support_underweight": {
        "name": "Plan Recommendation: Prioritize Plans with Specialized Nutritional Support Coverage",
        "description": (
            "Strongly consider a plan that provides specialized nutritional support if you have issues with being underweight. "
            "Nutritional support is a therapy that can help for people who have difficulty getting enough nourishment through eating or drinking."
        ),
    },
    "wellness_programs_overweight": {
        "name": "Plan Recommendation: Look Into Plans that Provide Health and Wellness Programs",
        "description": (
            "For those struggling with weight issues, plans that can provide health and wellness programs should be strongly considered. "
            "Moving towards a healthy lifestyle is critical in ensuring that you do not have health issues further down the line."
        ),
    },
    "chronic_care_low_frequency": {
        "name": "Plan Recommendation: Medication Therapy Management Program for Chronic Conditions",
        "description": (
            "If you do not need to see a doctor frequently but have a chronic condition, a medication therapy management program might be suited for you. "
            "This plan makes sure that you take your medications correctly and safely, and basic services related to your condition come at no cost."
        ),
    },
    "chronic_care_high_frequency": {
        "name": "Plan Recommendation: Chronic Care Coverage",
        "description": (
            "Chronic care coverage is a Medicare program that helps with chronic conditions. Services include a comprehensive care plan that lists your health problems and goals "
            "as well as provide needed medication and urgent care needs."
        ),
    },
    "enhanced_womens_health": {
        "name": "Plan Recommendation: Consider Plans with Enhanced Women's Health Coverage",
        "description": (
            "The predicted percentage of female beneficiaries is high. Consider plans that include robust maternity and women’s health services."
        ),
    },
    "preventive_care_chronic_conditions_african_americans": {
        "name": "Plan Recommendation: Consider Plans with Preventive Care for Chronic Conditions",
        "description": (
            "A higher predicted percentage of African American beneficiaries may indicate elevated risk for chronic conditions such as hypertension and diabetes. "
            "Plans with strong preventive care and chronic disease management are recommended."
        ),
    },
    "culturally_relevant_healthcare_hispanics": {
        "name": "Plan Recommendation: Consider Plans Offering Culturally Relevant Healthcare Services",
        "description": (
            "A significant predicted Hispanic population suggests that plans offering bilingual support and culturally tailored services could be beneficial."
        ),
    },
    "medicaid_or_subsidized_young_adults": {
        "name": "Plan Recommendation: Strongly Consider Medicaid or Subsidized Marketplace Plan with Chronic Care Management Services",
        "description": (
            "Young adults with low income and a chronic condition should prioritize Medicaid (if eligible) or subsidized marketplace plans. "
            "Medicaid is a government program that provides comprehensive benefits for low-income individuals at little to no cost. "
            "If you are not eligible for Medicaid, prioritize a low-cost plan or subsidized plan that still provides chronic care management services."
        ),
    },
    "catastrophic_young_adults": {
        "name": "Plan Recommendation: Catastrophic Health Plan",
        "description": (
            "Young adults with low income can benefit from catastrophic policies. These policies cover numerous essential health benefits like other marketplace plans, "
            "including preventive services at no cost. They have low monthly premiums and very high deductibles, making them an affordable option for low-income young adults."
        ),
    },
    "high_deductible_low_premium_young_adults": {
        "name": "Plan Recommendation: Prioritize Plans with High Deductible and Low Premium",
        "description": (
            "If you are a young, healthy adult, having a plan with high deductible and low premium can be well suited for you. "
            "These plans will keep you insured in case for getting sick or injured, and the low monthly premiums will ensure that you do not have to pay too much to stay insured."
        ),
    },
    "affordable_preventive_adults": {
        "name": "Plan Recommendation: Prioritize Affordable Plans that Provide Preventive Care Options",
        "description": (
            "Adults with low income should prioritize affordability over all else to ensure that they are covered. "
            "These plans should still include preventive care options to ensure basic health needs and services."
        ),
    },
    "moderate_deductible_adults": {
        "name": "Plan Recommendation: Prioritize a Moderate Deductible Plan",
        "description": (
            "If you have frequent medical care visits, it is important to have a deductible that is at the very least moderate. "
            "A moderate deductible ensures a reasonable balance between monthly premiums and out of pocket costs incurred until the deductible amount is hit."
        ),
    },
    "low_deductible_seniors": {
        "name": "Plan Recommendation: Prioritize Low-Deductible Plans",
        "description": (
            "For seniors with chronic conditions or frequent medical visits, it is best to prioritize a low-deductible plan. "
            "These types of plans ensure that you will not have to pay too much out of pocket to reach your deductible amount. "
            "This in turn will allow you to have your plan cover frequent medical expenses after reaching the deductible amount."
        ),
    },
    "ppo_family_coverage_high_income": {
        "name": "Plan Recommendation: Consider Preferred Provider Organization (PPO) Plans with Family Coverage and Comprehensive Benefits",
        "description": (
            "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. "
            "You ensure that your family members will get the care they need while not restricting them."
        ),
    },
    "family_coverage_pediatric_maternity": {
        "name": "Plan Recommendation: Consider Family Coverage with Pediatric and Maternity Benefits",
        "description": (
            "Pediatric and maternity benefits will help cover medical care for children and mothers (for pregnancy and childbirth) respectively. "
            "Making sure that your family gets the benefits they need is important with a family."
        ),
    },
    "family_plans_preventive_moderate_deductibles": {
        "name": "Plan Recommendation: Consider Family Plans with Preventive Care and Moderate Deductibles",
        "description": (
            "For smaller families, a family health plan that balances affordability with services like preventive care is ideal. "
            "Moderate deductibles ensure manageable out-of-pocket costs while providing coverage for routine check-ups and family-specific needs."
        ),
    },
    "maternity_preventive_care_females": {
        "name": "Plan Recommendation: Prioritize Plans with Maternity and Preventive Care Benefits",
        "description": (
            "For females, plans with maternity benefits and preventive care services are highly recommended to ensure comprehensive coverage for women's health needs."
        ),
    },
    "low_cost_subsidized_coverage": {
        "name": "Plan Recommendation: Consider Low Cost or Subsidized Coverage",
        "description": (
            "Low cost plans ensure that you stay covered for basic medical needs. Subsidized coverage is health coverage that comes at reduced or low costs for people below certain income thresholds. "
            "Both are affordable ways to ensure you still get sufficient medical coverage."
        ),
    },
    "hdhp_with_hsa": {
        "name": "Plan Recommendation: High Deductible Health Plans (HDHPs) with Health Savings Account (HSA)",
        "description": (
            "HDHPs have high deductibles and low premiums. A HSA allows you to pay that deductible amount and other medical costs with money set aside in an account where you can contribute and withdraw tax-free. "
            "HDHPs and HSAs usually go hand in hand, making them a solid option for people in your income range."
        ),
    },
    "ppo_flexibility_high_income": {
        "name": "Plan Recommendation: Preferred Provider Organization (PPO) Plans that Ensure Flexibility",
        "description": (
            "PPO plans give great flexibility that allow you to see doctors in and outside of the plan network. "
            "You can get the care that you want at any provider you would like, though costs will be lower if you do choose to go with an in-network provider."
        ),
    },
    "hmo_plan": {
        "name": "Plan Recommendation: Health Maintenance Organization (HMO) Plan",
        "description": (
            "HMO plans typically offer lower premiums and coordinated care within a network, making them ideal if you prefer managed care."
        ),
    },
    "ppo_plan": {
        "name": "Plan Recommendation: Preferred Provider Organization (PPO) Plan",
        "description": (
            "PPO plans provide more flexibility in choosing providers and generally offer comprehensive benefits."
        ),
    },
    "epo_plan": {
        "name": "Plan Recommendation: Exclusive Provider Organization (EPO) Plan",
        "description": (
            "EPO plans require using a network of providers but often have lower premiums than PPOs. They’re a good option if you don’t need out-of-network coverage."
        ),
    },
    "pos_plan": {
        "name": "Plan Recommendation: Point of Service (POS) Plan",
        "description": (
            "POS plans combine features of HMOs and PPOs, offering more flexibility than HMOs while keeping costs relatively low."
        ),
    },
    # Priority-based recommendations
    "low_premiums": {
        "name": "Plan Recommendation: High Deductible, Low Premium Plan",
        "description": (
            "Since you prioritize low premiums, a high deductible plan is recommended. These plans have lower monthly costs, making them more affordable."
        ),
    },
    "comprehensive_coverage": {
        "name": "Plan Recommendation: Comprehensive PPO Plan",
        "description": (
            "Since you value comprehensive coverage, a PPO plan is recommended. These plans provide flexibility and access to a wide range of healthcare providers."
        ),
    },
    "preventive_care": {
        "name": "Plan Recommendation: Preventive Care-Focused Plan",
        "description": (
            "Since you prioritize preventive care, this plan includes extensive preventive services to help you maintain good health."
        ),
    },
    "low_deductibles": {
        "name": "Plan Recommendation: Low Deductible Plan",
        "description": (
            "Since you prefer low deductibles, this plan minimizes out-of-pocket costs before insurance coverage begins."
        ),
    },
}
