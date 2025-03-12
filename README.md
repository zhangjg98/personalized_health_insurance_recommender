# Personalized Health Insurance Recommender

This program is a health insurance plan recommendation system built using Flask and propositional logic. It provides plan suggestions based on user input for fields such as age, income, BMI, smoker status, and more. It is currently in the process of being re-purposed to incorporate machine learning to handle more complex cases as well as incorporating React on the front-end.

## How the Code Works

The application is built on the following components:
- **Flask Application**: Handles the user input and displays recommendations via web routes.
- **Propositional Rule Logic**: Propositional logic rules evaluate user input to generate plan recommendations.

## Basis for Rule Logic and Justifications

The rules are designed around considerations like:
- **Age**: Specific recommendations for different age groups (e.g., catastrophic plans for young adults).
- **Income**: Recommendations balance affordability and coverage.
- **Health Status**: Chronic conditions and BMI are factored into specialized plan suggestions.

Each recommendation includes a justification to explain its reasoning, ensuring users understand the logic.

## How to Run the Code

Make sure that you have downloaded all of the files in this repository.

Running this program requires Python, Flask, and React. If you do not have Flask downloaded on your system, you should download the `requirements.txt` file and use the command `pip install -r requirements.txt` to download the necessary libraries to run this program.

Use the command `chmod +x start.sh` to be able to run the script. Then use the command `./start.sh`, and the program should automatically open on your Internet browser. The python command might vary depending on what version of Python is installed.

### Future Improvements

- Integrate Machine Learning techniques to handle more complex cases and ensure more scalable recommendations.
- Utilization of datasets to apply real-life data into the system and help with more nuanced, niche cases.