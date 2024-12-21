# Personalized Health Insurance Recommender

This program is a simple health insurance plan recommendation system built using Flask and propositional logic. It provides plan suggestions based on user input for fields such as age, income, BMI, smoker status, and more.

## How the Code Works

The application is built on the following components:
- **Flask Application**: Handles the user input and displays recommendations via web routes.
- **Propositional Rule Logic**: Propositional logic rules evaluate user input to generate plan recommendations.
- **Templates**: HTML files (`index.html` and `result.html`) provide a user-friendly interface.

## Basis for Rule Logic and Justifications

The rules are designed around considerations like:
- **Age**: Specific recommendations for different age groups (e.g., catastrophic plans for young adults).
- **Income**: Recommendations balance affordability and coverage.
- **Health Status**: Chronic conditions and BMI are factored into specialized plan suggestions.

Each recommendation includes a justification to explain its reasoning, ensuring users understand the logic.

## How to Run the Code

Make sure that you have downloaded the file `insurance_recommender.py` along with the templates folder that contains the `index.html` and `result.html` files.

Running this program requires Python and Flask. If you do not have Flask downloaded on your system, you should download the `requirements.txt` file and use the command `pip install -r requirements.txt` to download the necessary libraries to run this program.

Use the command `python3 insurance_recommender.py` to compile the code. The python command might vary depending on what version of Python is installed. You should then open a web browser and navigate to `http://127.0.0.1:5000`. Once that page has loaded, you should be able to use the Personalized Health Insurance Recommender.

### Future Improvements

- Integrate Machine Learning techniques to handle more complex cases and ensure more scalable recommendations.
- Utilization of datasets to apply real-life data into the system and help with more nuanced, niche cases.