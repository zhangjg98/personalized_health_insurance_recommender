import pandas as pd
from flask_backend import app, db, Interaction

def analyze_feedback():
    with app.app_context():
        interactions = Interaction.query.all()
        data = [(i.item_id, i.rating) for i in interactions]
        df = pd.DataFrame(data, columns=['plan', 'rating'])

        # Calculate average rating for each plan
        plan_feedback = df.groupby('plan')['rating'].mean().reset_index()
        plan_feedback = plan_feedback.sort_values(by='rating', ascending=False)

        # Save feedback analysis to a CSV file
        plan_feedback.to_csv('plan_feedback_analysis.csv', index=False)
        print("Feedback analysis saved to plan_feedback_analysis.csv")

if __name__ == "__main__":
    analyze_feedback()
