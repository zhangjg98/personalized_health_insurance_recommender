import pandas as pd
from flask_backend import app, db, Interaction, User

def analyze_feedback():
    with app.app_context():
        # Ensure the database engine is properly initialized
        if db.session.bind is None:
            db.engine.dispose()  # Dispose of any existing connections
            db.session.bind = db.engine  # Rebind the session to the engine

        interactions = Interaction.query.all()
        data = [(i.item_id, i.rating, i.user_id, i.user_inputs) for i in interactions]
        df = pd.DataFrame(data, columns=['plan', 'rating', 'user_id', 'user_inputs'])

        # Use SQLAlchemy engine to read the User table
        sql_query = str(User.query.statement)  # Convert the SQLAlchemy statement to a string
        users = pd.read_sql(sql_query, con=db.session.bind)

        # Join with user preferences to include preferred plan type
        df = df.merge(users[['id', 'preferred_plan_type']], left_on='user_id', right_on='id', how='left')

        # Calculate average rating for each plan and preferred plan type
        plan_feedback = df.groupby(['plan', 'preferred_plan_type'])['rating'].mean().reset_index()
        plan_feedback = plan_feedback.sort_values(by='rating', ascending=False)

        # Save feedback analysis to a CSV file
        plan_feedback.to_csv('plan_feedback_analysis.csv', index=False)

        # Save user inputs for further analysis
        user_inputs_df = df[['plan', 'user_inputs']]
        user_inputs_df.to_csv('user_inputs_analysis.csv', index=False)

        print("Feedback analysis saved to plan_feedback_analysis.csv")
        print("User inputs analysis saved to user_inputs_analysis.csv")

if __name__ == "__main__":
    analyze_feedback()
