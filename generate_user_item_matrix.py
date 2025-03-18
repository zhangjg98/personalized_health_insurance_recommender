import pandas as pd
from flask_backend import app, db, Interaction

def generate_user_item_matrix():
    with app.app_context():  # Set up the Flask application context
        interactions = Interaction.query.all()
        data = [(i.user_id, i.item_id, i.rating) for i in interactions]
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])

        # Aggregate duplicate entries by taking the average rating
        df = df.groupby(['user_id', 'item_id'], as_index=False).mean()

        # Pivot the table to create the user-item matrix
        user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

        # Save the user-item matrix to a CSV file
        user_item_matrix.to_csv('user_item_matrix.csv')
        print("User-item matrix saved to user_item_matrix.csv")

if __name__ == "__main__":
    generate_user_item_matrix()
