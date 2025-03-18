import pandas as pd
from flask_backend import app, db, Interaction
import argparse

def manage_user_item_matrix(mode="update"):
    """
    Manage the user-item matrix by either generating it from scratch or updating it incrementally.

    Parameters:
        mode (str): "generate" to create the matrix from scratch, "update" to update it incrementally.
    """
    with app.app_context():
        interactions = Interaction.query.all()
        data = [(i.user_id, i.item_id, i.rating) for i in interactions]
        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])

        # Aggregate duplicate entries by taking the average rating
        df = df.groupby(['user_id', 'item_id'], as_index=False).mean()

        if mode == "generate":
            # Generate the user-item matrix from scratch
            print("Generating user-item matrix from scratch...")
            user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        elif mode == "update":
            # Update the existing user-item matrix
            print("Updating user-item matrix...")
            try:
                # Load the existing matrix
                user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)
                # Update the matrix with new interactions
                for _, row in df.iterrows():
                    user_id, item_id, rating = int(row['user_id']), int(row['item_id']), row['rating']
                    if user_id not in user_item_matrix.index:
                        user_item_matrix.loc[user_id] = 0  # Add new user row
                    if item_id not in user_item_matrix.columns:
                        user_item_matrix[item_id] = 0  # Add new item column
                    user_item_matrix.loc[user_id, item_id] = rating  # Update the rating
            except FileNotFoundError:
                print("No existing user-item matrix found. Generating from scratch...")
                user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
        else:
            raise ValueError("Invalid mode. Use 'generate' or 'update'.")

        # Save the user-item matrix to a CSV file
        user_item_matrix.to_csv('user_item_matrix.csv')
        print(f"User-item matrix saved to user_item_matrix.csv (mode: {mode}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage the user-item matrix.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["generate", "update"],
        default="update",
        help="Mode to run the script: 'generate' to create the matrix from scratch, 'update' to update it incrementally."
    )
    args = parser.parse_args()
    manage_user_item_matrix(mode=args.mode)
