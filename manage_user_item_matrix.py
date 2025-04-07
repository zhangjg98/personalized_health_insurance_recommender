import os
import pandas as pd
from flask_backend import app, db, Interaction, Item
import argparse
from multiprocessing import resource_tracker
import platform
import warnings

warnings.filterwarnings("ignore", message="resource_tracker: There appear to be .* leaked semaphore objects")

def manage_user_item_matrix(mode="generate"):
    """
    Manage the user-item matrix by either generating it from scratch or updating it incrementally.

    Parameters:
        mode (str): "generate" to create the matrix from scratch, "update" to update it incrementally.
    """
    with app.app_context():
        # Step 1: Query interactions and validate item_id values
        interactions = Interaction.query.all()
        data = [(i.user_id, i.item_id, i.rating) for i in interactions]
        if not data:
            print("No interactions found in the database.")
            return

        df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating'])
        print("Interactions DataFrame:")
        print(df.head())  # Debugging log

        # Validate item_id values against the items table
        valid_item_ids = set(item.id for item in Item.query.all())
        df = df[df['item_id'].isin(valid_item_ids)]
        print(f"Filtered interactions to include only valid item_id values. Remaining rows: {len(df)}")

        # Aggregate duplicate entries by taking the most recent rating (if timestamp exists) or the average
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').groupby(['user_id', 'item_id'], as_index=False).last()
        else:
            df = df.groupby(['user_id', 'item_id'], as_index=False).mean()

        # Include only items with interactions
        active_item_ids = df['item_id'].unique()
        print(f"Active item IDs with interactions: {active_item_ids}")

        # Log items in the database but without interactions
        all_item_ids = set(item.id for item in Item.query.all())
        missing_interactions = all_item_ids - set(active_item_ids)
        if missing_interactions:
            print(f"Warning: The following item_ids are in the database but have no interactions: {missing_interactions}")

        if mode == "generate":
            # Step 2: Generate the user-item matrix from scratch
            print("Generating user-item matrix from scratch...")
            user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').reindex(columns=active_item_ids, fill_value=0)
        elif mode == "update":
            # Step 3: Update the existing user-item matrix
            print("Updating user-item matrix...")
            try:
                # Load the existing matrix
                user_item_matrix = pd.read_csv('user_item_matrix.csv', index_col=0)
                print("Existing user-item matrix loaded:")
                print(user_item_matrix.head())  # Debugging log

                # Add missing columns for new items
                for item_id in active_item_ids:
                    if item_id not in user_item_matrix.columns:
                        user_item_matrix[item_id] = 0

                # Update the matrix with new interactions
                for _, row in df.iterrows():
                    user_id, item_id, rating = int(row['user_id']), int(row['item_id']), row['rating']
                    if user_id not in user_item_matrix.index:
                        user_item_matrix.loc[user_id] = 0  # Add new user row
                    user_item_matrix.loc[user_id, item_id] = rating  # Update the rating
            except FileNotFoundError:
                print("No existing user-item matrix found. Generating from scratch...")
                user_item_matrix = df.pivot(index='user_id', columns='item_id', values='rating').reindex(columns=active_item_ids, fill_value=0)
        else:
            raise ValueError("Invalid mode. Use 'generate' or 'update'.")

        # Step 4: Save the user-item matrix to a CSV file
        if not user_item_matrix.empty:
            print("Final user-item matrix:")
            print(user_item_matrix.head())  # Debugging log
            user_item_matrix.to_csv('user_item_matrix.csv')
            print(f"User-item matrix saved to user_item_matrix.csv (mode: {mode}).")
        else:
            print("User-item matrix is empty. No file was saved.")

        # Explicitly clean up semaphore objects
        print("Cleaning up leaked semaphore objects...")
        try:
            shared_memory_path = "/dev/shm" if platform.system() == "Linux" else "/private/var/run"
            if os.path.exists(shared_memory_path):
                for semaphore in os.listdir(shared_memory_path):
                    if semaphore.startswith("sem."):
                        try:
                            resource_tracker.unregister(f"{shared_memory_path}/{semaphore}", "semaphore")
                        except KeyError:
                            pass  # Suppress KeyError if semaphore does not exist
        except Exception as e:
            print(f"Error during semaphore cleanup: {e}")
        print("Semaphore cleanup completed.")

def encode_user_inputs(user_inputs):
    """Convert user inputs into an encoded vector."""
    # Example encoding logic (one-hot encoding or similar)
    encoded_vector = []
    age_mapping = {"18-29": [1, 0, 0], "30-59": [0, 1, 0], "60+": [0, 0, 1]}
    encoded_vector.extend(age_mapping.get(user_inputs.get('age', ''), [0, 0, 0]))
    # Add similar mappings for other inputs (e.g., smoker, income, etc.)
    return encoded_vector

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
