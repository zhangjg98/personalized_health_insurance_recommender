import pandas as pd
from database import db, Interaction

def rebuild_user_item_matrix():
    """
    Rebuild the user-item matrix dynamically from the Interactions table.
    """
    print("Rebuilding user-item matrix from Interactions table...")  # Debugging log
    interactions = Interaction.query.all()
    if not interactions:
        print("No interactions found in the database. Returning an empty matrix.")  # Debugging log
        return pd.DataFrame()

    # Create a DataFrame from interactions
    data = [(i.user_id, i.item_id, i.rating) for i in interactions]
    df = pd.DataFrame(data, columns=["user_id", "item_id", "rating"])

    # Pivot the DataFrame to create the user-item matrix
    user_item_matrix = df.pivot(index="user_id", columns="item_id", values="rating").fillna(0)

    # Debugging log: Check the shape of the rebuilt matrix
    print(f"Rebuilt user-item matrix shape: {user_item_matrix.shape}")
    return user_item_matrix

def validate_user_item_matrix(user_item_matrix):
    """
    Validate the user-item matrix for common issues.
    """
    if user_item_matrix.empty:
        print("The user-item matrix is empty.")
        return False

    if user_item_matrix.shape[0] < 2 or user_item_matrix.shape[1] < 2:
        print("The user-item matrix has insufficient rows or columns.")
        return False

    print("The user-item matrix is valid.")
    return True

def save_user_item_matrix(user_item_matrix, filepath="user_item_matrix.csv"):
    """
    Save the user-item matrix to a CSV file.
    """
    user_item_matrix.to_csv(filepath)
    print(f"User-item matrix saved to {filepath}")

if __name__ == "__main__":
    # Rebuild the user-item matrix
    user_item_matrix = rebuild_user_item_matrix()

    # Validate the matrix
    if validate_user_item_matrix(user_item_matrix):
        # Save the matrix to a CSV file
        save_user_item_matrix(user_item_matrix)
    else:
        print("User-item matrix validation failed. No file was saved.")
