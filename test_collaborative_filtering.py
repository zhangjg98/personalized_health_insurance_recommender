from neural_collaborative_filtering import load_ncf_model, predict_user_item_interactions
import pandas as pd

def test_collaborative_filtering():
    # Load the trained model
    model, _ = load_ncf_model()

    # Load the correct user-item matrix
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)

    # Map the actual user_id to the zero-based index in the matrix
    actual_user_id = 1  # The user_id from the database
    user_index = user_item_matrix.index.tolist().index(actual_user_id)  # Map to zero-based index

    # Dynamically adjust top_k based on the number of items in the matrix
    num_items = user_item_matrix.shape[1]
    top_k = min(5, num_items)  # Ensure top_k does not exceed the number of items

    # Check if the user_index exists in the matrix
    if user_index < 0 or user_index >= user_item_matrix.shape[0]:
        print(f"User ID {actual_user_id} (index {user_index}) not found in the user-item matrix.")
        print(f"Available user IDs: {user_item_matrix.index.tolist()}")
        return

    recommendations = predict_user_item_interactions(model, user_item_matrix, user_index, top_k=top_k)

    if recommendations:
        print(f"Top {top_k} recommendations for user {actual_user_id}:")
        for rec in recommendations:
            print(rec)
    else:
        print(f"No recommendations available for user {actual_user_id}.")

if __name__ == "__main__":
    test_collaborative_filtering()
