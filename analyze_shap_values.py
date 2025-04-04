import pandas as pd
import shap
import numpy as np
import torch
from neural_collaborative_filtering import load_ncf_model

def aggregate_shap_values():
    # Load the trained model and user-item matrix
    model, user_item_matrix = load_ncf_model()

    # Prepare representative data for SHAP
    num_users, num_items = user_item_matrix.shape
    combined_inputs = np.array([[i, j] for i in range(num_users) for j in range(num_items)])

    # Define a wrapper function for SHAP
    def model_predict(inputs):
        """
        SHAP-compatible prediction function.
        Inputs should be a NumPy array where each row contains [user_id, item_id].
        """
        user_inputs = torch.tensor(inputs[:, 0], dtype=torch.long)
        item_inputs = torch.tensor(inputs[:, 1], dtype=torch.long)
        with torch.no_grad():
            predictions = model(user_inputs, item_inputs)
        return predictions.numpy()

    # Define a SHAP explainer
    explainer = shap.Explainer(model_predict, combined_inputs)

    # Compute SHAP values
    shap_values = explainer(combined_inputs)

    # Aggregate SHAP values for each item
    item_shap_values = shap_values.values[:, 1]  # Extract SHAP values for the `item` dimension
    item_shap_matrix = item_shap_values.reshape(num_users, num_items)

    # Create a DataFrame for aggregated SHAP values
    shap_df = pd.DataFrame(item_shap_matrix.T, index=user_item_matrix.columns, columns=user_item_matrix.index)
    aggregated_shap = shap_df.abs().mean(axis=1).sort_values(ascending=False)

    # Save aggregated SHAP values to a CSV file
    aggregated_shap.to_csv("aggregated_shap_values.csv", header=["Mean Absolute Impact"])
    print("Aggregated SHAP values saved to aggregated_shap_values.csv")

def generate_user_friendly_shap_explanations(model, user_item_matrix, user_id, item_id):
    """
    Generate simplified SHAP explanations for a specific user-item pair.

    Parameters:
        model (NeuralCollaborativeFiltering): Trained NCF model.
        user_item_matrix (DataFrame): User-item interaction matrix.
        user_id (int): User ID.
        item_id (int): Item ID.

    Returns:
        dict: Simplified SHAP explanations with feature impacts.
    """
    try:
        # Prepare input for SHAP
        num_users, num_items = user_item_matrix.shape
        combined_inputs = np.array([[i, j] for i in range(num_users) for j in range(num_items)])

        # Define a wrapper function for SHAP
        def model_predict(inputs):
            user_inputs = torch.tensor(inputs[:, 0], dtype=torch.long)
            item_inputs = torch.tensor(inputs[:, 1], dtype=torch.long)
            with torch.no_grad():
                predictions = model(user_inputs, item_inputs)
            return predictions.numpy()

        # Define a SHAP explainer
        explainer = shap.Explainer(model_predict, combined_inputs)

        # Compute SHAP values for the specific user-item pair
        shap_values = explainer(np.array([[user_id, item_id]]))

        # Extract SHAP values for the specific user-item pair
        user_shap_values = shap_values.values[0]
        feature_impact = {
            feature: round(abs(impact), 4)
            for feature, impact in zip(["user", "item"], user_shap_values)
        }

        # Sort features by impact
        sorted_features = sorted(feature_impact.items(), key=lambda x: x[1], reverse=True)

        # Return the top 2 features with their impacts
        return {
            "top_features": sorted_features[:2],
            "explanation": "These features had the highest impact on the recommendation."
        }
    except Exception as e:
        print(f"Error generating SHAP explanations: {e}")
        return {"error": "Unable to generate SHAP explanations."}

if __name__ == "__main__":
    aggregate_shap_values()
