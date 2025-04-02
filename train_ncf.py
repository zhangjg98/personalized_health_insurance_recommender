import pandas as pd
from neural_collaborative_filtering import train_and_save_model, evaluate_model

# Load the updated user-item matrix
user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0).values

# Validate the user-item matrix
if user_item_matrix.size == 0:
    raise ValueError("The user-item matrix is empty. Ensure that interactions are logged before training.")

# Retrain and save the model
print("Retraining the Neural Collaborative Filtering (NCF) model...")
model = train_and_save_model(
    user_item_matrix,
    latent_dim=50,  # Ensure this matches the saved model's latent dimension
    hidden_dim=128,
    epochs=20,
    lr=0.001,
    model_path="ncf_model.pth"
)

# Evaluate the retrained model
mse, f1 = evaluate_model(model, user_item_matrix)
print(f"Model Evaluation - Mean Squared Error (MSE): {mse:.4f}, F1-score: {f1:.4f}")
