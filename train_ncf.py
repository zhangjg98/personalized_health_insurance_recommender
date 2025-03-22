import pandas as pd
from neural_collaborative_filtering import train_and_save_model, evaluate_model

# Load user-item matrix
user_item_matrix = pd.read_csv("processed_user_item_matrix.csv", index_col=0).values

# Train and save the model
model = train_and_save_model(user_item_matrix, latent_dim=50, hidden_dim=128, epochs=20, lr=0.001)

# Evaluate the model
mse, f1 = evaluate_model(model, user_item_matrix)
print(f"Model Evaluation - Mean Squared Error (MSE): {mse:.4f}, F1-score: {f1:.4f}")
