from flask_backend import app, db, Interaction
from database import verify_encryption

def verify_interactions():
    with app.app_context():
        interactions = Interaction.query.all()
        for interaction in interactions:
            print(f"Interaction ID: {interaction.id}")
            decrypted_data = verify_encryption(interaction.user_inputs)
            if decrypted_data:
                print(f"Decrypted user_inputs: {decrypted_data}")
            else:
                print("Failed to decrypt user_inputs.")

if __name__ == "__main__":
    verify_interactions()
