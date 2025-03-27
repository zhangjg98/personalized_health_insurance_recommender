from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.sql import text
from cryptography.fernet import Fernet
import os

db = SQLAlchemy()

# Generate or load a valid Fernet key
key_file = "fernet_key.key"
if not os.path.exists(key_file):
    print("Fernet key not found. Generating a new key...")
    with open(key_file, "wb") as f:
        f.write(Fernet.generate_key())

with open(key_file, "rb") as f:
    encryption_key = f.read()

# Validate the Fernet key
try:
    cipher = Fernet(encryption_key)
except ValueError as e:
    raise ValueError("Invalid Fernet key. Ensure it is 32 URL-safe base64-encoded bytes.") from e

# Define database models
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    age_group = db.Column(db.String(50))
    smoker = db.Column(db.Boolean)
    bmi_category = db.Column(db.String(50))
    income = db.Column(db.String(50))
    family_size = db.Column(db.String(50))
    chronic_condition = db.Column(db.Boolean)
    medical_care_frequency = db.Column(db.String(50))
    preferred_plan_type = db.Column(db.String(50))

class Item(db.Model):
    __tablename__ = 'items'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    description = db.Column(db.Text)

class Interaction(db.Model):
    __tablename__ = 'interactions'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'))
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    user_inputs = db.Column(db.LargeBinary)  # Store encrypted data as binary

    def set_user_inputs(self, inputs):
        """Encrypt and store user inputs."""
        try:
            if not isinstance(inputs, str):
                raise ValueError("user_inputs must be a string before encryption.")
            self.user_inputs = cipher.encrypt(inputs.encode('utf-8'))
        except Exception as e:
            raise

    def get_user_inputs(self):
        """Decrypt and retrieve user inputs."""
        try:
            return cipher.decrypt(self.user_inputs).decode('utf-8')
        except Exception as e:
            raise

class UserItemMatrix(db.Model):
    __tablename__ = 'user_item_matrix'
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    item_id = db.Column(db.Integer, db.ForeignKey('items.id'), primary_key=True)
    rating = db.Column(db.Float)

def clear_database():
    """
    Utility function to drop all tables in the database and recreate them.
    WARNING: This will delete all data in the database.
    """
    with db.engine.connect() as connection:
        transaction = connection.begin()
        try:
            # Drop all tables
            print("Dropping all tables...")
            db.metadata.drop_all(bind=connection)

            # Recreate all tables
            print("Recreating all tables...")
            db.metadata.create_all(bind=connection)

            transaction.commit()
            print("Database cleared and recreated successfully.")
        except Exception as e:
            transaction.rollback()
            print(f"Error clearing database: {e}")

def verify_encryption(encrypted_data):
    """Decrypt and verify the encrypted user_inputs."""
    try:
        return cipher.decrypt(encrypted_data).decode('utf-8')
    except Exception as e:
        return None
