from flask_backend import app, db, User

def add_test_user():
    with app.app_context():
        # Check if the user already exists
        existing_user = User.query.filter_by(id=1).first()
        if existing_user:
            print("User with id=1 already exists.")
            return

        # Add a new test user
        new_user = User(
            id=1,
            age_group="young_adult",
            smoker=False,
            bmi_category="normal",
            income="below_30000",
            family_size="1",
            chronic_condition=False,
            medical_care_frequency="Low",
            preferred_plan_type="HMO"
        )
        db.session.add(new_user)
        db.session.commit()
        print("Test user added successfully.")

if __name__ == "__main__":
    add_test_user()
