from database import db, clear_database
from flask_backend import app

with app.app_context():
    clear_database()