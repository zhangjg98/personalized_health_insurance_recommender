#!/bin/bash

# Make sure to run `chmod +x start.sh` to make this script executable

# Start Flask backend
echo "Starting Flask..."
cd "$(dirname "$0")"  # Change to the script's directory

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

source venv/bin/activate  # Activate virtual environment (Mac/Linux)

# Check if Flask app file exists
if [ ! -f "flask_backend.py" ]; then
    echo "Flask app file not found!"
    exit 1
fi

export FLASK_APP=flask_backend.py
flask run --port=5000 &  # Run in the background

# Start React frontend

cd health-insurance-recommender  # Change to React directory

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "package.json not found in frontend directory!"
    exit 1
fi

npm start