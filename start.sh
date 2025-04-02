#!/bin/bash

# Make sure to run `chmod +x start.sh` to make this script executable

# Retrain the NCF model
echo "Retraining the Neural Collaborative Filtering (NCF) model..."
python3 train_ncf.py
if [ $? -ne 0 ]; then
    echo "Error retraining the NCF model. Exiting..."
    exit 1
fi
echo "NCF model retrained successfully."

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

# Start Flask in the background and save its PID
flask run --port=5000 &
FLASK_PID=$!

# Function to clean up Flask process and semaphore objects on exit
cleanup() {
    echo "Stopping Flask..."
    kill $FLASK_PID 2>/dev/null
    wait $FLASK_PID 2>/dev/null
    echo "Flask stopped."

    # Explicitly clean up semaphore objects
    echo "Cleaning up leaked semaphore objects..."
    python3 -c "
from multiprocessing import resource_tracker
import os
import platform

# Determine the shared memory path based on the platform
shared_memory_path = '/dev/shm' if platform.system() == 'Linux' else '/private/var/run'

# Check if the shared memory path exists
if os.path.exists(shared_memory_path):
    # Unregister all semaphore objects
    for semaphore in os.listdir(shared_memory_path):
        if semaphore.startswith('sem.'):
            try:
                resource_tracker.unregister(f'{shared_memory_path}/{semaphore}', 'semaphore')
            except KeyError:
                print(f'Semaphore {semaphore} was already unregistered or does not exist.')
            except Exception as e:
                print(f'Error cleaning semaphore {semaphore}: {e}')
else:
    print(f'{shared_memory_path} does not exist. Skipping semaphore cleanup.')
"
    echo "Semaphore cleanup completed."
}

# Set trap to call cleanup on script exit (e.g., Ctrl+C)
trap cleanup EXIT

# Start React frontend
cd health-insurance-recommender  # Change to React directory

# Check if package.json exists
if [ ! -f "package.json" ]; then
    echo "package.json not found in frontend directory!"
    exit 1
fi

npm start