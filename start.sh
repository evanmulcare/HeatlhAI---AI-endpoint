#!/bin/bash

# Activate the virtual environment
source env/bin/activate

# Run your Flask app using Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app  # Update with your actual file name

# Deactivate the virtual environment (optional, depending on your requirements)
deactivate
