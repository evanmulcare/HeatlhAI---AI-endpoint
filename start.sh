#!/bin/bash
source env/bin/activate

# Run your Flask app
gunicorn app:app
