# utils/file_utils.py
import json
import os
import pickle

def load_users(users_file):
    if os.path.exists(users_file):
        try:
            with open(users_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_users(data, users_file):
    with open(users_file, "w") as f:
        json.dump(data, f, indent=4)

def load_encodings(encodings_file):
    if os.path.exists(encodings_file):
        with open(encodings_file, 'rb') as f:
            data = pickle.load(f)
            return data.get('encodings', []), data.get('usernames', [])
    return [], []

def save_encodings(encodings, usernames, encodings_file):
    with open(encodings_file, 'wb') as f:
        pickle.dump({
            'encodings': encodings,
            'usernames': usernames
        }, f)
