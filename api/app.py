from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and feature names
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Define the original categorical columns from adult.csv
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                    'relationship', 'race', 'gender', 'native-country']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input data
        data = request.get_json()
        
        # Convert input to DataFrame (expects 14 original features)
        input_df = pd.DataFrame([data])
        
        # One-hot encode the input data
        input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
        
        # Ensure all expected columns are present (fill missing with 0)
        for col in feature_names:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        
        # Reorder columns to match training data
        input_encoded = input_encoded[feature_names]
        
        # Scale the input
        input_scaled = scaler.transform(input_encoded)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0].tolist()
        
        # Return result
        return jsonify({
            'prediction': int(prediction),
            'probability': probability
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)