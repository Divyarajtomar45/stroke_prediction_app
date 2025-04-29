from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import pickle
import numpy as np
import pandas as pd
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Set a secret key for session

# Load the model
model_path = 'stroke_prediction_model.pkl'

# Check if model file exists, otherwise create it
if not os.path.exists(model_path):
    # Import functions from our model script to regenerate model
    from stroke_prediction_model import generate_synthetic_data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate dataset
    stroke_data = generate_synthetic_data(2000)
    
    # Prepare features and target
    X = stroke_data.drop('stroke', axis=1)
    y = stroke_data['stroke']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define categorical and numerical features
    categorical_features = ['gender', 'smoking_status']
    numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Create a pipeline with preprocessor and classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Save the model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
else:
    # Load the existing model
    with open(model_path, 'rb') as file:
        model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        age = float(request.form['age'])
        gender = request.form['gender']
        hypertension = int(request.form['hypertension'])
        heart_disease = int(request.form['heart_disease'])
        bmi = float(request.form['bmi'])
        avg_glucose_level = float(request.form['avg_glucose_level'])
        smoking_status = request.form['smoking_status']
        
        # Store input data in session for display on results page
        session['user_data'] = {
            'age': age,
            'gender': gender,
            'hypertension': 'Yes' if hypertension == 1 else 'No',
            'heart_disease': 'Yes' if heart_disease == 1 else 'No',
            'bmi': bmi,
            'avg_glucose_level': avg_glucose_level,
            'smoking_status': smoking_status
        }
        
        # Create a dataframe with the input values
        input_data = pd.DataFrame({
            'age': [age],
            'gender': [gender],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'bmi': [bmi],
            'avg_glucose_level': [avg_glucose_level],
            'smoking_status': [smoking_status]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get prediction probability
        prediction_proba = model.predict_proba(input_data)[0][1]
        
        # Determine risk level
        if prediction_proba < 0.2:
            risk_level = "Low"
            recommendation = "Continue maintaining a healthy lifestyle with regular check-ups."
        elif prediction_proba < 0.4:
            risk_level = "Moderate"
            recommendation = "Consider consulting with a healthcare provider for a detailed assessment."
        elif prediction_proba < 0.6:
            risk_level = "High"
            recommendation = "Strongly advised to consult with a healthcare provider for a detailed assessment and risk management plan."
        else:
            risk_level = "Very High"
            recommendation = "Urgent medical consultation recommended. Please consult with a healthcare provider as soon as possible."
        
        # Store results in session
        session['prediction_result'] = {
            'prediction': int(prediction),
            'probability': float(prediction_proba) * 100,  # Convert to percentage
            'risk_level': risk_level,
            'recommendation': recommendation
        }
        
        # Redirect to results page
        return redirect(url_for('results'))
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/results')
def results():
    # Check if we have results in the session
    if 'prediction_result' not in session:
        return redirect(url_for('home'))
    
    # Get results and user data from session
    prediction_result = session['prediction_result']
    user_data = session['user_data']
    
    return render_template('results.html', 
                          result=prediction_result,
                          user_data=user_data)

if __name__ == '__main__':
    app.run(debug=True)