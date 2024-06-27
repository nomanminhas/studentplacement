from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models

with open('Models/logregression_model.pkl', 'rb') as model_file:
    logistic_regression_model = pickle.load(model_file)

with open('Models/knn_model.pkl', 'rb') as model_file:
    grid_knn_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    gender_encoded = 1 if gender == 'Male' else 0
    
    hsc_s = request.form['hsc_s']
    hsc_s_encoded = {'Commerce': 0, 'Science': 1, 'Arts': 2}[hsc_s]
    
    degree_t = request.form['degree_t']
    degree_t_encoded = {'Comm&Mgmt': 0, 'Sci&Tech': 1, 'Others': 2}[degree_t]
     
    specialisation = request.form['specialisation']
    specialisation_encoded = {'Mkt&Fin': 0, 'Mkt&HR': 1}[specialisation]
    
    workex = float(request.form['workex'])
    ssc_p = float(request.form['ssc_p'])
    hsc_p = float(request.form['hsc_p'])
    degree_p = float(request.form['degree_p'])
    etest_p = float(request.form['etest_p'])
    mba_p = float(request.form['mba_p'])

    model_choice = request.form.get('model')
    
    # Get user input
    features = np.array([[gender_encoded, ssc_p, hsc_p, hsc_s_encoded, degree_p, degree_t_encoded, workex, etest_p, specialisation_encoded, mba_p]])
    
    if model_choice == 'Logistic Regression':
        prediction = logistic_regression_model.predict(features)
    elif model_choice == 'Grid KNN':
        prediction = grid_knn_model.predict(features)

# Map the prediction result to a string
    prediction_text = 'Placed' if prediction == 1 else 'Not Placed'
    
    return render_template('index.html', prediction_text=f'Predicted Placement: {prediction_text}')



if __name__ == "__main__":
    app.run()