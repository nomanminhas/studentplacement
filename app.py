from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models
log_reg = pickle.load(open('Models/logregression_model.pkl', 'rb' ))
grid_knn = pickle.load(open('Models/knn_model.pkl', 'rb' ))

# Define a dictionary to map model names to the actual model objects
models = {
    'Logistic Regression': log_reg,
    'Grid KNN': grid_knn,
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_choice = request.form.get('model')
    model = models.get(model_choice)
    
    # Get user input
    features = [float(x) for x in request.form.values() if x != model_choice]
    final_features = [np.array(features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'
    color = 'darkblue' if prediction[0] == 1 else 'red'
    
    return render_template('index.html', prediction_text=f'Predicted Placement Status: {output}', selected_model=model_choice, color=color)

if __name__ == "__main__":
    app.run()