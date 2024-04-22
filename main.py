
# Import the necessary libraries
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from joblib import dump, load

# Create a Flask application
app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

# Define the route for handling data upload
@app.route('/upload', methods=['POST'])
def upload():
    """Handle data upload and model training."""

    # Extract the uploaded data from the request
    csv_file = request.files['data_file']
    data = pd.read_csv(csv_file)

    # Train the model
    model = LinearRegression()
    model.fit(data[['Time']], data['Value'])

    # Save the model
    dump(model, 'model.joblib')

    # Render the results page with the model metrics
    return render_template('results.html', metrics={
        'R^2': r2_score(data['Value'], model.predict(data[['Time']]))
    })

# Define the route for model prediction
@app.route('/predict')
def predict():
    """Handle model prediction."""

    # Load the data and model
    data = pd.read_csv('data.csv')
    model = load('model.joblib')

    # Generate the predictions
    predictions = model.predict(data[['Time']])

    # Render the results page with the predictions
    return render_template('results.html', predictions=predictions)

# Main driver function
if __name__ == '__main__':
    app.run(debug=True)
