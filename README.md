## Flask Application Design for Time Series Forecasting and Visualization

### HTML Files

- **index.html**: The main page of the application. It will provide an interface for users to upload time series data and view model predictions. It should include elements for displaying model metrics and tracking model status.
- **upload.html**: A separate page for handling the upload of time series data. It should contain a form for users to select and upload a CSV file containing the time series data.
- **results.html**: The page for displaying model predictions and metrics. It will receive the uploaded data and generate the forecast, displaying the predicted values and relevant metrics like R^2 score.

### Routes

- **route for handling data upload**: This route will process the uploaded CSV file and extract the time series data. It will then initiate the model training process and store the trained model.
- **route for model prediction**: This route will take the uploaded or stored time series data, use the trained model to generate predictions, and return the predicted values and model metrics.
- **route for model status tracking**: This route will provide real-time information about the status of the model, including training progress and any errors encountered.
- **route for displaying model metrics**: This route will provide a detailed report of the model's performance metrics, such as R^2 score, MAE, and MAPE.
- **route for displaying predicted values**: This route will display the predicted values generated by the model, allowing users to visualize the forecast.