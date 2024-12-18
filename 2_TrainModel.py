import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Weather Forecast dataset
#datafilename = "https://raw.githubusercontent.com/shravyapendyala/CCE_Assignment_1/refs/heads/main/weather_forecast_data.csv"
datafilename = "data/weather_forecast_data.csv"
dataset=pd.read_csv(datafilename)

# Label Encoding on Rain Column
label_encoder=preprocessing.LabelEncoder()
dataset['Rain']=label_encoder.fit_transform(dataset['Rain'])

# Dimentionality reduction by removing Wind_speed column
new_dataset=dataset.drop(columns=['Wind_Speed'])
# Get X and Y 
X=new_dataset.iloc[:, 0:4]
y=new_dataset.iloc[:, -1]


#X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 0,
}

# Train the model: LogisticRegression
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')  # For binary classification
recall = recall_score(y_test, y_pred, average='binary')  # For binary classification
f1 = f1_score(y_test, y_pred, average='binary') 

print("LogisticRegression Model Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
#print("\nClassification Report:\n", classification_report(y_test, y_pred))

#-------------------------------------------------------------------------------------------------------
# Log the model and its metadata to MLflow

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow WeatherForcast-1")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    model_name = "WeatherForecastModel-LR"  # Add your model name here
    mlflow.log_param("model_name", model_name)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Weather Forcast Leanear Regression model for weather data")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="weather_forcast",
        signature=signature,
        input_example=X_train,
        registered_model_name="ML-model-LR",
    )
#-------------------------------------------------------------------------------------------------------
# Load the model as a Python Function (pyfunc) and use it for inference
# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

weather_feature_names = new_dataset.columns

result = pd.DataFrame(X_test, columns=weather_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions
print("result[:4] = /n", result[:4])
