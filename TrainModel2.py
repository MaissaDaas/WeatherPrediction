import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import os

# Load the dataset
datafilename = "data/weather_forecast_data.csv"
dataset = pd.read_csv(datafilename)

# Preprocess the dataset
label_encoder = LabelEncoder()
dataset["Rain"] = label_encoder.fit_transform(dataset["Rain"])
# Drop unnecessary columns
new_dataset = dataset.drop(columns=["Wind_Speed"])

# Split features and labels
X = new_dataset.drop(columns=["Rain"])
y = new_dataset["Rain"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:8080") 
mlflow.set_experiment("MLflow WeatherForcast-1") 

# Define models and training logic
models = {
    "LogisticRegression-data1": LogisticRegression(
        solver="lbfgs", max_iter=1000, random_state=42
    ),
    "RandomForest-data1": RandomForestClassifier(n_estimators=100, random_state=42),
}

for model_name, model in models.items():
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_file", os.path.basename(datafilename))
        if model_name == "LogisticRegression-data1":
            mlflow.log_param("solver", "lbfgs")
            mlflow.log_param("max_iter", 1000)
        elif model_name == "RandomForest-data1":
            mlflow.log_param("n_estimators", 100)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"{model_name}_model",
            signature=signature,
            registered_model_name=f"Weather_Forecast_{model_name}",
        )

        # Log additional metadata
        mlflow.set_tag("description", f"Training run for {model_name}")
        print(f"{model_name} logged with accuracy: {accuracy:.4f}")