# from mlflow.tracking import MlflowClient

# client = MlflowClient()

# # Fetch all registered models
# registered_models = client.search_registered_models()

# for model in registered_models:
#     print(f"Model Name: {model.name}")
#     for version in model.latest_versions:
#         print(f"  Version: {version.version}, Stage: {version.current_stage}, Status: {version.status}")


import mlflow

# Set the tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# List versions of the model
model_name = "RandomForest-data1"
client = mlflow.tracking.MlflowClient()
model_versions = client.get_registered_model(model_name).latest_versions

for version in model_versions:
    print(f"Version: {version.version}, Stage: {version.current_stage}")