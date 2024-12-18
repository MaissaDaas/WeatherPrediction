import mlflow
import mlflow.sklearn
import pandas as pd

# Set the MLflow experiment and URI
mlflow.set_tracking_uri("http://127.0.0.1:8080")
mlflow.set_experiment("MLflow WeatherForcast-1")

# Retrieve the runs for the experiment
experiment_id = mlflow.get_experiment_by_name("MLflow WeatherForcast-1").experiment_id
runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string="")

# Collect the relevant metrics for comparison
metrics = {}
run_ids = {}
for _, run in runs.iterrows():  # Use iterrows() instead of itertuples() for row access
    model_name = run['params.model_name']  # Access params as columns
    accuracy = run['metrics.accuracy']  # Access metrics as columns
    precision = run['metrics.precision']
    recall = run['metrics.recall']
    f1_score = run['metrics.f1_score']
    
    # Store the metrics and run ID for each model
    metrics[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    run_ids[model_name] = run['run_id']  # Keep track of run IDs for model registration

# Print the metrics for comparison
print("Model Comparison:")
for model_name, metric in metrics.items():
    print(f"\n{model_name}:")
    for metric_name, value in metric.items():
        print(f"  {metric_name}: {value}")

# Compare models and select the best based on a chosen metric (e.g., accuracy or F1-score)
best_model_name = max(metrics, key=lambda x: metrics[x]['f1_score'])  # or 'accuracy' depending on preference
best_model_metrics = metrics[best_model_name]
best_model_run_id = run_ids[best_model_name]

print(f"\nBest Model: {best_model_name} with F1-Score: {best_model_metrics['f1_score']}")

# Register the best model to the Model Registry
model_uri = f"runs:/{best_model_run_id}/model"  # Specify the model's URI based on run ID

with mlflow.start_run(run_id=best_model_run_id):
    registered_model = mlflow.register_model(model_uri, best_model_name)
    print(f"Registered Model: {registered_model.name}, Version: {registered_model.version}")

# Transition the model to "Production" or other stages
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=best_model_name,
    version=registered_model.version,
    stage="Production",  # Change stage to "Production", "Staging", etc.
)

print(f"Model {best_model_name} version {registered_model.version} is now in 'Production'.")
