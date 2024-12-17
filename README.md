# G1_AIML_Project_WeatherPrediction
ML Based Weather/Rainfall Prediction with MLOps 

## FileSructure
Below files are used for this-
notebook/Weather_Prediction.ipynb: notebook to be run on google collab
data/weather_forecast_data.csv : Dataset
1_Data_EDA.py : code to do exploratory data anlysis
2_TrainModel.py : code to train model using LR and connecting to MLFlow
3_dashboard.py : code for webapp for weather forecast to take inputs and predict based on MLFlow model

# Weather Prediction Model

The objective of this project is to develop a model that can accurately forecast rainfall using a variety of input variables, such as temperature, humidity, wind speed, and atmospheric pressure.

## Features
- Temperature
- Humidity
- Wind Speed
- Cloud Cover
- Pressure


## Target Variable
- Will there be Rain or Not
## Team Members
- Aniket Shukla
- Abhishek Yadav
- Sagar Kamble
- Sandeep Mamoriya
- Sherine Martina
- Shravya Pendyala
- Sridhar Mulumoodi

## Data Set
- The Data set was downloaded from kaggle.com 
- It Consist of 2,500 rows
- It is a Classification Problem


## Methodology
- The process of solving a classification problem in machine learning typically begins with problem definition to understand the objectives and the nature of the data. This is followed by data collection and preprocessing, where the raw data is gathered, cleaned, and transformed to ensure consistency, removing missing values and handling outliers. Exploratory Data Analysis (EDA) is then performed to gain insights into the data, identifying trends, relationships, and patterns. Subsequently, feature selection and engineering are conducted to choose relevant features improve model performance. The data is then split into training and testing sets, often with an additional validation set if hyperparameter tuning is needed. Various classification algorithms are applied to the training data. Hyperparameter tuning is performed using techniques like grid search or random search to optimize the model. Finally, the model is deployed for predictions, followed by continuous monitoring and updates to ensure it performs well in real-world scenarios.

- The feature selection was done to improve the Model performance, reduced over fitting and simplify the model, and at the end, it was concluded that wind speed feature would not be required 


## Model used
The dataset was split into training (70%) and testing (30%) subsets to ensure effective model evaluation. Four machine learning algorithms—Random Forest Classifier, Logistic Regression, Support Vector Machine (SVM), and XGBoost—were employed for training. The performance of each model was assessed by evaluating their predictive accuracy on the testing dataset. The Pros and Cons of each model used are as under:-

- ## Random Forest Classifier
 Random Forest is a robust ML algorithm that provides high accuracy for classification and regression
   - ## Pros
        -   High accuracy
        -   Versatile in dealing with complex datasets containing outliers
            Effective classification and regression
   - ## Cons
        -   Handling huge datasets can be time consuming
        - Not preferred when the model must be highly interpretable
- ## Logistic Regression
Logistic Regression is a classification algorithm than finds a linear relationship in the dataset and provides a non-linearity in the form of a Sigmoid function

   - ## Pros
        -   Good accuracy for simply datasets
        -   Easy to implement and interpret
        -   Interprets model coefficients as indicators of feature importance
   - ## Cons
Non-linear problems can’t be dealt with this model
Not preferred when the number of features is more than the observations

- ## SVM Method
 Support Vector Machine is a supervised ML algorithm which gives the hyperplane and separates the data into two classes
   - ## Pros
        -   Models high-dimensional data
        -   Possess good generalization performance (classifies new and unseen data)
   - ## Cons
        -   Handling huge datasets is not possible
        -   Limited to two-class problems (multi-class is dealt using other strategies)

- ## XGBoost Method
 XGBoost (Extreme Gradient Boosting) is an ensemble method combining multiple decision trees to give high accuracy and precision
   - ## Pros
        -   High accuracy and high precision
        -   Regularization techniques avoids over-fitting

   - ## Cons
        -   Tuning the hyper-parameters of this algorithm can be time consuming
        -   “Black Box” algorithm as it is difficult to interpret and understand the predictions.





## Result and Analysis
![WeatherPred_EDA](https://github.com/user-attachments/assets/22a8bd19-e245-4584-b411-0d85d4e52d87)


| Training Model            | Accuracy                                                              |
| ----------------- | ------------------------------------------------------------------ |
| Random Forest Classifier | 99.3 |
| Logistic Regression | 93.2 |
| SVM | 86.4 |
| XG Boost | 100|


The Random forest classifier model was selected for the final prediction

## MLOps
Below files are used for this
data/weather_forecast_data.csv : Dataset
1_Data_EDA.py : code to do exploratory data anlysis
2_TrainModel.py : code to train model using LR and connecting to MLFlow
3_dashboard.py : code for webapp for weather forecast to take inputs and predict based on MLFlow model

Settings needed:
1. MLflow experiment/model training - Run a local Tracking Server uri on Terminal-->  mlflow server --host 127.0.0.1 --port 8080 (then call it via script as mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
2. To connect Streamlit with mlflow,  run this in terminal--> mlflow server --host 0.0.0.0 --port 5000
3. Then run this in terminal to see webapp in browser (Streamlit on server: Opens automatically) --> streamlit run 3_dashboard.py

Details:
- A PoC was made to setup MLOps for this project
- Code in file '2_TrainModel.py' gets data from the Git repo and uses for training and validating a model. Here, for reference, logistic regression method is used. However, this can be replaced with any other method. Trained model then gets registed into MLFlow registry. Registered model is then called for deployement
- Deployed model is then called over local server and is connected to Webapp developed using Streamlit. The code in '3_dashboard.py' calls model for prediction by using input data given by user for rainfall prediction.

Models in MLOps and Output Results/Dashboard:
![image](https://github.com/user-attachments/assets/0cee56cb-1459-4f12-8f25-80344aebf4f9)
![image](https://github.com/user-attachments/assets/1566e071-1255-41eb-8465-ac2c42885445)
![RainPred_Yes](https://github.com/user-attachments/assets/12e3be82-1e37-4a10-9347-e0ec86aac430)
![RainPred_No](https://github.com/user-attachments/assets/bfa20b79-3703-466f-8f91-ce357c58a672)

## Help/contact:
For any questions/feedback, please contact: Sagar Kamble, sbkamble816@gmail.com
