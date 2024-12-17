import streamlit as st
from sklearn.preprocessing import StandardScaler
import mlflow
import pandas as pd

# Giving Title
st.title('_Weather Prediction Web App_')
 
# Getting input data from the user
#st.subheader("Temperature: ")
temp = st.slider("Temperature", min_value=0.00, max_value=50.00, value=10.00)
#print("temp=", temp)

#st.subheader("Humidity: ")
humd = st.slider("Humidity", min_value=0.00, max_value=200.00, value=10.00)

#st.subheader("Wind_Speed: ")
#winspd = st.slider("Wind_Speed", min_value=0, max_value=20, value=10)

#st.subheader("Cloud_Cover: ")
cldcov = st.slider("Cloud_Cover", min_value=0.00, max_value=200.00, value=10.00)

#st.subheader("Pressure: ")
pres = st.slider("Pressure", min_value=0.00, max_value=1100.00, value=100.00)
prediction = 2 #init

#------------------------------
# Setup MLflow

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Logged model
logged_model = 'runs:/a6ed523cccc3490cb616ec59758b41ce/weather_forcast'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


 # Creating button for Prediction
if st.button('Predict Raifall'):
    #prediction = loaded_model.predict({"Temperature":32.539070,	"Humidity": 98.918446,	"Cloud_Cover": 97.130298,	"Pressure": 1034.046342}) #Yes
    #prediction = loaded_model.predict({"Temperature":27.726270,	"Humidity": 32.703594,	"Cloud_Cover": 22.387709,	"Pressure": 1008.501313}) #No
    prediction = loaded_model.predict({"Temperature":temp,	"Humidity": humd,	"Cloud_Cover": cldcov,	"Pressure": pres}) #GetInputsViaDashboard
    print("prediction=", prediction)

output = st.columns([2,1])
output[0].markdown("Will it Rain?")

if prediction ==1:
    output[1].success("Yes")
elif prediction==0:
    output[1].error("No")



#---------------------------------------------------------------------------------
# Footnote
st.subheader('')
st.image("CCE-New-logo.jpg", width=100)
st.caption("Developed by Group-1, CCE-IISc-Nov-2024")
