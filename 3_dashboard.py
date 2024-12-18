import streamlit as st
from sklearn.preprocessing import StandardScaler
import mlflow
import pandas as pd
import base64

st.markdown("""
            
    <style>
    .stMainBlockContainer{
           padding-top:2rem; 
    }
    .center-content {
        text-align: center; /* Centrer le contenu */
    }
    .center-image {
        display: block;
        margin: 0 auto; /* Centrer l'image horizontalement */
    }
            
    .stSlider > div[data-baseweb="slider"] > div {
        color: #FF5733; 
    }
    .stSlider > div[data-baseweb="slider"] > div > div {
        # color: green; 
    }
    </style>
""", unsafe_allow_html=True)

def get_image_base64(image_path):
    with open(image_path, "rb") as file:
        data = file.read()
    return base64.b64encode(data).decode("utf-8")

# Charger l'image en base64
logo_base64 = get_image_base64("logo.png")

# Utilisation de balises HTML pour centrer l'image et le titre
st.markdown(f"""
    <div class="center-content">
        <img src="data:image/png;base64,{logo_base64}" alt="Logo" class="center-image" width="150">
        <h1 style="font-style: italic;  margin-bottom:2rem">Weather Prediction Web App</h1>
    </div>
""", unsafe_allow_html=True)

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

mlflow.set_tracking_uri("http://127.0.0.1:8080")

# Logged model
logged_model = 'runs:/6a6ba87376624b2eb18f3828cc6374ce/RandomForest-data1_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


 # Creating button for Prediction
if st.button('Predict Raifall'):
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


