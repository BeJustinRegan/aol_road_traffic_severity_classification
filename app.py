import pandas as pd
import numpy as np
import sklearn
import streamlit as st
import joblib
import shap
import matplotlib
from PIL import Image
from sklearn.preprocessing import OrdinalEncoder


model = joblib.load("rta_model_deploy3.joblib")
encoder = joblib.load("ordinal_encoder2.joblib")

st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config(page_title="Accident Severity Prediction Generator",
                   page_icon="🚧", layout="wide")


options_No_vehicles = ['1','2','3','4','5','6','7','8','9','10']
options_No_casualites = ['1','2','3','4','5','6','7','8','9','10']
options_day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", 'Sunday']
options_age = ['Under 18','18-30', '31-50', 'Over 51']

options_types_collision = ['Vehicle with vehicle collision', 'Collision with roadside objects',
                           'Collision with pedestrians', 'Rollover', 'Collision with animals',
                           'Unknown', 'Collision with roadside-parked vehicles', 'Fall from vehicles',
                           'Other', 'With Train']

options_sex = ['Male', 'Female', 'Unknown']

options_education_level = ['Junior high school', 'Elementary school', 'High school',
                           'Unknown', 'Above high school', 'Writing & reading', 'Illiterate']

options_services_year = ['2-5yrs', 'Above 10yr', '5-10yrs', '1-2yr', 'Below 1yr']

options_acc_area = ['Office areas', 'Residential areas', 'Church areas',
                    'Industrial areas', 'School areas', 'Recreational areas',
                    'Outside rural areas', 'Hospital areas', 'Market areas',
                    'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
                    'Recreational areas']

# Features list
features = ['Number_of_vehicles_involved', 'Number_of_casualties', 'Hour_of_Day', 'Type_of_collision', 'Age_band_of_driver', 'Sex_of_driver',
            'Educational_level', 'Service_year_of_vehicle', 'Day_of_week', 'Area_accident_occured']


st.markdown("Accident Severity Prediction App 🚧", unsafe_allow_html=True)


def main():
    with st.form("road_traffic_severity_form"):
        st.subheader("Please enter the following inputs:")
        
        No_vehicles = st.selectbox("Number of vehicles involved:", options = options_No_vehicles)
        No_casualties = st.selectbox("Number of casualties:", options = options_No_casualites)
        Hour = st.slider("Hour of the day:", 0, 23, value=0, format="%d")
        collision = st.selectbox("Type of collision:", options=options_types_collision)
        Age_band = st.selectbox("Driver age group:", options=options_age)
        Sex = st.selectbox("Sex of the driver:", options=options_sex)
        Education = st.selectbox("Education of driver:", options=options_education_level)
        service_vehicle = st.selectbox("Service year of vehicle:", options=options_services_year)
        Day_week = st.selectbox("Day of the week:", options=options_day)
        Accident_area = st.selectbox("Area of accident:", options=options_acc_area)
        
        submit = st.form_submit_button("Predict")

        if submit:
            input_array = np.array([collision, Age_band, Sex, Education, service_vehicle,
                                    Day_week, Accident_area], ndmin=2)   
           
            encoded_arr = list(encoder.transform(input_array).ravel())
            
            num_arr = [No_vehicles, No_casualties, Hour]
            pred_arr = np.array(num_arr + encoded_arr).reshape(1, -1)
            
            prediction = model.predict(pred_arr)
            
            if prediction == 0:
                st.write(f"The severity prediction is Fatal Injury")
                st.write(prediction)
            elif prediction == 1:
                st.write(f"The severity prediction is Serious Injury")
                st.write(prediction)
            else:
                st.write(f"The severity prediction is Slight Injury")
                st.write(prediction)

a, b, c = st.columns([0.2, 0.6, 0.2])
with b:
    st.image("1000_F_127266399_ELis7NCcniWfqfqu25VOQrvSTJTOVghZ.jpg", use_column_width=True)


st.subheader("🧾Description:")
st.text("""Accident Severity Prediction App helps predict the severity of road traffic accidents based on various factors such as the number of vehicles involved, the age and sex of the driver, 
education level, type of collision, and more. By inputting relevant details, users can obtain an immediate prediction on whether an accident is likely to result in slight injury, serious injury, or a fatal injury. This tool leverages machine learning models trained on historical data to provide accurate and actionable insights, 
aiming to enhance road safety awareness and preventive measures.This data set is collected from Addis Ababa Sub-city police departments for master's research work. 

""")



               
if __name__ == '__main__':
    main()
