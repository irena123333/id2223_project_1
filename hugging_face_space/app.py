import streamlit as st
import hopsworks
import joblib
import pandas as pd
import datetime
from functions import *
import pytz

st.set_page_config(layout="wide")

st.title('AQI prediction for Beijing in next week')

project = hopsworks.login()

today=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).date()
#today=datetime.date.today()
#city = "zurich"
city = "beijing"
weekly_data = get_weather_data_weekly(city, today)

# get Hopsworks Model Registry
mr = project.get_model_registry()
# get model object
model = mr.get_model("aqi_ensemble", version=2)
model_dir = model.download()
model = joblib.load(model_dir + "/model.pkl")

weekly_data = data_encoder_0(weekly_data)
preds=model.predict(weekly_data)

next_week = [f"{(today + datetime.timedelta(days=d)).strftime('%Y-%m-%d')},{(today + datetime.timedelta(days=d)).strftime('%A')}" for d in range(7)]

aqi_level = encoder_range(preds.T.reshape(-1, 1))
df = pd.DataFrame(data=[map(int,preds), aqi_level], index=["aqi","Air Pollution Level"], columns=next_week)

st.write(df) 

st.button("Re-run")
