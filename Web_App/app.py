import joblib
from sklearn.ensemble import RandomForestRegressor
from tempfile import tempdir
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from PIL import Image
import requests
from datetime import datetime as dt
import datetime 
import time
import plotly.express as px
from pysolar.solar import *



own_endpoint = 'https://api.openweathermap.org/data/2.5/onecall?'
api_key = 'd59d52838df092fbe8a02e73e7298af7'

weather_params = {
    'lat': 50.01,
    'lon': -114.14,
    'appid': api_key,
    'exclude': 'minutely, hourly, daily, alerts', 
    'units': 'metric'
    
}

response = requests.get(own_endpoint, params=weather_params)
response.raise_for_status()
weather_data = requests.get(own_endpoint, params=weather_params).json()

temp = weather_data['current']['temp']
humidity = weather_data['current']['humidity']
pressure = weather_data['current']['pressure']
dew_point = weather_data['current']['dew_point']
wind_speed = weather_data['current']['wind_speed']
image = Image.open('solar.png')


def utc2local(utc):
    epoch = time.mktime(utc.timetuple())
    offset = dt.fromtimestamp(epoch) - dt.utcfromtimestamp(epoch)
    return utc + offset
sun_rise = utc2local(dt.utcfromtimestamp(weather_data['current']['sunrise'])).strftime('%H:%M:%S')

sun_set = utc2local(dt.utcfromtimestamp(weather_data['current']['sunset'])).strftime('%H:%M:%S')
current_time = utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).strftime('%H:%M:%S')
current_time_date = utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).strftime('%Y-%m-%d')
year = utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).year
month = utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).month
day = utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).day
hour = utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).hour
if utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).minute > 45 | utc2local(dt.utcfromtimestamp(weather_data['current']['dt'])).minute < 45:
    min = 0
    minute = 0
else:
    min = 0.5
    minute = 30
input_time = hour + min



cloud = {'Cirrus': 0,
         'Clear': 0,
          'Fog': 0,
          'Opaque Ice':0,
          'Overlapping': 0,
          'Overshooting': 0,
          'Probably Clear':0,
          'Super-Cooled Water': 0,
          'Unknown': 0,
          'Water': 0}

st.write("""

# ☀️ Calgary Solar Power and Energy Output Prediction Web App ☀️



😎 This app etimates your current solar power and monthly solar energy output! 😎
***
"""
)
col1, col2, col3 = st.columns(3)
col1.metric("🗓 Date", current_time_date)
col2.metric("🌄 Local sunrise Time", sun_rise)
col3.metric("🌇 Local sunset Time", sun_set)


st.sidebar.image(image)
st.sidebar.header('Customer Input')
input_panelnum = st.sidebar.number_input('Number of panels')
input_panellength = st.sidebar.number_input('Single panel Length(m)')
input_panelwidth = st.sidebar.number_input('Single panel width(m)')
input_panelefficiency = st.sidebar.number_input('Panel efficiency (%)')
input_maxoutput = st.sidebar.number_input('Max single panel output (w)')
input_roofslope = st.sidebar.selectbox('Roof Slope',list('%d/12' % i for i in range (4, 9)))
input_rooffacing = st.sidebar.selectbox('Roof Facing', list(['S', 'SW', 'W', 'NW', 'N', 'NE', 'E', 'SE']))
select_cloudtype = st.sidebar.selectbox('Cloud Type', list(['Cirrus', 'Clear', 'Fog', 'Opaque Ice', 'Overlapping', 'Overshooting', 'Probably Clear', 'Super-Cooled Water', 'Unknown', 'Water']))
solar_date = datetime.datetime(year, month, day, hour, minute, tzinfo=datetime.timezone.utc) - datetime.timedelta(hours=-7)

roof_tilt = {'4/12': 2.67,
             '5/12': 2.59,
             '6/12': 2.45,
             '7/12': 2.28,
             '8/12': 2.08}

roof_facing = {'S': 1,
               'SW': 0.875,
               'W': 0.58,
               'NW': 0.29, 
               'N': 0.19,
               'NE': 0.29, 
               'E': 0.58,
               'SE': 0.875}


total_panel = input_panelnum*input_panellength*input_panelwidth*input_panelefficiency/100

result = st.sidebar.button('Enter')

col1, col2 = st.columns([3, 1])


col2.subheader("Current Weather")
col2.text_input('Local time', value=current_time)
col2.text_input('Temperature (Unit: Celsius)', value=temp)
col2.text_input('Dew Point (Unit: Celsius)', value=dew_point)
col2.text_input('Wind Speed (Unit: m/s)', value=wind_speed)
col2.text_input('Relative Humidity (%)', value=humidity)
col2.text_input('Pressure (Unit: kpa)', value=pressure/10)

input_pressure = np.round(0.02953*pressure*((288-0.0065*1089)/288)**5.2561*33.8639+0.1, decimals=-1)
input_temperature = np.round(temp)
input_solar_azimuth = 90 - get_altitude(51.01, -114.14, solar_date)

filename = 'finalized_xgboost_model.pkl'
loaded_model = joblib.load(filename)


cloud_data = {'Cirrus': 0,
              'Clear': 0,
              'Fog': 0,
              'Opaque Ice':0,
              'Overlapping': 0,
              'Overshooting': 0,
              'Probably Clear':0,
              'Super-Cooled Water': 0,
              'Unknown': 0,
              'Water': 0}

cloud_data[select_cloudtype]=1

defualt_data = {'Day': day, 
                'Solar Zenith Angle': input_solar_azimuth, 
                'Dew Point' : round(dew_point), 
                'Wind Speed': wind_speed, 
                'Relative Humidity': humidity, 
                'Temperature': round(temp), 
                'Pressure': input_pressure, 
                'time': input_time, 
                'Month': month}


energy = pd.read_csv('app_energy.csv', index_col=0)
energy = np.round(energy*roof_tilt[input_roofslope]*roof_facing[input_rooffacing]*total_panel, decimals=2)

df_default = pd.DataFrame(defualt_data, index=[1])
df_cloud = pd.DataFrame(cloud_data, index=[1])
df_input = pd.concat([df_default,df_cloud], axis=1)
model_result = loaded_model.predict(df_input)[0]
output_result = round(model_result*roof_tilt[input_roofslope]*roof_facing[input_rooffacing]*total_panel)

fig = px.bar(energy, x=energy.index, y='predicted_mean', text='predicted_mean')

fig.update_layout(
    yaxis_title="Energy Output (kWh)", 
    xaxis_title="Month",
    title='Your Estimated Monthly Solar Energy Output'
)
if output_result >= input_panelnum*input_maxoutput:
    power_output = input_panelnum*input_maxoutput
elif model_result <= 0:
    power_output = 0
elif current_time < sun_rise or current_time > sun_set:
    power_output = 0
else:
    power_output = output_result



if result:
    col1.write('')

    col1.header(f'You are expecting {power_output}w solar power output now')
    if power_output > 0:
        col1.write('😃')
    else:
        col1.write('😞')
    col1.plotly_chart(fig, use_container_width=True)
          

    







