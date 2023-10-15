# Import necessary libraries
import json
from io import BytesIO
from PIL import Image
import os
import pandas as pd
import numpy as np
import time
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, explained_variance_score
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import warnings

# Import argparse for command line arguments
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='AI-Ocean Current Prediction')
parser.add_argument('--model', type=str, choices=["None","GI21-Model", "GI31-Model", "GI41-Model", "GULF3-Model", "GULF4-Model", "GULF-Model"], default="None",
                    help='Choose Ocean Current Classifier')
parser.add_argument('--date', type=str, default="2022-12-7",
                    help='Specify the date for ocean current prediction (YYYY-MM-DD)')
parser.add_argument('--hour', type=int, default=0, choices=range(24),
                    help='Specify the hour to start prediction (0-23)')
parser.add_argument('--latitude', type=float, default=12.1146134,
                    help='Specify the latitude for prediction')
parser.add_argument('--longitude', type=float, default=100.8672236,
                    help='Specify the longitude for prediction')

args = parser.parse_args()

# Set the model, date, hour, latitude, and longitude from command line arguments
models = args.model
date_str = args.date
hours = args.hour
lat = args.latitude
lon = args.longitude

# Convert date string to datetime
d = datetime.datetime.strptime(date_str, "%Y-%m-%d")

Day = int(d.day)
Month = int(d.month)
Year = int(d.year)
Hour = int(hours)

Latitude = float(lat)
Longitude = float(lon)

if models != 'None':
	U_MODEL, V_MODEL = st.columns(2)

	U_MODEL.success('OCEAN CURRENT MODEL IS '+str(models).upper()+' AS U-COMPOMENT')
	V_MODEL.success('OCEAN CURRENT MODEL IS '+str(models).upper()+' AS V-COMPOMENT')
	BOUND = (7.1393061, 99.8913451,  9.3034079, 102.9258328)
	if (BOUND[0] <= Latitude <= BOUND[1]) and (BOUND[2] <= Longitude <= BOUND[3]) :

		with st.spinner('Wait for it...'):
		    time.sleep(3)
		#st.success('Done!')

		print('THIS MODEL HAS SUPPORTED THIS LAT, LON.')

		dt = datetime.datetime(Year, Month, Day, Hour, 0, 0)
		end = datetime.datetime(Year, Month, Day+1, 23, 59, 59)
		step = datetime.timedelta(hours=1)

		result = []

		while dt < end:
		    result.append(dt.strftime('%Y-%m-%d %H:%M:%S'))
		    dt += step
		    
		Timestamp = result[:24]

		data = {'Timestamp': Timestamp,
		'Longitude':[Longitude]*24,
		'Latitude':[Latitude]*24
		}

		# Create DataFrame
		df = pd.DataFrame(data)

		df['date'] =  pd.to_datetime(df['Timestamp']) ## pandas recognizes your format
		df['day'] = df['date'].dt.day
		df['month'] = df['date'].dt.month
		df['year'] = df['date'].dt.year
		df['hour'] = df['date'].dt.hour
		df['week_number'] = df['date'].dt.isocalendar().week

		df_to_show = df.copy()

		models_name = models.split('-')[0]
		print('models_name:',models_name+'_ocean_current_U_model.pkl')

		print('models/'+models_name+'_ocean_current_U_model.pkl')
		print('models/'+models_name+'_ocean_current_V_model.pkl')

		u_model_path = 'models/'+models_name+'_ocean_current_U_model.pkl'
		v_model_path = 'models/'+models_name+'_ocean_current_V_model.pkl'

		model_inference_U = pickle.load(open(u_model_path, 'rb'))
		model_inference_V = pickle.load(open(v_model_path, 'rb'))

		df_inference = df[['Longitude', 'Latitude', 'day', 'month', 'hour','week_number']] 

		predict_U = model_inference_U.predict(df_inference)
		predict_V = model_inference_V.predict(df_inference)

		df_to_show['U-Forecast'] = predict_U
		df_to_show['V-Forecast'] = predict_V

		print(df_to_show.columns)

		U_MODEL.dataframe(df_to_show[['Timestamp', 'Longitude', 'Latitude','U-Forecast']])
		V_MODEL.dataframe(df_to_show[['Timestamp', 'Longitude', 'Latitude','V-Forecast']])

		U_MODEL.line_chart(df_to_show[['U-Forecast']])
		V_MODEL.line_chart(df_to_show[['V-Forecast']])


	else:
	    print('THIS MODEL HAS NOT SUPPORTED THIS LAT, LON.')
	    st.warning('THIS MODEL HAS NOT SUPPORTED THESE LAT, LON !!!')







