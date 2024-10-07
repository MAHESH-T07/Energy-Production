import pandas as pd
import numpy as np
import streamlit as st
from pickle import load
from sklearn.preprocessing import StandardScaler

st.title('PREDICTING ENERGY PRODUCTION')

def create_page():
	temp_val = st.number_input('Enter Tempareture')
	st.write(temp_val)
	vaccum_val = st.number_input('Enter Exhaust_Vaccum')
	st.write(vaccum_val)
	pressure_val = st.number_input('Enter Pressure')
	st.write(pressure_val)
	hum_val = st.number_input('Enter Humidity')
	st.write(hum_val)

	data = {'temperature':temp_val,
		'exhaust_vacuum':vaccum_val,
		'amb_pressure' :pressure_val,
		'humidity' :hum_val}
	df = pd.DataFrame(data,index=[0])
	return df
features = create_page()

if st.button('Submit'):
	scaler = load(open('xscale.pkl','rb'))
	features = scaler.transform(features)
	#st.write(features)

	loaded_model = load(open('xgb_model.pkl','rb'))
	res = loaded_model.predict(features)

	yscal = load(open('yscale.pkl','rb'))
	res = np.array(res).reshape(-1,1)
	inversed_res = yscal.inverse_transform(res)

	st.write('PREDICTED ENERGY')
	st.write(inversed_res)

