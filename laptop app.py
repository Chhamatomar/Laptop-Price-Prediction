import streamlit as st
import pickle
import numpy as np
import pandas as pd

df = pickle.load(open('df.pkl', 'rb'))
pipe = pickle.load(open('pipe.pkl', 'rb'))

# ---------- UI ----------
st.title("Laptop Price Predictor")

company = st.selectbox('Brand', df['Company'].unique())
type_name = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM(in GB)', [2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight')

touch = st.selectbox('TouchScreen',['No','Yes'])
ips = st.selectbox('IPS',['No','Yes'])

screen = st.number_input('Screen Size', min_value=1.0)

resolution = st.selectbox('Resolution',[
'1920-1080','1366-768','1600-900'
])

cpu = st.selectbox('CPU', df['Cpu_brand'].unique())
hdd = st.selectbox('HDD',[0,128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
os = st.selectbox('OS', df['OS'].unique())

# ---------- PREDICT ----------
if st.button('Predict Price'):

    touch = 1 if touch == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    X = int(resolution.split('-')[0])
    Y = int(resolution.split('-')[1])

    ppi = ((X**2 + Y**2)**0.5) / screen

    query = pd.DataFrame({
        'Company':[company],
        'TypeName':[type_name],
        'Ram':[ram],
        'Weight':[weight],
        'TouchScreen':[touch],
        'IPS':[ips],
        'ppi':[ppi],
        'Cpu_brand':[cpu],
        'HDD':[hdd],
        'SSD':[ssd],
        'Gpu Brand':[gpu],
        'OS':[os]
    })

    try:
        price = np.exp(pipe.predict(query)[0])
        st.success(f"₹ {int(price)}")
    except Exception as e:
        st.error(e)