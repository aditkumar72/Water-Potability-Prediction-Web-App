import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Water Potability Prediction")

st.header("Introduction")
st.write("Access to safe drinking-water is essential to health, a basic human right and a component of effective policy for health protection. This is important as a health and development issue at a national, regional and local level. In some regions, it has been shown that investments in water supply and sanitation can yield a net economic benefit, since the reductions in adverse health effects and health care costs outweigh the costs of undertaking the interventions.")

st.header("Attributes Information")

st.subheader("(I) pH Value")
st.write("PH is an important parameter in evaluating the acid–base balance of water. It is also the indicator of acidic or alkaline condition of water status. WHO has recommended maximum permissible limit of pH from 6.5 to 8.5.")

st.subheader("(II) Hardness")
st.write("Hardness is mainly caused by calcium and magnesium salts. These salts are dissolved from geologic deposits through which water travels. The length of time water is in contact with hardness producing material helps determine how much hardness there is in raw water. Hardness was originally defined as the capacity of water to precipitate soap caused by Calcium and Magnesium.")

st.subheader("(III) Solids (Total dissolved solids - TDS)")
st.write("Water has the ability to dissolve a wide range of inorganic and some organic minerals or salts such as potassium, calcium, sodium, bicarbonates, chlorides, magnesium, sulfates etc. These minerals produced un-wanted taste and diluted color in appearance of water. This is the important parameter for the use of water. The water with high TDS value indicates that water is highly mineralized. Desirable limit for TDS is 500 mg/l and maximum limit is 1000 mg/l which prescribed for drinking purpose.")

st.subheader("(IV) Chloramines")
st.write("Chlorine and chloramine are the major disinfectants used in public water systems. Chloramines are most commonly formed when ammonia is added to chlorine to treat drinking water. Chlorine levels up to 4 milligrams per liter (mg/L or 4 parts per million (ppm)) are considered safe in drinking water.")

st.subheader("(V) Sulfate")
st.write("Sulfates are naturally occurring substances that are found in minerals, soil, and rocks. They are present in ambient air, groundwater, plants, and food. The principal commercial use of sulfate is in the chemical industry. Sulfate concentration in seawater is about 2,700 milligrams per liter (mg/L). It ranges from 3 to 30 mg/L in most freshwater supplies, although much higher concentrations (1000 mg/L) are found in some geographic locations.")

st.subheader("(VI) Conductivity")
st.write("Pure water is not a good conductor of electric current rather’s a good insulator. Increase in ions concentration enhances the electrical conductivity of water. Generally, the amount of dissolved solids in water determines the electrical conductivity. Electrical conductivity (EC) actually measures the ionic process of a solution that enables it to transmit current. According to WHO standards, EC value should not exceeded 400 μS/cm.")

st.subheader("(VII) Organic Carbon")
st.write("Total Organic Carbon (TOC) in source waters comes from decaying natural organic matter (NOM) as well as synthetic sources. TOC is a measure of the total amount of carbon in organic compounds in pure water. According to US EPA < 2 mg/L as TOC in treated / drinking water, and < 4 mg/Lit in source water which is use for treatment.")

st.subheader("(VIII) Trihalomethanes")
st.write("THMs are chemicals which may be found in water treated with chlorine. The concentration of THMs in drinking water varies according to the level of organic material in the water, the amount of chlorine required to treat the water, and the temperature of the water that is being treated. THM levels up to 80 ppm is considered safe in drinking water.")

st.subheader("(IX) Turbidity")
st.write("The turbidity of water depends on the quantity of solid matter present in the suspended state. It is a measure of light emitting properties of water and the test is used to indicate the quality of waste discharge with respect to colloidal matter. The mean turbidity value obtained for Wondo Genet Campus (0.98 NTU) is lower than the WHO recommended value of 5.00 NTU.")

st.header("Target")
st.subheader("Potability")
st.write("Indicates if water is safe for human consumption where 'True' means Potable and 'False' means Not potable.")

st.header("Source")
st.write("https://www.kaggle.com/adityakadiwal/water-potability/tasks?taskId=4773")

st.sidebar.header("Water Potability App")
st.sidebar.subheader("Enter the Values")

data = []

def number_input(label, value=0.0):
    return st.sidebar.number_input(label=label, value=value)

pH = number_input('pH Value', 7.08)
data.append(pH)
hardness = number_input('Hardness', 205.09)
data.append(hardness)
tds = number_input('Solids (Total dissolved solids - TDS)', 27064.42)
data.append(tds)
chloramines = number_input('Chloramines', 7.4)
data.append(chloramines)
sulfate = number_input('Sulfate', 304.93)
data.append(sulfate)
conductivity = number_input('Conductivity', 409.5)
data.append(conductivity)
org_carb = number_input('Organic Carbon', 18.52)
data.append(org_carb)
trihm = number_input('Trihalomethanes', 61.49)
data.append(trihm)
turbidity = number_input('Turbidity', 4.6)
data.append(turbidity)

if st.sidebar.button('Predict'):
    data = np.array(data)[None, :]
    num_models = 5
    preds = []
    for model_no in range(num_models):
        model = joblib.load(f'models/rf_{model_no}.bin')
        preds.append(model.predict(data).item())
    
    st.sidebar.subheader("Potable")
    if sum(preds) >= 3:
        st.sidebar.success('True')
    else:
        st.sidebar.warning('False')
