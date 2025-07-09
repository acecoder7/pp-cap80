import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
import sklearn  # for unpickling

# -- Load models & history once --
@st.cache_resource
def load_models():
    low   = joblib.load("model/model_lowd.pkl")
    med   = joblib.load("model/model_medd.pkl")
    high  = joblib.load("model/model_highd.pkl")
    hist  = pd.read_csv("data/com_mon_final.csv", parse_dates=["Date"])
    precip_med = hist["precip"].median()
    years = range(hist["Year"].min(), hist["Year"].max() + 2)
    ind_hols = holidays.CountryHoliday("IN", years=years)
    return low, med, high, hist, precip_med, ind_hols

lower_model, median_model, upper_model, hist_df, precip_med, ind_hols = load_models()


st.title("Capsicum Price Forcast")
st.write("Enter your feature values in the sidebar to get price forecasts.")


# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Input Features")
    year     = st.slider("Year", 1990, 2030, 2025)
    month    = st.selectbox(
                  "Month",
                  list(range(1,13)),
                  format_func=lambda m: pd.to_datetime(m, format="%m").strftime("%b")
               )
    arrivals = st.number_input("Arrivals (tonnes)",    min_value=0.0, value=1000.0)
    temp_max = st.number_input("Max Temp (°F)",        value=96.00)
    temp_min = st.number_input("Min Temp (°F)",        value=81.0)
    humidity = st.number_input("Humidity (%)",         value=60.0)
    precip   = st.number_input("Precipitation (mm)",   value=2.0)
    solar    = st.number_input("Solar Energy (MJ/m²)", value=15.0)
    uv_index = st.number_input("UV Index",             value=7.0)

# --- Main: Button & Results ---
if st.button("Predict Price"):
        # Assemble new row and concat
    date = pd.to_datetime(dict(year=[year], month=[month], day=[1]))
    new = {
        "Year": year, "Month": month, "Arrivals": arrivals,
        "tempmax": temp_max, "tempmin": temp_min,
        "humidity": humidity, "precip": precip,
        "solarenergy": solar, "uvindex": uv_index,
        "Date": date[0]
    }
    new_row = pd.DataFrame([new])
    df = pd.concat([hist_df, new_row], ignore_index=True)

    # Feature engineering
    df["ModalPrice_lag1"] = df["ModalPrice"].shift(1)
    df["temp_range"]      = df["tempmax"] - df["tempmin"]
    df["humidity_precip"] = df["humidity"] * df["precip"]
    df["humid_uv"]        = df["humidity"] * df["uvindex"]
    for col in ["Arrivals","humidity","precip","tempmax","tempmin","solarenergy","uvindex","temp_range"]:
        df[f"{col}_lag1"] = df[col].shift(1)
    df["month_sin"]       = np.sin(2*np.pi * df["Month"]/12)
    df["month_cos"]       = np.cos(2*np.pi * df["Month"]/12)
    df["rain_flag"]       = (df["precip"] > precip_med).astype(int)
    df["arr_3mo_mean"]    = df["Arrivals"].rolling(3).mean()
    df["arr_3mo_max"]     = df["Arrivals"].rolling(3).max()
    df["sin365_1"]        = np.sin(2*np.pi * df["Date"].dt.dayofyear/365)
    df["cos365_1"]        = np.cos(2*np.pi * df["Date"].dt.dayofyear/365)
    df["sin365_2"]        = np.sin(4*np.pi * df["Date"].dt.dayofyear/365)
    df["cos365_2"]        = np.cos(4*np.pi * df["Date"].dt.dayofyear/365)
    df["price_momentum"]  = df["ModalPrice"] - df["ModalPrice_lag1"]
    df["holiday_flag"]    = df["Date"].apply(lambda d: int(bool(ind_hols.get(d))))

    # Select features & predict
    feat_cols = [
        'Arrivals','Year','Month','humidity','precip','solarenergy','uvindex',
        'ModalPrice_lag1','temp_range','humidity_lag1','temp_range_lag1',
        'month_sin','humid_uv','sin365_1','sin365_2','price_momentum'
    ]
    X_new = df.iloc[[-1]][feat_cols].fillna(0)
    lo  = lower_model.predict(X_new)[0]
    md  = median_model.predict(X_new)[0]
    hi  = upper_model.predict(X_new)[0]

    # Display
    st.subheader("Forecasted Price (Rs./Quintal)")
    st.metric("Lower bound (5th %ile)", f"₹{lo:,.2f}")
    st.metric("Median estimate",         f"₹{md:,.2f}")
    st.metric("Upper bound (95th %ile)", f"₹{hi:,.2f}")
