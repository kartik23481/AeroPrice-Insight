import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys
import streamlit as st
import pandas as pd
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy.stats import percentileofscore
import requests

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.feature_utils import (
    is_north, find_part_of_month, part_of_day,
    make_month_object, remove_duration, have_info,
    duration_category
)

from utils.rbf import RBFPercentileSimilarity

RENDER_API_URL = "https://aeroprice-backend-new.onrender.com/predict"

def call_backend(payload):
    try:
        res = requests.post(RENDER_API_URL, json=payload, timeout=20)
        if res.status_code == 200:
            return res.json().get("predicted_price")
        else:
            st.error(res.text)
            return None
    except Exception as e:
        st.error(str(e))
        return None

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
    <style>
        .stApp {
            background-color: black;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select, .stDateInput>div>div>input {
            background: rgba(20,20,20,0.9);
            color: #00ffcc;
            border: 1px solid #00ffcc;
            border-radius: 8px;
        }
        label {
            color: #00ffcc !important;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(45deg, #00ffcc, #0066ff);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            box-shadow: 0 0 15px #00ffcc;
            transition: transform 0.2s ease-in-out;
            border: none;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 25px #00ffcc;
        }
        .result-box {
            background: rgba(0,255,204,0.1);
            padding: 25px;
            border-radius: 15px;
            border: 2px solid #00ffcc;
            color: #00ffcc;
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            box-shadow: 0 0 30px #00ffcc;
            max-width: 500px;
            margin: auto;
        }
        .highlight-title {
            color: #00ffcc;
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(0, 255, 204, 0.3);
            margin-bottom: 25px;
        }
        .info-box {
            background-color: rgba(0, 102, 255, 0.1);
            border: 1px solid #0066ff;
            border-left: 5px solid #0099FF;
            padding: 15px;
            border-radius: 8px;
            color: #E0E0E0;
            font-family: 'sans serif';
            margin-top: 10px;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_training_data():
    data_path = "data/processed/train_data.csv"
    try:
        train_df = pd.read_csv(data_path)
        train_df['source'] = train_df['source'].str.lower()
        train_df['destination'] = train_df['destination'].str.lower()
        return train_df
    except:
        return pd.DataFrame()

train_df = load_training_data()

def get_part_of_day_label(hour):
    if 0 <= hour < 6:
        return "Night"
    elif 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour <= 23:
        return "Evening"
    return "Unknown"

airline_list = [
    "Indigo", "Air India", "Jet Airways", "Spicejet",
    "Multiple Carriers", "Goair", "Vistara", "Air Asia", "Trujet"
]
source_list = ["delhi", "banglore", "mumbai", "chennai", "kolkata"]
destination_list = ["cochin", "banglore", "delhi", "new delhi", "hyderabad","kolkata"]
additional_info_list = [
    "no info", "in-flight meal not included", "no check-in baggage included",
    "1 long layover", "change airports", "business class",
    "1 short layover", "red-eye flight"
]

def create_deal_gauge(price, historical_prices):
    if historical_prices.empty:
        return None
    perc = percentileofscore(historical_prices, price)
    if perc <= 25:
        deal_text = "Great Deal!"
        color = "#00FF00"
    elif perc <= 50:
        deal_text = "Good Deal"
        color = "#ADFF2F"
    elif perc <= 75:
        deal_text = "Average Price"
        color = "#FFA500"
    else:
        deal_text = "Expensive"
        color = "#FF4B4B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=price,
        number={'prefix': "₹", 'font': {'size': 30}},
        delta={'reference': historical_prices.mean(), 'relative': False, 'valueformat': '.0f'},
        title={'text': f"<b>{deal_text}</b> (vs. avg. ₹{historical_prices.mean():,.0f})", 'font': {'size': 20, 'color': color}},
        gauge={
            'axis': {'range': [historical_prices.min(), historical_prices.max()], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#666"
        }
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color="white",
        height=300
    )
    return fig

st.markdown("<h1 style='text-align: center; color: #00ffcc;'>✈️ Flight Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: white;'>Predict your flight ticket prices instantly and plan smarter.</p>", unsafe_allow_html=True)
st.divider()

with st.form("flight_input_form1"):
    st.markdown("<div class='highlight-title'>📝 Flight Price Details</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        airline = st.selectbox("Airline", airline_list)
    with col2:
        source = st.selectbox("Source", source_list)
    with col3:
        destination = st.selectbox("Destination", destination_list)
    col4, col5, col6 = st.columns(3)
    with col4:
        duration = st.number_input("Duration (minutes)", min_value=0)
    with col5:
        total_stops = st.number_input("Total Stops", min_value=0)
    with col6:
        additional_info = st.selectbox("Additional Info", additional_info_list)
    col7, col8 = st.columns(2)
    with col7:
        date_input = st.date_input("Date of Journey", value=datetime(2019, 6, 1))
    with col8:
        dept_time_hour = st.number_input("Departure Hour (0-23)", min_value=0, max_value=23)
    submitted = st.form_submit_button("🔮 Predict Flight Price")

if submitted:
    payload = {
        "airline": airline,
        "source": source,
        "destination": destination,
        "duration": int(duration),
        "total_stops": int(total_stops),
        "additional_info": additional_info,
        "dep_time_hour": int(dept_time_hour),
        "date": str(date_input)
    }

    prediction = call_backend(payload)
    if prediction is None:
        st.stop()

    result_placeholder = st.empty()
    for i in range(0, int(prediction)+1, max(1, int(prediction)//100)):
        result_placeholder.markdown(f"<div class='result-box'>💰 Predicted Flight Price: ₹{i:,.2f}</div>", unsafe_allow_html=True)
        time.sleep(0.01)
    result_placeholder.markdown(f"<div class='result-box'>💰 Predicted Flight Price: ₹{prediction:,.2f}</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if not train_df.empty:
        route_prices = train_df[
            (train_df['source'] == source.lower()) &
            (train_df['destination'] == destination.lower())
        ]['price']
        if not route_prices.empty:
            gauge_fig = create_deal_gauge(prediction, route_prices)
            if gauge_fig:
                st.plotly_chart(gauge_fig, use_container_width=True)

    highlight_color = "#0099FF"
    default_color = "#00BFA6"
    st.markdown("<div class='highlight-title'>📊 Compare Airlines for Same Journey</div>", unsafe_allow_html=True)
    airline_predictions = []
    for air in airline_list:
        temp_payload = payload.copy()
        temp_payload["airline"] = air
        pred = call_backend(temp_payload)
        airline_predictions.append((air, pred))
    airline_df = pd.DataFrame(airline_predictions, columns=['Airline', 'Predicted Price'])
    fig = px.bar(airline_df, x='Airline', y='Predicted Price')
    st.plotly_chart(fig, use_container_width=True)
    best_airline = airline_df.loc[airline_df['Predicted Price'].idxmin()]
    st.markdown(f"""
    <div style="
        background: rgba(0,255,204,0.1); border: 1px solid #00ffcc; 
        border-radius: 10px; padding: 15px; text-align: center; 
        color: #00ccff; font-size: 18px; font-weight: bold; 
        margin-top: 15px; box-shadow: 0 0 15px rgba(0,255,204,0.5);
    ">
        ✅ Best option for your journey: <b>{best_airline['Airline']}</b> at <b>₹{best_airline['Predicted Price']:,.2f}</b>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("<div class='highlight-title'>💡 Deeper Price Insights</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "☀️ Price by Time",
        "🗓️ Flexible Dates",
        "🚦 Price by Stops",
        "🗺️ Flexible Sources",
        "🔍 Prediction Context"
    ])

    with tab1:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Price by Departure Time</h3>", unsafe_allow_html=True)
        time_buckets = [(6, "Morning"), (12, "Afternoon"), (18, "Evening"), (22, "Night")]
        time_predictions = []
        for hour, label in time_buckets:
            temp_payload = payload.copy()
            temp_payload["dep_time_hour"] = hour
            pred = call_backend(temp_payload)
            time_predictions.append({"Part of Day": label, "Predicted Price": pred})
        day_part_df = pd.DataFrame(time_predictions)
        fig2 = px.bar(day_part_df, x='Part of Day', y='Predicted Price')
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("<div class='info-box'>Shows estimates for your selected airline and route at different times of the day.</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Prices for Nearby Dates</h3>", unsafe_allow_html=True)
        date_predictions = []
        for i in range(-3, 4):
            new_date = date_input + pd.Timedelta(days=i)
            temp_payload = payload.copy()
            temp_payload["date"] = str(new_date)
            pred = call_backend(temp_payload)
            date_predictions.append({"Date": new_date.strftime("%Y-%m-%d"), "Predicted Price": pred})
        date_df = pd.DataFrame(date_predictions)
        fig3 = px.line(date_df, x='Date', y='Predicted Price', markers=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("<div class='info-box'>Shows estimates for your flight on surrounding dates. The red dashed line is your selected date.</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Price by Number of Stops</h3>", unsafe_allow_html=True)
        stops_predictions = []
        for s in [0, 1, 2]:
            temp_payload = payload.copy()
            temp_payload["total_stops"] = s
            pred = call_backend(temp_payload)
            stops_predictions.append({"Stops": f"{s} Stop(s)", "Predicted Price": pred})
        stops_df = pd.DataFrame(stops_predictions)
        fig4 = px.bar(stops_df, x='Stops', y='Predicted Price')
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("<div class='info-box'>Shows price estimates for your selected airline and route, but with different numbers of stops.</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Check Prices from Other Sources</h3>", unsafe_allow_html=True)
        source_predictions = []
        for src in source_list:
            temp_payload = payload.copy()
            temp_payload["source"] = src
            pred = call_backend(temp_payload)
            source_predictions.append({"Source": src.title(), "Predicted Price": pred})
        source_df = pd.DataFrame(source_predictions)
        fig5 = px.bar(source_df, x='Source', y='Predicted Price')
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("<div class='info-box'>Shows estimates for flying to your chosen destination from other source cities on the same day.</div>", unsafe_allow_html=True)

    with tab5:
        st.markdown("<h3 style='text-align: center; color: #00ffcc;'>Validating Your Prediction</h3>", unsafe_allow_html=True)
        if not train_df.empty:
            st.markdown("<h4 style='text-align: center; color: #00ffcc;'>1. Analysis for Your Route</h4>", unsafe_allow_html=True)
            user_month = date_input.month
            month_name = date_input.strftime("%B")
            filter_by_month = st.toggle(f"Show historical data for {month_name} only", value=False)
            route_data = train_df[(train_df['source'] == source.lower()) & (train_df['destination'] == destination.lower())]
            if filter_by_month:
                route_data = route_data[route_data['dtoj_month'] == user_month]
                plot_title = f"Historical Data: {source.title()} to {destination.title()} (in {month_name})"
            else:
                plot_title = f"Historical Data: {source.title()} to {destination.title()} (All Months)"
            if not route_data.empty:
                route_data['total_stops'] = route_data['total_stops'].astype(str)
                fig6 = px.scatter(route_data, x='duration', y='price', color='total_stops', opacity=0.7, title=plot_title)
                fig6.add_trace(go.Scatter(x=[duration], y=[prediction], mode='markers', marker=dict(color='#00FF00', size=15, line=dict(color='white', width=2), symbol='star')))
                st.plotly_chart(fig6, use_container_width=True)
                st.markdown("<div class='info-box'>This plot shows all historical flights on your route. Your predicted flight (the green star) is plotted to see how it compares. Use the toggle to see data for a specific month.</div>", unsafe_allow_html=True)

            st.markdown("<h4 style='text-align: center; color: #00ffcc; margin-top: 30px;'>2. Analysis for Similar Duration Flights (Any Route)</h4>", unsafe_allow_html=True)
            duration_margin = 50
            duration_min = duration - duration_margin
            duration_max = duration + duration_margin
            similar_flights_data = train_df[(train_df['duration'] >= duration_min) & (train_df['duration'] <= duration_max)]
            if not similar_flights_data.empty:
                similar_flights_data['total_stops'] = similar_flights_data['total_stops'].astype(str)
                fig7 = px.scatter(similar_flights_data, x='duration', y='price', color='total_stops', opacity=0.5, title=f"Price vs. Duration for Flights between {duration_min}-{duration_max} min (Any Route)")
                fig7.add_trace(go.Scatter(x=[duration], y=[prediction], mode='markers', marker=dict(color='#00FF00', size=15, line=dict(color='white', width=2), symbol='star')))
                st.plotly_chart(fig7, use_container_width=True)
                st.markdown("<div class='info-box'>This chart shows all historical flights (regardless of route) that have a similar duration to yours. Your flight is the green star.</div>", unsafe_allow_html=True)
