import streamlit as st
from predict import predict_delivery_time

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="centered"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #0f0f0f;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
        color: #f0f0f0;
    }

    .hero {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }

    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #888;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .accent {
        color: #FF6B35;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #FF6B35;
        margin-bottom: 0.5rem;
        margin-top: 1.5rem;
    }

    .result-box {
        background: linear-gradient(135deg, #FF6B35, #FF8C42);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(255, 107, 53, 0.3);
    }

    .result-box h2 {
        color: white;
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .result-time {
        color: white;
        font-size: 4rem;
        font-weight: 700;
        line-height: 1;
    }

    .result-unit {
        color: rgba(255,255,255,0.8);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stNumberInput"] label {
        color: #cccccc !important;
        font-size: 0.9rem !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #FF8C42);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(255, 107, 53, 0.4);
    }

    .tip-box {
        background: rgba(255, 107, 53, 0.08);
        border-left: 3px solid #FF6B35;
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #aaa;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── HERO ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🛵 <span class="accent">Food Delivery</span> Time Predictor</h1>
    <p>Enter order details below to estimate delivery time using XGBoost</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── INPUT FORM ───────────────────────────────────────────────
st.markdown('<div class="section-label">Delivery Person</div>', unsafe_allow_html=True)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", min_value=18, max_value=50, value=28)
    with col2:
        rating = st.slider("Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

st.markdown('<div class="section-label">Order Details</div>', unsafe_allow_html=True)
with st.container():
    col3, col4 = st.columns(2)
    with col3:
        distance = st.number_input("Distance (km)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
        order_type = st.selectbox("Order Type", ["Buffet", "Drinks", "Meal", "Snack"])
    with col4:
        multiple = st.selectbox("Multiple Deliveries?", [0, 1, 2, 3])
        vehicle = st.selectbox("Vehicle Type", ["Bicycle", "Electric Scooter", "Motorcycle", "Scooter"])

st.markdown('<div class="section-label">Conditions</div>', unsafe_allow_html=True)
with st.container():
    col5, col6 = st.columns(2)
    with col5:
        weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy"])
        festival = st.selectbox("Festival?", ["No", "Yes"])
    with col6:
        traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
        city = st.selectbox("City Type", ["Metropolitian", "Urban", "Semi-Urban"])

st.markdown('<div class="section-label">Order Time</div>', unsafe_allow_html=True)
order_hour = st.slider("Hour of Order (24h)", min_value=0, max_value=23, value=19, format="%d:00")

# ── PREDICT BUTTON ───────────────────────────────────────────
if st.button("🛵 Predict Delivery Time"):
    try:
        prediction = predict_delivery_time(
            age, rating, distance, weather, traffic, order_type, vehicle, multiple, festival, city, order_hour
        )

        st.markdown(f"""
        <div class="result-box">
            <h2>Estimated Delivery Time</h2>
            <div class="result-time">{prediction:.0f}</div>
            <div class="result-unit">minutes</div>
        </div>
        """, unsafe_allow_html=True)

        # Contextual tip
        if prediction < 25:
            tip = "⚡ Fast delivery! Low traffic and good conditions."
        elif prediction < 40:
            tip = "🕐 Average delivery time. Typical for urban orders."
        else:
            tip = "⏳ Longer than usual. Likely due to traffic or weather."

        st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)
        
    except FileNotFoundError as e:
        st.error(str(e))

# ── FOOTER NOTE ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "Built with XGBoost · Kaggle Food Delivery Dataset · Streamlit"
    "</p>",
    unsafe_allow_html=True
)
