import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests


# ==========================================
# 1. ROBUST EXCHANGE RATE FUNCTION
# ==========================================
def get_live_exchange_rate(base_currency="INR", target_currency="NGN"):
    """
    Tries to fetch the exchange rate from two different free sources.
    Source 1: Frankfurter API (Open Source, very reliable)
    Source 2: Open-ER API (Backup)
    Fallback: Static rate if both fail.
    """

    # Define a user-agent to avoid being blocked as a "bot"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    # --- ATTEMPT 1: FRANKFURTER API ---
    try:
        url = f"https://api.frankfurter.app/latest?from={base_currency}&to={target_currency}"
        response = requests.get(url, headers=headers, timeout=3)

        if response.status_code == 200:
            data = response.json()
            # Frankfurter returns format: {"rates": {"NGN": 18.5}}
            if "rates" in data and target_currency in data["rates"]:
                return data["rates"][target_currency]
    except Exception:
        pass  # If Source 1 fails, silently move to Source 2

    # --- ATTEMPT 2: OPEN EXCHANGE RATES API ---
    try:
        url = f"https://open.er-api.com/v6/latest/{base_currency}"
        response = requests.get(url, headers=headers, timeout=3)

        if response.status_code == 200:
            data = response.json()
            if "rates" in data and target_currency in data["rates"]:
                return data["rates"][target_currency]
    except Exception:
        pass  # If Source 2 fails, move to fallback

    # --- FALLBACK: STATIC RATE ---
    # Used only if both APIs fail (e.g., no internet on server)
    return 18.5


# ==========================================
# 2. LOAD MODEL
# ==========================================
try:
    with open('laptop_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠️ Error: Model file not found. Make sure 'laptop_price_model.pkl' is in the same folder.")
    st.stop()

# ==========================================
# 3. APP INTERFACE
# ==========================================
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")

st.title("💻 Laptop Price Predictor")
st.markdown("Predict laptop prices and convert them to **Naira (₦)** automatically.")
st.write("---")

col1, col2 = st.columns(2)

with col1:
    company = st.selectbox('Brand',
                           ['Apple', 'HP', 'Dell', 'Lenovo', 'Asus', 'Acer', 'MSI', 'Toshiba', 'Samsung', 'Razer',
                            'Mediacom', 'Microsoft', 'Xiaomi', 'Vero', 'Chuwi', 'Google', 'Fujitsu', 'Huawei', 'LG'])
    type_name = st.selectbox('Type',
                             ['Notebook', 'Ultrabook', 'Gaming', '2 in 1 Convertible', 'Workstation', 'Netbook'])
    ram = st.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.01)
    touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
    ips = st.selectbox('IPS Panel', ['No', 'Yes'])
    os = st.selectbox('OS', ['Windows', 'Mac', 'Linux/Chrome', 'Other'])

with col2:
    screen_size = st.number_input('Screen Size (Inches)', min_value=10.0, max_value=20.0, value=15.6, step=0.1)
    resolution = st.selectbox('Screen Resolution',
                              ['1920x1080', '1366x768', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',
                               '2304x1440'])
    cpu_speed = st.number_input('CPU Clock Speed (GHz)', min_value=0.5, max_value=5.0, value=2.5, step=0.1)
    cpu_brand = st.selectbox('CPU Family', ['Core i7', 'Core i5', 'Core i3', 'Other', 'Pentium'])
    hdd = st.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = st.selectbox('SSD (GB)', [0, 128, 256, 512, 1024])

    st.markdown("**GPU Configuration**")
    gpu_brand_input = st.selectbox('GPU Brand', ['Intel', 'Nvidia', 'AMD'])
    gpu_dedicated_check = st.radio("Is it a Dedicated GPU?", ['No', 'Yes'])

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
if st.button('Predict Price 🚀', use_container_width=True):

    # 1. Feature Engineering (Pixels, Touchscreen, etc.)
    try:
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        total_pixels = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size
    except:
        total_pixels = 0

    type_map = {'Netbook': 0, 'Notebook': 1, '2 in 1 Convertible': 2, 'Ultrabook': 3, 'Gaming': 4, 'Workstation': 5}
    type_score = type_map[type_name]

    touch_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0
    fhd_val = 1 if '1920x1080' in resolution else 0
    gpu_dedicated_val = 1 if gpu_dedicated_check == 'Yes' else 0

    # 2. Build Data Dictionary
    data = {
        'Ram': ram, 'Weight': weight, 'Total_Pixels': total_pixels, 'IPS_Panel': ips_val,
        'Touchscreen': touch_val, 'Full_HD': fhd_val, 'Clock_Speed_GHz': cpu_speed,
        'SSD_GB': ssd, 'HDD_GB': hdd, 'gpu_dedicated': gpu_dedicated_val, 'type_score': type_score,
        'Company_Apple': 0, 'Company_Asus': 0, 'Company_Chuwi': 0, 'Company_Dell': 0,
        'Company_Fujitsu': 0, 'Company_Google': 0, 'Company_HP': 0, 'Company_Huawei': 0,
        'Company_LG': 0, 'Company_Lenovo': 0, 'Company_MSI': 0, 'Company_Mediacom': 0,
        'Company_Microsoft': 0, 'Company_Razer': 0, 'Company_Samsung': 0,
        'Company_Toshiba': 0, 'Company_Vero': 0, 'Company_Xiaomi': 0,
        'Cpu_Family_Core i3': 0, 'Cpu_Family_Core i5': 0, 'Cpu_Family_Core i7': 0,
        'Cpu_Family_Other': 0, 'Cpu_Family_Pentium': 0,
        'gpu_Intel': 0, 'gpu_Nvidia': 0,
        'OpSys_Simplified_Other': 0, 'OpSys_Simplified_Windows': 0
    }

    # 3. Set One-Hot Encoded Values
    if company != 'Acer': data[f'Company_{company}'] = 1
    if cpu_brand != 'Celeron': data[f'Cpu_Family_{cpu_brand}'] = 1
    if gpu_brand_input == 'Nvidia':
        data['gpu_Nvidia'] = 1
    elif gpu_brand_input == 'Intel':
        data['gpu_Intel'] = 1
    if os == 'Windows':
        data['OpSys_Simplified_Windows'] = 1
    elif os == 'Other' or os == 'Mac':
        data['OpSys_Simplified_Other'] = 1

    # 4. Create DataFrame & Align Columns
    df_input = pd.DataFrame([data])
    df_input = df_input.reindex(columns=list(data.keys()), fill_value=0)

    # 5. Predict & Convert
    try:
        # Get Model Prediction (Log Price)
        log_prediction = model.predict(df_input)
        price_inr = np.expm1(log_prediction)[0]

        # Get Live Exchange Rate (Try Multiple Sources)
        exchange_rate = get_live_exchange_rate("INR", "NGN")

        price_ngn = price_inr * exchange_rate

        st.success(f"### 💰 Price: ₹{price_inr:,.2f} INR")
        st.info(f"### 🇳🇬 Price: ₦{price_ngn:,.2f} NGN")

        if exchange_rate == 18.5:
            st.warning(f"⚠️ Note: Could not fetch live rate. Using fallback: 1 INR = ₦{exchange_rate}")
        else:
            st.caption(f"✅ Live Rate Fetched: 1 INR = ₦{exchange_rate:,.2f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.write("---")
st.warning("**Note:** Prices are estimates based on historical data and current exchange rates.")