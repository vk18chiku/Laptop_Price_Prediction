import streamlit as st
import pickle
import pandas as pd
import numpy as np

# =============== PAGE CONFIG ===============
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Beautiful Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 100%);
        background-attachment: fixed;
    }
    
    [data-testid="stSidebar"] {
        background: #151d2e !important;
        border-right: 2px solid #00d4ff !important;
    }
    
    .header-box {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 50%, #ff006e 100%);
        padding: 50px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 20px 60px rgba(0, 212, 255, 0.4),
                    0 0 80px rgba(255, 0, 110, 0.2);
        border: 2px solid rgba(0, 212, 255, 0.5);
        position: relative;
        overflow: hidden;
    }
    
    .header-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .header-box h1 {
        margin: 0;
        font-size: 3.2rem;
        font-weight: 800;
        position: relative;
        z-index: 1;
        text-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        letter-spacing: -1px;
    }
    
    .header-box p {
        margin: 18px 0 0 0;
        font-size: 1.15rem;
        font-weight: 500;
        opacity: 0.95;
        position: relative;
        z-index: 1;
        letter-spacing: 0.5px;
    }
    
    .price-box-inr {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        padding: 50px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 30px 0;
        box-shadow: 0 20px 60px rgba(0, 212, 255, 0.35);
        border: 2px solid rgba(0, 212, 255, 0.5);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .price-box-inr::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: glow 6s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { transform: translate(0, 0); }
        50% { transform: translate(30px, 30px); }
    }
    
    .price-box-inr:hover {
        transform: translateY(-12px);
        box-shadow: 0 30px 80px rgba(0, 212, 255, 0.5);
        border-color: rgba(0, 212, 255, 0.8);
    }
    
    .price-label {
        font-size: 1.1rem;
        font-weight: 600;
        opacity: 0.95;
        letter-spacing: 1px;
        text-transform: uppercase;
        position: relative;
        z-index: 2;
    }
    
    .price-number {
        font-size: 3.8rem;
        font-weight: 800;
        margin: 25px 0 0 0;
        letter-spacing: -2px;
        position: relative;
        z-index: 2;
        text-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
    }
    
    .stMetric {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(255, 0, 110, 0.05) 100%);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        border-color: rgba(0, 212, 255, 0.6);
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.15) 0%, rgba(255, 0, 110, 0.1) 100%);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
    }
    
    .footer-text {
        text-align: center;
        color: rgba(0, 212, 255, 0.8);
        padding: 35px;
        font-size: 1rem;
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(255, 0, 110, 0.04) 100%);
        border-radius: 15px;
        border: 2px solid rgba(0, 212, 255, 0.3);
        margin-top: 40px;
        font-weight: 500;
    }
    
    h2, h3 {
        color: #00d4ff;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    [data-testid="stButton"] > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 15px 40px !important;
        box-shadow: 0 12px 30px rgba(0, 212, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stButton"] > button:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 18px 50px rgba(0, 212, 255, 0.5) !important;
    }
</style>
""", unsafe_allow_html=True)

# =============== LOAD DATA & MODEL ===============
@st.cache_resource
def load_models():
    data_dict = pickle.load(open('clean_df.pkl', 'rb'))
    data = pd.DataFrame(data_dict)
    pipe = pickle.load(open('pipe_xgb.pkl', 'rb'))
    return data, pipe

data, pipe = load_models()

# =============== HEADER SECTION ===============
st.markdown("""
<div class='header-box'>
    <h1>💻 LAPTOP PRICE PREDICTOR</h1>
    <p>🚀 AI-Powered XGBoost Model | Real-time Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# Model Info
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("📊 Dataset", "1300+ Laptops", None)
with col_info2:
    st.metric("🤖 Model", "XGBoost", None)
with col_info3:
    st.metric("⚡ Speed", "Instant", None)

st.markdown("---")

# =============== SIDEBAR SPECIFICATIONS ===============
with st.sidebar:
    st.markdown("### 🎛️ LAPTOP SPECIFICATIONS")
    st.markdown("---")
    
    # Company & Type Row
    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox('🏢 Company', data['Company'].unique(), key='company')
    with col2:
        typename = st.selectbox('🖥️ Type', data['TypeName'].unique(), key='type')
    
    # RAM & OS Row
    col3, col4 = st.columns(2)
    with col3:
        ram_options = sorted(data['Ram'].unique())
        ram = st.select_slider('🧠 RAM (GB)', options=ram_options, value=8)
    with col4:
        opsys = st.selectbox('💾 OS', data['OpSys'].unique(), key='os')
    
    # IPS & CPU
    col5, col6 = st.columns(2)
    with col5:
        ips = st.radio('📱 IPS Panel', ['YES', 'NO'], horizontal=True)
        ips_panel = 1 if ips == 'YES' else 0
    with col6:
        cpu_category = st.selectbox('⚙️ CPU', data['CPU_Category'].unique(), key='cpu')
    
    st.markdown("---")
    st.markdown("### 📡 ADVANCED SETTINGS")
    
    # Advanced Settings
    ppi = st.slider('📍 PPI (Pixels/Inch)', 50.0, 500.0, 100.0, step=5.0)
    
    col7, col8 = st.columns(2)
    with col7:
        gpu_brand = st.selectbox('📺 GPU Brand', data['GPU_Brand'].unique(), key='gpu')
    with col8:
        ssd_options = sorted(data['ssd'].unique())
        ssd = st.select_slider('💾 SSD (GB)', options=ssd_options, value=256)

# CPU FLAGS
i3 = 1 if cpu_category == 'Intel Core i3' else 0
i7 = 1 if cpu_category == 'Intel Core i7' else 0

# =============== MAIN DISPLAY ===============
st.markdown("### 📊 SELECTED CONFIGURATION")

# 3-Column Specs Display
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("🏢 Company", company)
    st.metric("🖥️ Type", typename)
    st.metric("🧠 RAM", f"{ram} GB")

with col_b:
    st.metric("💾 SSD", f"{ssd} GB")
    st.metric("📍 PPI", f"{ppi:.0f}")
    st.metric("📺 GPU", gpu_brand)

with col_c:
    st.metric("💻 OS", opsys)
    st.metric("⚙️ CPU", cpu_category.split()[-1])
    st.metric("IPS", "✅ YES" if ips == 'YES' else "❌ NO")

st.markdown("---")

# =============== PREDICTION SECTION ===============
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button(
        '🔮 PREDICT PRICE',
        use_container_width=True,
        key='predict_btn',
        type='primary'
    )

if predict_button:
    # Create input dataframe
    query_df = pd.DataFrame({
        'Company': [company],
        'TypeName': [typename],
        'Ram': [ram],
        'OpSys': [opsys],
        'IPS_Panel': [ips_panel],
        'CPU_Category': [cpu_category],
        'i3': [i3],
        'i7': [i7],
        'PPI': [ppi],
        'GPU_Brand': [gpu_brand],
        'ssd': [ssd]
    })
    
    try:
        # Show loading spinner
        with st.spinner('🔍 Analyzing specifications...'):
            log_price = pipe.predict(query_df)[0]
            predicted_price = np.exp(log_price)
        
        # Success message
        st.success('✅ PREDICTION SUCCESSFUL!', icon="✅")
        
        st.markdown("")
        
        # Price Display - Full Width
        st.markdown(f"""
        <div class='price-box-inr'>
            <div class='price-label'>💰 Estimated Price</div>
            <div class='price-number'>₹ {predicted_price:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Additional Info
        st.info(
            "✨ This prediction is powered by XGBoost ML model trained on 1,303 real laptop specifications with high accuracy!",
            icon="ℹ️"
        )
        
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}", icon="❌")
        st.warning("Please check your inputs and try again.", icon="⚠️")

# =============== FOOTER ===============
st.markdown("---")
st.markdown("""
<div class='footer-text'>
    <p><b>🎓 Model Information:</b></p>
    <p>XGBoost Regressor | Optuna Hyperparameter Tuning | Log-Transformed Target</p>
    <p><b>📈 Features:</b> 11 Laptop Specifications | <b>📊 Training Data:</b> 1,303 Laptops</p>
    <p style='margin-top: 20px; font-size: 0.85rem;'>Built with ❤️ using Streamlit | Data Science ML Project</p>
</div>
""", unsafe_allow_html=True)
