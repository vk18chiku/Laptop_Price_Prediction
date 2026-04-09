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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background-color: #f8fafc;
        background-attachment: fixed;
    }
    
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    .header-box {
        background: linear-gradient(135deg, #3b82f6 0%, #06b6d4 100%);
        padding: 50px 40px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 35px;
        box-shadow: 0 10px 30px rgba(59, 130, 246, 0.15);
    }
    
    .header-box h1 {
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .header-box p {
        margin: 15px 0 0 0;
        font-size: 1.1rem;
        font-weight: 400;
        opacity: 0.95;
    }
    
    .price-box-usd {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 40px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 12px 35px rgba(59, 130, 246, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .price-box-usd:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 45px rgba(59, 130, 246, 0.3);
    }
    
    .price-box-inr {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        padding: 40px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 12px 35px rgba(6, 182, 212, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .price-box-inr:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 45px rgba(6, 182, 212, 0.3);
    }
    
    .price-label {
        font-size: 0.95rem;
        font-weight: 500;
        opacity: 0.9;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    
    .price-number {
        font-size: 3.2rem;
        font-weight: 800;
        margin: 18px 0 0 0;
        letter-spacing: -1px;
    }
    
    .stMetric {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .footer-text {
        text-align: center;
        color: #64748b;
        padding: 30px;
        font-size: 0.95rem;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-top: 30px;
    }
    
    h2, h3 {
        color: #1e293b;
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
    <p>🤖 AI-Powered XGBoost Model | Real-time Price Prediction</p>
</div>
""", unsafe_allow_html=True)

# Model Info
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("📊 Dataset", "1303 Laptops", None)
with col_info2:
    st.metric("🎯 Model", "XGBoost", None)
with col_info3:
    st.metric("🔧 Optimization", "Optuna", None)

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
