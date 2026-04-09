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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .header-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 40px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.2);
    }
    
    .header-box h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-box p {
        margin: 15px 0 0 0;
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    .price-box-blue {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 35px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .price-box-blue:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(102, 126, 234, 0.4);
    }
    
    .price-box-pink {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 35px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 15px 40px rgba(245, 87, 108, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .price-box-pink:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(245, 87, 108, 0.4);
    }
    
    .price-number {
        font-size: 3rem;
        font-weight: 800;
        margin: 20px 0 0 0;
        text-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stMetric {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .footer-text {
        text-align: center;
        color: rgba(255, 255, 255, 0.8);
        padding: 25px;
        font-size: 0.95rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
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
        
        # Price Display - 2 Column Layout
        col_price1, col_price2 = st.columns(2)
        
        with col_price1:
            st.markdown(f"""
            <div class='price-box-blue'>
                <h3 style='color: white; margin: 0;'>� USD Price</h3>
                <div class='price-number'>{predicted_price:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_price2:
            st.markdown(f"""
            <div class='price-box-pink'>
                <h3 style='color: white; margin: 0;'>₹ INR Price</h3>
                <div class='price-number'>₹{predicted_price * 83:,.0f}</div>
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
