import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Laptop Price Prediction", layout="wide")

# Load data and model
data_dict = pickle.load(open('clean_df.pkl', 'rb'))
data = pd.DataFrame(data_dict)
pipe = pickle.load(open('pipe_xgb.pkl', 'rb'))

st.title('💻 LAPTOP PRICE PREDICTION APP')

# =============== SIDEBAR INPUTS ===============
st.sidebar.header('🎛️ SPECIFICATIONS')

company = st.sidebar.selectbox('COMPANY', data['Company'].unique())

typename = st.sidebar.selectbox('TYPE', data['TypeName'].unique())

ram_options = sorted(data['Ram'].unique())
ram = st.sidebar.select_slider('RAM (GB)', options=ram_options, value=8)

opsys = st.sidebar.selectbox('OPERATING SYSTEM', data['OpSys'].unique())

ips = st.sidebar.radio('IPS PANEL', ['YES', 'NO'], horizontal=True)
ips_panel = 1 if ips == 'YES' else 0

cpu_category = st.sidebar.selectbox('CPU CATEGORY', data['CPU_Category'].unique())

ppi = st.sidebar.slider('PPI (Pixels Per Inch)', 50.0, 500.0, 100.0, step=5.0)

gpu_brand = st.sidebar.selectbox('GPU BRAND', data['GPU_Brand'].unique())

ssd_options = sorted(data['ssd'].unique())
ssd = st.sidebar.select_slider('SSD (GB)', options=ssd_options, value=256)

# CPU FLAGS
i3 = 1 if cpu_category == 'Intel Core i3' else 0
i7 = 1 if cpu_category == 'Intel Core i7' else 0

# =============== MAIN CONTENT ===============
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader('📊 SELECTED CONFIGURATION')
    
    # Display specs in a nice format
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.metric("🏢 Company", company)
        st.metric("💾 SSD", f"{ssd} GB")
    
    with col_b:
        st.metric("🖥️ Type", typename)
        st.metric("📺 GPU", gpu_brand)
    
    with col_c:
        st.metric("🧠 RAM", f"{ram} GB")
        st.metric("📍 PPI", f"{ppi:.1f}")

with col2:
    st.subheader('ℹ️ INFO')
    st.info(f"""
    **OS**: {opsys}
    **CPU**: {cpu_category}
    **IPS**: {ips}
    **i3**: {'✅ YES' if i3 == 1 else '❌ NO'}
    **i7**: {'✅ YES' if i7 == 1 else '❌ NO'}
    """)

st.markdown('---')

# =============== PREDICTION ===============
if st.button('🔮 PREDICT PRICE', use_container_width=True, key='predict_btn'):
    
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
        # Make prediction
        log_price = pipe.predict(query_df)[0]
        predicted_price = np.exp(log_price)
        
        # Display result
        st.success('✅ PREDICTION SUCCESSFUL!')
        
        # Big price display
        col_price_left, col_price_right = st.columns(2)
        
        with col_price_left:
            st.metric("💰 PREDICTED PRICE in Rs", f"{predicted_price:}")
        
    
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

