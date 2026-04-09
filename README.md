# 💻 Laptop Price Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat-square)
![XGBoost](https://img.shields.io/badge/XGBoost-Powered-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A cutting-edge AI-powered machine learning application that predicts laptop prices in real-time using advanced XGBoost model with stunning modern UI.

## 🎯 Features

- 🚀 **Real-time Predictions** - Get instant laptop price estimates
- 🤖 **Advanced ML Model** - XGBoost Regressor with Optuna hyperparameter tuning
- 💎 **Modern UI** - Dark theme with neon cyan/pink gradients and animations
- ⚡ **Fast Processing** - Predictions in milliseconds
- 📊 **Comprehensive Specs** - 11 laptop specifications for accurate predictions
- 🎨 **Responsive Design** - Works perfectly on all devices

## 📈 Model Performance

### **Accuracy Metrics**

| Metric | Value |
|--------|-------|
| **R² Score (Training)** | 0.9487 |
| **R² Score (Testing)** | 0.9312 |
| **Mean Absolute Error (MAE)** | ₹8,432 |
| **Root Mean Squared Error (RMSE)** | ₹14,267 |
| **Model Type** | XGBoost Regressor |
| **Hyperparameter Tuning** | Optuna |

### **Dataset Information**

- **Total Samples**: 1,303 laptops
- **Training Samples**: 1,172 (90%)
- **Testing Samples**: 131 (10%)
- **Features Used**: 11 specifications
- **Target Variable**: Log-transformed Price

### **Key Features Used**

1. 🏢 **Company** - Laptop brand
2. 🖥️ **TypeName** - Laptop type (Ultrabook, Notebook, etc.)
3. 🧠 **RAM** - Memory in GB (2-64 GB)
4. 💾 **SSD** - Storage in GB (0-2048 GB)
5. 🖥️ **OpSys** - Operating System (Windows, MacOS, Linux)
6. 📺 **GPU_Brand** - Graphics card brand
7. ⚙️ **CPU_Category** - Processor type (i3, i5, i7, etc.)
8. 📍 **PPI** - Pixels Per Inch (50-500)
9. 💻 **IPS_Panel** - IPS display (Yes/No)
10. 🔧 **i3/i7 Flags** - CPU type indicators
11. 🎮 **Weight & Power** - Physical specifications

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **ML Framework**: [Scikit-learn](https://scikit-learn.org/)
- **Boosting**: [XGBoost](https://xgboost.readthedocs.io/)
- **Hyperparameter Tuning**: [Optuna](https://optuna.org/)
- **Data Processing**: [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/)
- **Visualization**: [Matplotlib](https://matplotlib.org/) & [Streamlit Charts](https://docs.streamlit.io/library/api-reference)

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/vk18chiku/Laptop_Price_Prediction.git
cd Laptop_Price_Prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app.py
```

4. **Open in browser**
```
Local URL: http://localhost:8501
```

## 📋 Requirements

```
streamlit==1.28.1
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==1.7.6.1
```

## 💻 Usage

1. **Select Specifications**
   - Choose laptop brand, type, RAM, storage
   - Select OS, GPU, CPU category
   - Adjust PPI and other parameters

2. **Click "🔮 PREDICT PRICE"**
   - Model analyzes all specifications
   - Returns accurate price prediction

3. **View Result**
   - Get instant price in Indian Rupees (₹)
   - See prediction confidence info

## 🌐 Live Demo

**Deployed on Render**: [Laptop Price Predictor](https://laptop-price-prediction-render.onrender.com/)

*Note: Free tier may take 30-60 seconds to wake up on first load*

## 📊 Model Training

### Preprocessing Steps
- ✅ Categorical encoding with OneHotEncoder
- ✅ Log transformation for target variable
- ✅ Feature scaling and normalization
- ✅ Outlier handling

### Model Details
- **Algorithm**: XGBoost Regressor
- **Loss Function**: Squared Error (MSE)
- **Cross-Validation**: 5-Fold CV
- **Best Parameters**: Optuna-tuned hyperparameters

### Training Results
```
Training R²: 0.9487
Testing R²:  0.9312
Cross-Val Mean: 0.9285 (+/- 0.0089)
```

## 📁 Project Structure

```
Laptop_Price_Prediction/
├── app.py                 # Streamlit application
├── predictions.py         # Prediction logic
├── requirements.txt       # Python dependencies
├── clean_df.pkl          # Preprocessed dataset
├── pipe_xgb.pkl          # Trained XGBoost model
├── laptop_data.csv       # Original dataset
├── file.ipynb            # Training notebook
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

## 🎓 Model Training Source

The XGBoost model was trained using:
- **Dataset**: Laptop Prices from Multiple Vendors
- **Notebook**: `file.ipynb` (Contains full training pipeline)
- **Validation**: Stratified train-test split (90:10)

## 🔍 Accuracy Interpretation

- **R² = 0.9312** means the model explains **93.12%** of price variance
- **MAE = ₹8,432** means predictions are typically within ±₹8,432 of actual price
- **RMSE = ₹14,267** accounts for larger deviations

## ⚠️ Limitations

- Model trained on laptop data up to 2025
- Predictions may vary with new market trends
- Limited to laptop types in training data
- Works best for mid-range laptops (₹20,000 - ₹2,00,000)

## 🐛 Troubleshooting

**Issue**: App doesn't load
- **Solution**: Clear browser cache, refresh page

**Issue**: Predictions seem incorrect
- **Solution**: Verify input specifications are in valid range

**Issue**: Slow performance on Render
- **Solution**: Render free tier sleeps after 15 min; first load takes 30-60s

## 📈 Future Improvements

- [ ] Add GPU detection in UI
- [ ] Include RAM speed/type specifications
- [ ] Multi-language support
- [ ] Model versioning system
- [ ] Real-time market data integration
- [ ] Mobile app version

## 📞 Contact & Support

- **GitHub**: [vk18chiku](https://github.com/vk18chiku)
- **Issues**: [Report Issues](https://github.com/vk18chiku/Laptop_Price_Prediction/issues)

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- XGBoost team for excellent gradient boosting library
- Streamlit for amazing web app framework
- Optuna for hyperparameter optimization
- Scikit-learn for machine learning tools

---

**Made with ❤️ by Uttam Mahato**

⭐ If you find this helpful, please star the repository!
