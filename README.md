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

### **Model Comparison - R² Scores**

| Model | Training R² | Testing R² | Status |
|-------|-------------|-----------|--------|
| **XGBoost** ⭐ | 0.9240 | **0.8773** | ✅ **SELECTED** |
| Random Forest | 0.9623 | 0.8685 | Good |
| Decision Tree | 0.9733 | 0.8437 | Fair |
| Linear Regression | 0.8225 | 0.8268 | Baseline |

**Best Model**: XGBoost (Selected for production)

### **Dataset Information**

- **Total Samples**: 1,303 laptops
- **Training Samples**: 1,172 (90%)
- **Testing Samples**: 131 (10%)
- **Features Used**: 11 specifications
- **Target Variable**: Log-transformed Price

### **Feature Engineering Pipeline**

1. **Data Cleaning**
   - Remove duplicate entries
   - Drop unnecessary index columns
   - Handle missing values

2. **Memory & Weight Processing**
   - Extract numeric values from RAM (GB)
   - Extract numeric values from Weight (kg)
   - Convert to appropriate data types

3. **Display Features**
   - Extract TouchScreen indicator from ScreenResolution
   - Extract IPS Panel indicator from ScreenResolution
   - Calculate PPI (Pixels Per Inch) = √(Width² + Height²) / Inches

4. **CPU Feature Engineering**
   - Categorize processors into: Intel i3, i5, i7, Other Intel, AMD
   - Create binary flags for i3, i5, i7 processors
   - Extract CPU category from processor name

5. **GPU Features**
   - Extract GPU brand (Intel, NVIDIA, AMD)
   - Identify discrete vs integrated graphics

6. **Feature Selection**
   - Calculate correlation with Price
   - Remove features with correlation between -0.25 and +0.25
   - Drop redundant columns: Inches, ScreenResolution, Cpu, Gpu, Width, Height, GPU_Type, GPU_Number

7. **Final Features Used**
   - Company (Brand)
   - TypeName (Laptop Type)
   - Ram (Memory in GB)
   - OpSys (Operating System)
   - IPS_Panel (Display Type)
   - CPU_Category (Processor Type)
   - i3, i7 (CPU Flags)
   - PPI (Pixels Per Inch)
   - GPU_Brand (Graphics Card)
   - ssd (Storage in GB)

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

### **Training Pipeline**
- **Algorithm**: XGBoost Regressor (Best performing model)
- **Loss Function**: Squared Error (MSE)
- **Hyperparameter Tuning**: Optuna optimization framework
- **Cross-Validation**: 5-Fold CV
- **Target Transformation**: Log-transformed prices for better distribution

### **Final Performance**
```
Training R²:  0.9240
Testing R²:   0.8773 ✅ (Selected)
```

### **Model Selection Rationale**

| Model | Why Selected/Not Selected |
|-------|--------------------------|
| **XGBoost** ✅ | Best test R² (0.8773), excellent generalization, robust to outliers |
| Random Forest | Good performance (0.8685) but longer prediction time |
| Decision Tree | Overfitting (Train: 0.9733 vs Test: 0.8437), less stable |
| Linear Regression | Underfitting, poor baseline (0.8268) |

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

- **R² = 0.8773** means the model explains **87.73%** of price variance in test data
- **Better generalization** than other models due to XGBoost's boosting strategy
- **Robust predictions** even with unseen laptop configurations
- **Optimal complexity**: Avoids both underfitting and severe overfitting

## ⚠️ Model Limitations

- Model trained on ~1,300 laptop samples - new brands may not be recognized
- Predictions based on historical data; market trends may have changed
- R² of 0.8773 indicates ~12% of price variance is unexplained
- Works best for mid-range laptops with common configurations
- May not predict accurately for specialty/gaming laptops with extreme specs

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
