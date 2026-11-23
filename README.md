# 🏡 Bengaluru House Price Predictor | Machine Learning Web Application

[![Live Demo](https://img.shields.io/badge/Streamlit_Cloud-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://banglorehousepriceprediction-cxd7x5fsvydqdjmp8ubng8.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Portfolio](https://img.shields.io/badge/Portfolio-mohit3015.github.io-8b5cf6?style=for-the-badge&logo=googlechrome&logoColor=white)](https://mohit3015.github.io/Mohit-Portfolio/)

An end-to-end Machine Learning web application designed to forecast residential real estate prices across Bangalore, India. Powered by a hyperparameter-tuned **Random Forest Regressor** and an interactive **Streamlit** user interface complete with financial telemetry, location intelligence, and market trend visual analytics.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Machine Learning Architecture](#-machine-learning-architecture)
- [Model Evaluation & Benchmarks](#-model-evaluation--benchmarks)
- [Web Application Walkthrough](#-web-application-walkthrough)
- [Project Directory Structure](#-project-directory-structure)
- [Local Installation & Setup](#-local-installation--setup)
- [Author & Links](#-author--links)

---

## 🎯 Project Overview

Real estate prices in metropolitan cities like Bangalore are heavily influenced by geographic location, square footage, bedroom count, and bathroom availability. Traditional manual appraisals can be inconsistent and slow.

This project delivers:
1. **Accurate Price Valuation:** Predicts house prices in **₹ Lakhs and ₹ Crores** with ~83.6% statistical precision ($R^2$).
2. **Instant EMI Estimation:** Computes estimated monthly home loan installments (assuming 80% loan at standard 8.5% p.a. interest over 20 years).
3. **Market Analytics:** Visualizes premium vs budget-friendly localities and BHK pricing distributions using interactive native charts.

---

## ✨ Key Features

* **⚡ 0-Lag Streamlit Caching:** Uses `@st.cache_data` and `@st.cache_resource` to load data and serialized models in milliseconds.
* **📍 Quick-Pick Popular Localities:** 1-click pills for Bangalore's major tech corridors (Whitefield, Electronic City, HSR Layout, Koramangala, Indiranagar, etc.).
* **📐 Modern Numeric Controls:** Clean `st.number_input` for square footage (300 to 35,000 sq.ft) with 50 sq.ft stepping, eliminating tedious dropdown scrolling.
* **🛏️ Modern Selection Pills:** Intuitive 1-to-5 BHK and Bathrooms selection with pill chips.
* **🏦 Financial Telemetry:**
  - Automated Lakhs vs Crores currency formatting.
  - Valuation per square foot (`₹ / sq.ft`).
  - Estimated 20-year monthly home loan EMI breakdown.
* **📈 Market Analytics Tab:**
  - Top 10 most premium Bangalore localities.
  - Average price distribution grouped by BHK.
  - High-volume value localities table.

---

## 🧠 Machine Learning Architecture

```
[Raw Real Estate CSV] 
         │
         ▼
[Data Cleaning & Preprocessing]
   ├── Missing Value Imputation
   ├── Total SqFt Range Parsing
   └── Location Categorical Encoding (LabelEncoder)
         │
         ▼
[Feature Selection: location, total_sqft, bath, bhk]
         │
         ▼
[Model Training: Random Forest Regressor]
   ├── GridSearchCV Tuning (n_estimators: 100-300, max_depth: 3-6)
   └── 5-Fold Stratified Cross-Validation
         │
         ▼
[Export: RFmodel.pkl] ──▶ [Streamlit Web App: app.py]
```

---

## 📊 Model Evaluation & Benchmarks

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Algorithm** | `RandomForestRegressor` | Multi-decision-tree ensemble modeling complex non-linear relationships |
| **Best Hyperparameters** | `n_estimators=300, max_depth=6` | Selected via 5-Fold Grid Search |
| **$R^2$ Score (Test Set)** | **`0.836` (83.6%)** | Proportion of variance explained by model features |
| **Mean Absolute Error (MAE)** | **`14.18 Lakhs`** | Average deviation from ground-truth listing prices |
| **Dataset Volume** | **`10,230+ Properties`** | Filtered and normalized residential listings across 240+ localities |

---

## 🖥️ Web Application Walkthrough

The application is structured into 3 intuitive views:

1. **Predict Price:** Enter locality, total area, BHK, and bathrooms $\rightarrow$ click **Predict Property Valuation** $\rightarrow$ opens a modal dialog displaying price, price/sqft, and estimated EMI.
2. **Market Insights:** Interactive analytics showing premium vs budget corridors, BHK pricing comparisons, and high-availability neighborhoods.
3. **About Project:** Architecture highlights, methodology, and engineering specifications.

---

## 📁 Project Directory Structure

```
Banglore_House_Price_Prediction/
│
├── app.py                             # Optimized Streamlit frontend with caching & EMI math
├── app (1).py                         # Original legacy frontend script (retained for reference)
├── RFmodel.pkl                        # Serialized Random Forest Regressor (GridSearchCV tuned)
├── cleaned_df.csv                     # Cleaned & encoded dataset used by inference engine
├── Bengaluru_House_Data.csv           # Original raw housing dataset
├── house_price_prediction.ipynb       # Jupyter notebook containing EDA, cleaning, and model training
│
├── home_anime.json                    # Lottie animation for homepage banner
├── loading_anime.json                 # Lottie animation for inference loading
│
├── icons/
│   ├── house_logo.png                 # Main project logo & browser favicon
│   └── placeholder.png                # Fallback property thumbnail
│
├── .streamlit/
│   └── config.toml                    # Streamlit layout and theme settings
│
├── requirements.txt                   # Production package dependencies
├── .gitignore                         # Git exclusion rules
└── README.md                          # Comprehensive project documentation
```

---

## 💻 Local Installation & Setup

To clone and run this application on your local machine:

```bash
# 1. Clone the repository
git clone https://github.com/mohit3015/Banglore_House_Price_Prediction.git

# 2. Navigate into project directory
cd Banglore_House_Price_Prediction

# 3. Create a virtual environment (optional but recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch the Streamlit web application
streamlit run app.py
```

The application will automatically start in your default web browser at `http://localhost:8501`.

---

## 👨‍💻 Author & Contact

* **Developer:** **Mohit Kumar** (B.Tech CSE - AI/ML)
* 🚀 **Live Streamlit App:** [banglorehousepriceprediction-cxd7x5fsvydqdjmp8ubng8.streamlit.app](https://banglorehousepriceprediction-cxd7x5fsvydqdjmp8ubng8.streamlit.app/)
* 🌐 **Live Portfolio:** [https://mohit3015.github.io/Mohit-Portfolio/](https://mohit3015.github.io/Mohit-Portfolio/)
* 🐙 **GitHub:** [@mohit3015](https://github.com/mohit3015)
* 💼 **LinkedIn:** [Mohit Kumar](https://www.linkedin.com/in/mohit-kumar-9a33412b3/)
* 📧 **Email:** [mk5972345@gmail.com](mailto:mk5972345@gmail.com)

---
⭐ *If you find this project helpful, please consider starring the repository!*
