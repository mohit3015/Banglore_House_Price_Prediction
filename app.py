import json
import time
import warnings
import numpy as np
import pandas as pd
import pickle
import streamlit as st
try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE = True
except Exception:
    HAS_LOTTIE = False

warnings.filterwarnings("ignore")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="icons/house_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished cyber-clean look
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .price-badge {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
        margin-top: 4px;
        margin-bottom: 2px;
    }
    .emi-card {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 8px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATA & MODEL CACHING (0-LAG PERFORMANCE)
# ==========================================
@st.cache_data
def load_data():
    """Loads and caches the cleaned dataset and location mappings."""
    df = pd.read_csv("cleaned_df.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    # Location map dictionary for instant O(1) lookup
    loc_map = dict(zip(df['location'], df['encoded_loc']))
    return df, loc_map

@st.cache_resource
def load_model():
    """Loads and caches the trained Random Forest model."""
    with open("RFmodel.pkl", "rb") as file:
        model = pickle.load(file)
    return model

@st.cache_data
def load_lottie(filepath):
    """Loads Lottie animation JSON files."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

df, loc_map = load_data()
model = load_model()
home_anime = load_lottie("home_anime.json")


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("icons/house_logo.png", width=140)
    st.title("Bengaluru Housing AI")
    st.markdown("Estimate property prices across Bangalore using **Machine Learning** trained on verified real-estate data.")
    
    st.divider()
    st.markdown("### 📊 Model Telemetry")
    st.markdown("""
    - **Algorithm:** Random Forest Regressor
    - **Tuning:** GridSearchCV (5-Fold CV)
    - **Accuracy (R² Score):** `83.6%`
    - **Mean Absolute Error:** `~14.1 L`
    - **Dataset Size:** 10,230+ Properties
    """)
    
    st.divider()
    st.caption("Developed by **Mohit Kumar** • AI/ML Developer")


# ==========================================
# NAVIGATION TABS
# ==========================================
st.title("🏡 Bengaluru House Price Predictor")
st.caption("Accurate real estate valuation powered by Random Forest regression & historical sales trends.")

selection = st.segmented_control(
    "Navigation",
    options=["Predict Price", "Market Insights", "About Project"],
    default="Predict Price",
    label_visibility="collapsed"
)


# ==========================================
# MODAL DIALOG: PREDICTION RESULT
# ==========================================
@st.dialog("🏡 Property Valuation Summary")
def show_valuation_dialog(loc, sqft, bhk, bath, price_lakhs):
    total_price_inr = price_lakhs * 100000
    price_per_sqft = total_price_inr / sqft if sqft > 0 else 0
    
    # Currency formatting: Crores vs Lakhs
    if price_lakhs >= 100:
        display_price = f"₹ {price_lakhs / 100:.2f} Crore"
    else:
        display_price = f"₹ {price_lakhs:.2f} Lakhs"
    
    st.markdown(f"<div class='price-badge'>{display_price}</div>", unsafe_allow_html=True)
    st.caption(f"Full Amount: **₹ {total_price_inr:,.0f} INR** • **₹ {price_per_sqft:,.0f} / sq.ft**")
    
    st.divider()
    
    # Property Specs
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"📍 **Location:** {loc}")
        st.write(f"📐 **Total Area:** {sqft:,.0f} sq.ft")
    with col_b:
        st.write(f"🛏️ **Bedrooms:** {bhk} BHK")
        st.write(f"🚿 **Bathrooms:** {bath} Bath")
    
    # Home Loan EMI Estimator
    # Loan = 80% of price, Interest = 8.5% p.a., Tenure = 20 years (240 months)
    loan_amount = total_price_inr * 0.80
    annual_rate = 0.085
    monthly_rate = annual_rate / 12
    tenure_months = 20 * 12
    emi = loan_amount * monthly_rate * ((1 + monthly_rate) ** tenure_months) / (((1 + monthly_rate) ** tenure_months) - 1)
    
    st.markdown(f"""
    <div class='emi-card'>
        <strong>🏦 Estimated Home Loan EMI:</strong> <strong>₹ {emi:,.0f} / month</strong><br>
        <small style='color: #a1a1aa;'>Assuming 80% loan amount (₹ {loan_amount:,.0f}) @ 8.5% interest for 20 years tenure.</small>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# TAB 1: PREDICT PRICE
# ==========================================
if selection == "Predict Price":
    st.markdown("### 🎯 Enter Property Specifications")
    
    # Popular locations quick pills
    popular_hubs = [
        "Whitefield", "Electronic City", "Sarjapur Road",
        "HSR Layout", "Koramangala", "Indiranagar",
        "Thanisandra", "Hebbal", "Yelahanka", "Marathahalli"
    ]
    
    # Filter available popular hubs that exist in dataset
    available_popular = [h for h in popular_hubs if h in df['location'].values]
    
    st.markdown("**⚡ Quick Pick Popular Localities:**")
    selected_quick_loc = st.pills(
        "Popular Localities",
        options=available_popular,
        default=None,
        label_visibility="collapsed"
    )
    
    # Sorted location list
    all_locations = sorted(df['location'].unique().tolist())
    
    # Default index for selectbox
    default_idx = 0
    if selected_quick_loc and selected_quick_loc in all_locations:
        default_idx = all_locations.index(selected_quick_loc)
    elif "Whitefield" in all_locations:
        default_idx = all_locations.index("Whitefield")
    
    with st.container(border=True):
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            loc = st.selectbox("📍 Location / Locality:", options=all_locations, index=default_idx)
            
            # Easy number input with sensible defaults
            sqft = st.number_input(
                "📐 Built-up Area (Total Sq. Ft.):",
                min_value=300.0,
                max_value=35000.0,
                value=1200.0,
                step=50.0,
                help="Enter total square footage of the property (300 to 35,000 sq.ft)"
            )
        
        with col2:
            bhk = st.pills(
                "🛏️ Number of Bedrooms (BHK):",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: f"{x} BHK",
                default=2
            )
            if bhk is None:
                bhk = 2
            
            bath = st.pills(
                "🚿 Number of Bathrooms:",
                options=[1, 2, 3, 4, 5],
                format_func=lambda x: f"{x} Bath",
                default=2
            )
            if bath is None:
                bath = 2
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button("💰 Predict Property Valuation", use_container_width=True, type="primary")
    
    if predict_btn:
        encoded_location = loc_map.get(loc, 0)
        input_data = pd.DataFrame(
            [[encoded_location, sqft, bath, bhk]],
            columns=['location', 'total_sqft', 'bath', 'bhk']
        )
        
        with st.spinner("Analyzing market parameters and computing valuation..."):
            time.sleep(0.4)
            prediction_lakhs = float(model.predict(input_data)[0])
            show_valuation_dialog(loc, sqft, bhk, bath, prediction_lakhs)


# ==========================================
# TAB 2: MARKET INSIGHTS & ANALYTICS
# ==========================================
elif selection == "Market Insights":
    st.markdown("### 📈 Bangalore Real Estate Market Analytics")
    
    # Top Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Properties", f"{len(df):,}")
    with m2:
        st.metric("Average Price", f"₹ {df['price'].mean():.2f} Lakhs")
    with m3:
        st.metric("Median Area", f"{df['total_sqft'].median():,.0f} sq.ft")
    with m4:
        st.metric("Covered Localities", f"{df['location'].nunique()} Areas")
    
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.markdown("#### 🏆 Top 10 Premium Localities (Avg Price in Lakhs)")
        top_expensive = (
            df.groupby('location')['price']
            .mean()
            .reset_index()
            .sort_values(by='price', ascending=False)
            .head(10)
            .rename(columns={'location': 'Locality', 'price': 'Avg Price (Lakhs)'})
        )
        st.bar_chart(top_expensive.set_index('Locality'), color="#8b5cf6")
    
    with col_chart2:
        st.markdown("#### 🏷️ Average Price by Bedroom Count (BHK)")
        bhk_summary = (
            df.groupby('bhk')['price']
            .mean()
            .reset_index()
            .sort_values(by='bhk')
            .rename(columns={'bhk': 'Bedrooms (BHK)', 'price': 'Average Price (₹ Lakhs)'})
        )
        bhk_summary['Bedrooms (BHK)'] = bhk_summary['Bedrooms (BHK)'].apply(lambda x: f"{int(x)} BHK")
        st.bar_chart(bhk_summary.set_index('Bedrooms (BHK)'), color="#06b6d4")
    
    st.divider()
    st.markdown("#### 💡 Value Localities (High Availability & Budget Friendly)")
    top_volume = (
        df.groupby('location')
        .agg(Total_Units=('price', 'count'), Avg_Price_Lakhs=('price', 'mean'))
        .reset_index()
        .sort_values(by='Total_Units', ascending=False)
        .head(10)
    )
    st.dataframe(top_volume, use_container_width=True)


# ==========================================
# TAB 3: ABOUT PROJECT
# ==========================================
elif selection == "About Project":
    c_left, c_right = st.columns([1, 2], gap="large")
    
    with c_left:
        if home_anime and HAS_LOTTIE:
            st_lottie(home_anime, width=280)
        else:
            st.image("icons/house_logo.png", width=200)
            
    with c_right:
        st.markdown("""
        ### 🏡 About This Application
        
        The **Bengaluru House Price Predictor** is a production-grade machine learning application designed to estimate residential property prices across Bangalore with high statistical fidelity.
        
        #### 📌 Key Engineering Highlights:
        - **Random Forest Regressor:** Multi-tree ensemble tuned with `GridSearchCV` (n_estimators=300, max_depth=6) with 5-fold cross validation.
        - **Evaluation Metrics:** Achieved an **R² Score of 83.6%** with Mean Absolute Error of ~14.1 Lakhs.
        - **Data Pipeline:** Handled missing data, transformed total sqft strings, filtered multi-room outliers, and encoded categorical locations.
        - **Streamlit Caching:** Zero-lag query response using `@st.cache_data` and `@st.cache_resource`.
        - **Financial Telemetry:** Built-in 20-year Home Loan EMI estimation and price-per-sqft valuation metrics.
        """)
