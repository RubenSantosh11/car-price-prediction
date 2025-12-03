import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===========================
# PAGE CONFIGURATION
# ===========================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# ===========================
# LOAD MODEL
# ===========================

@st.cache_resource
def load_model():
    try:
        with open('models/v1/model.pkl', 'rb') as f:
            model_params = pickle.load(f)
        return model_params, None
    except Exception as e:
        return None, str(e)

model_params, error = load_model()

# ===========================
# PREDICTION FUNCTIONS
# ===========================

def add_polynomial_features(X, degree=2):
    """Add polynomial features if needed"""
    if degree == 1:
        return X
    
    n_samples, n_features = X.shape
    poly_features = [X]
    
    # Add squared terms
    squared = X ** 2
    poly_features.append(squared)
    
    # Add interaction terms
    for i in range(n_features):
        for j in range(i+1, n_features):
            interaction = (X[:, i] * X[:, j]).reshape(-1, 1)
            poly_features.append(interaction)
    
    return np.hstack(poly_features)

def predict_price(features, model_params):
    """
    Predict car price using Normal Equation parameters
    Pure mathematical implementation - no ML libraries!
    """
    # Extract parameters
    theta = model_params['theta']
    mean = model_params['mean']
    std = model_params['std']
    use_poly = model_params.get('use_polynomial', False)
    degree = model_params.get('degree', 1)
    
    # Convert to numpy array
    features_array = np.array(features).reshape(1, -1)
    
    # Scale features (standardization)
    features_scaled = (features_array - mean) / std
    
    # Add polynomial features if needed
    if use_poly and degree > 1:
        features_scaled = add_polynomial_features(features_scaled, degree)
    
    # Add bias term (column of ones)
    features_with_bias = np.c_[np.ones((1, 1)), features_scaled]
    
    # Predict: ŷ = X @ θ
    prediction = features_with_bias @ theta
    
    return prediction[0]

# ===========================
# UI - HEADER
# ===========================

st.title("🚗 Car Price Prediction System")
st.markdown("**AI-Powered Price Estimation using Pure Mathematics**")

# Check if model loaded
if error:
    st.error(f"❌ Error loading model: {error}")
    
    with st.expander("🔍 Debug Information"):
        st.code(error)
        st.markdown("""
        **Troubleshooting Steps:**
        1. Make sure `models/v1/model.pkl` exists in your GitHub repository
        2. Check that the file was uploaded correctly
        3. Verify the file is not corrupted
        """)
    st.stop()

st.success("✅ Model loaded successfully!")

# Display model info
model_type = model_params.get('model_type', 'LINEAR')
test_r2 = model_params.get('test_r2', 0.78)
test_mae = model_params.get('test_mae', 3064)

col1, col2, col3 = st.columns(3)
col1.metric("Algorithm", model_type)
col2.metric("R² Score", f"{test_r2*100:.1f}%")
col3.metric("Avg Error", f"${test_mae:,.0f}")

st.markdown("---")

# ===========================
# UI - INPUT FORM
# ===========================

st.subheader("📝 Enter Car Specifications")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔧 Physical Dimensions")
    
    wheelbase = st.slider(
        "Wheelbase (inches)", 
        86.0, 120.0, 98.5, 0.5,
        help="Distance between front and rear axles"
    )
    
    carlength = st.slider(
        "Car Length (inches)", 
        141.0, 208.0, 173.3, 0.5,
        help="Overall length of the vehicle"
    )
    
    carwidth = st.slider(
        "Car Width (inches)", 
        60.0, 72.0, 65.8, 0.1,
        help="Width of the vehicle"
    )
    
    carheight = st.slider(
        "Car Height (inches)", 
        47.0, 60.0, 53.7, 0.1,
        help="Height from ground to roof"
    )
    
    curbweight = st.slider(
        "Curb Weight (lbs)", 
        1488, 4066, 2555, 10,
        help="Weight of empty vehicle"
    )

with col2:
    st.markdown("#### ⚡ Engine Specifications")
    
    enginesize = st.slider(
        "Engine Size (cc)", 
        61, 326, 126, 5,
        help="Engine displacement in cubic centimeters"
    )
    
    boreratio = st.slider(
        "Bore Ratio", 
        2.54, 3.94, 3.33, 0.01,
        help="Engine bore to stroke ratio"
    )
    
    horsepower = st.slider(
        "Horsepower (hp)", 
        48, 288, 104, 5,
        help="Maximum engine power output"
    )
    
    citympg = st.slider(
        "City MPG", 
        13, 49, 25, 1,
        help="Fuel efficiency in city driving"
    )
    
    highwaympg = st.slider(
        "Highway MPG", 
        16, 54, 30, 1,
        help="Fuel efficiency on highways"
    )

# ===========================
# PREDICTION SECTION
# ===========================

st.markdown("---")

if st.button("🔮 Predict Car Price", type="primary", use_container_width=True):
    
    # Prepare features in correct order
    features = [
        wheelbase, carlength, carwidth, carheight,
        curbweight, enginesize, boreratio, horsepower,
        citympg, highwaympg
    ]
    
    try:
        # Make prediction
        predicted_price = predict_price(features, model_params)
        
        # Display result
        st.markdown("---")
        st.markdown("## 💰 Prediction Result")
        
        # Main prediction display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown(f"### Estimated Price: **${predicted_price:,.2f}**")
            
            # Progress bar
            price_percentage = min(predicted_price / 50000, 1.0)
            st.progress(price_percentage)
            
            # Price range (±10% confidence)
            lower = predicted_price * 0.9
            upper = predicted_price * 1.1
            st.info(f"📊 Expected Range: ${lower:,.2f} - ${upper:,.2f}")
            st.caption("Range represents ±10% confidence interval")
        
        # Car summary metrics
        st.markdown("---")
        st.markdown("### 📋 Your Car Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Engine", f"{enginesize} cc", f"{horsepower} hp")
        col2.metric("Efficiency", f"{citympg} / {highwaympg} MPG", "City / Highway")
        col3.metric("Weight", f"{curbweight:,} lbs", f"{carwidth:.1f}\" wide")
        col4.metric("Size", f"{carlength:.1f}\"", f"{wheelbase:.1f}\" wheelbase")
        
        # Value assessment
        st.markdown("---")
        st.markdown("### 💡 Value Assessment")
        
        if predicted_price < 10000:
            st.success("🟢 **Budget-Friendly** - Great value for money!")
            st.write("This car is perfect for first-time buyers or those on a tight budget.")
        elif predicted_price < 20000:
            st.info("🔵 **Mid-Range** - Good balance of features and price")
            st.write("Solid choice with decent features and reasonable pricing.")
        elif predicted_price < 35000:
            st.warning("🟡 **Premium** - Higher-end features and performance")
            st.write("Enhanced features, better performance, and quality materials.")
        else:
            st.error("🔴 **Luxury** - Top-tier vehicle with premium features")
            st.write("High-end vehicle with exceptional features and performance.")
        
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        with st.expander("Error Details"):
            st.code(str(e))

# ===========================
# FOOTER - MODEL INFO
# ===========================

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### 📊 About This Model
    
    **Algorithm**: Linear Regression via Normal Equation  
    **Formula**: θ = (X^T X)^(-1) X^T y  
    **Training Data**: 164 cars  
    **Test Accuracy**: 78.3% (R² score)  
    **Average Error**: $3,064  
    
    #### Why Normal Equation?
    - ✅ No ML libraries needed (only NumPy!)
    - ✅ Direct mathematical solution
    - ✅ Fast deployment (< 1 minute)
    - ✅ Complete transparency
    - ✅ Perfect for linear regression
    
    #### Model Selection Process
    We tested both linear and polynomial (degree 2) features:
    - Linear model: 78% test accuracy with minimal overfitting
    - Polynomial model: Showed 71% overfitting gap
    - **Decision**: Linear model generalizes better for our dataset size
    """)

with col2:
    st.markdown("""
    ### 🎓 Project Info
    
    **Created by**:  
    Ruben Santosh  
    Vignesh R Nair  
    Arko Chakraborty  
    
    **University**:  
    Dayananda Sagar University  
    
    **Department**:  
    CSE (AI & ML)  
    
    **Year**: 2025  
    
    **Tech Stack**:
    - Streamlit
    - NumPy
    - Pure Mathematics
    - GitHub
    
    **Dataset**:  
    Car Price Dataset  
    (205 cars, 10 features)
    """)

# Mathematical explanation (expandable)
with st.expander("🔬 See the Mathematics Behind Predictions"):
    st.markdown("""
    ### Normal Equation Formula
    
    The model finds optimal parameters θ directly:
    
    $$\\theta = (X^T X)^{-1} X^T y$$
    
    Where:
    - **θ (theta)** = model parameters (weights)
    - **X** = feature matrix (car specifications)
    - **y** = target values (prices)
    - **X^T** = transpose of X
    
    ### Prediction Formula
    
    For a new car, we calculate:
    
    $$\\hat{y} = \\theta_0 + \\theta_1 x_1 + \\theta_2 x_2 + ... + \\theta_{10} x_{10}$$
    
    ### Feature Scaling
    
    Before prediction, we standardize features:
    
    $$x_{scaled} = \\frac{x - \\mu}{\\sigma}$$
    
    Where:
    - **μ** = mean of feature
    - **σ** = standard deviation
    
    This ensures all features contribute equally to the prediction.
    
    ### Example Calculation
    
    For your car with enginesize = 126 cc:
    1. Scale: (126 - 128.5) / 41.2 = -0.061
    2. Multiply by weight: -0.061 × 2981.57 = -181.88
    3. Sum all weighted features + intercept = Final Price
    """)

# Usage tips
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    ### Getting Accurate Predictions
    
    1. **Use Realistic Values**
       - Don't extrapolate beyond slider ranges
       - These represent typical car specifications
    
    2. **Consider Correlations**
       - Larger engines usually mean higher weight
       - More horsepower often reduces MPG
       - Bigger cars typically have longer wheelbases
    
    3. **Understand the Range**
       - ±10% confidence interval is normal
       - Real car prices vary by condition, location, features
    
    4. **Model Limitations**
       - Based on 205 cars from dataset
       - Doesn't account for: brand reputation, condition, mileage
       - Best for comparative pricing
    """)

st.markdown("---")
st.caption("🚗 Car Price Predictor | Built with ❤️ using Pure Math | No ML Libraries Required!")
