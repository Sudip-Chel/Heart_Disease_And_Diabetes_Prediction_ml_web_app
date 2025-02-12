import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np

# Custom CSS for better UI
st.set_page_config(page_title="Disease Prediction System", layout="wide")

# Add custom CSS
 #Updated Custom CSS with better button styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #ff4b4b;
        color: white;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        color : black;
    }
    .stButton>button:active {
        transform: translateY(0);
    }
    .stButton>button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            120deg,
            transparent,
            rgba(200,219,255,0.3),
            transparent
        );
        transition: 0.5s;
    }
    .stButton>button:hover::before {
        left: 100%;
    }
    .error-msg {
        color: red;
        font-size: 14px;
    }
    .risk-factor-normal {
        color: green;
        font-weight: bold;
    }
    .risk-factor-warning {
        color: orange;
        font-weight: bold;
    }
    .risk-factor-danger {
        color: red;
        font-weight: bold;
    }
    /* Custom loading animation */
    .calculating {
        display: inline-block;
        position: relative;
        width: 80px;
        height: 80px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    try:
        diabetes_model = pickle.load(open("saved_models\diabetes_model.sav", 'rb'))
        heart_model = pickle.load(open('saved_models\heart_model.sav', 'rb'))
        return diabetes_model, heart_model
    except FileNotFoundError:
        st.error("Model files not found. Please check the file paths.")
        return None, None

diabetes_model, heart_disease_model = load_models()

# New Feature 1: BMI Calculator Helper
def bmi_calculator():
    st.markdown("### 📏 BMI Calculator Helper")
    col1, col2 = st.columns(2)
    
    with col1:
        height_unit = st.selectbox("Height Unit", ["cm", "feet"])
        if height_unit == "cm":
            height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0)
            height_m = height / 100
        else:
            feet = st.number_input("Feet", min_value=0, max_value=8)
            inches = st.number_input("Inches", min_value=0, max_value=11)
            height_m = (feet * 30.48 + inches * 2.54) / 100

    with col2:
        weight_unit = st.selectbox("Weight Unit", ["kg", "lbs"])
        if weight_unit == "kg":
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=300.0)
            weight_kg = weight
        else:
            weight = st.number_input("Weight (lbs)", min_value=0.0, max_value=660.0)
            weight_kg = weight * 0.45359237

    if height_m > 0 and weight_kg > 0:
        bmi = weight_kg / (height_m * height_m)
        st.write(f"Calculated BMI: **{bmi:.1f}**")
        
        # BMI Category
        if bmi < 18.5:
            st.warning("Category: Underweight")
        elif 18.5 <= bmi < 25:
            st.success("Category: Normal weight")
        elif 25 <= bmi < 30:
            st.warning("Category: Overweight")
        else:
            st.error("Category: Obese")
            
        return bmi
    return None

# New Feature 2: Risk Factor Summary
def show_risk_factor_summary(values_dict):
    st.markdown("### 🎯 Risk Factor Summary")
    
    def get_status_color(value, ranges):
        if value < ranges[0]:
            return "risk-factor-normal"
        elif value < ranges[1]:
            return "risk-factor-warning"
        else:
            return "risk-factor-danger"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Key Metrics:")
        for factor, value in values_dict.items():
            if factor == "Blood Pressure":
                color = get_status_color(value, [120, 140])
                st.markdown(f"- {factor}: <span class='{color}'>{value} mm Hg</span>", unsafe_allow_html=True)
            elif factor == "Glucose":
                color = get_status_color(value, [100, 126])
                st.markdown(f"- {factor}: <span class='{color}'>{value} mg/dL</span>", unsafe_allow_html=True)
            elif factor == "BMI":
                color = get_status_color(value, [25, 30])
                st.markdown(f"- {factor}: <span class='{color}'>{value:.1f}</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### Reference Ranges:")
        st.markdown("""
        - Blood Pressure: <120 Normal, 120-139 Elevated, ≥140 High
        - Glucose: <100 Normal, 100-125 Prediabetes, ≥126 Diabetes
        - BMI: <25 Normal, 25-29.9 Overweight, ≥30 Obese
        """)

# [Previous sidebar code remains the same...]
with st.sidebar:
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction'],
        icons=['activity', 'heart'],
        menu_icon='hospital-fill',
        default_index=0,
        styles={
            "container": {"padding": "5!important"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
        }
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    
    # Add BMI Calculator Tab
    tab1, tab2 = st.tabs(["📊 Prediction", "🔧 BMI Calculator"])
    
    with tab1:
        # [Previous diabetes prediction code remains the same until BMI input...]
        col1, col2 = st.columns(2)
        
        with col1:
            pregnancies = st.number_input('Number of Pregnancies', 
                                        min_value=0, max_value=20,
                                        help="Enter number of pregnancies (0-20)")
            
            glucose = st.number_input('Glucose Level (mg/dL)', 
                                    min_value=0, max_value=300,
                                    help="Normal fasting glucose range is 70-200 mg/dL")
            
            blood_pressure = st.number_input('Blood Pressure (mm Hg)',
                                           min_value=0, max_value=150,
                                           help="Normal range is 60-130 mm Hg")
            
            skin_thickness = st.number_input('Skin Thickness (mm)',
                                           min_value=0, max_value=100,
                                           help="Typical range is 0-100 mm")

        with col2:
            insulin = st.number_input('Insulin Level (mu U/ml)',
                                    min_value=0, max_value=900,
                                    help="Normal fasting insulin range is 0-850 mu U/ml")
            
            bmi = st.number_input('BMI',
                                min_value=0.0, max_value=70.0,
                                help="Normal BMI range is 18.5-40")
            
            dpf = st.number_input('Diabetes Pedigree Function',
                                min_value=0.0, max_value=3.0,
                                help="Genetic influence score (0.0-3.0)")
            
            age = st.number_input('Age',
                                min_value=0, max_value=120,
                                help="Enter age in years")

        if st.button('Check Diabetes Risk'):
            try:
                input_data = [pregnancies, glucose, blood_pressure, skin_thickness, 
                             insulin, bmi, dpf, age]
                
                # Show risk factor summary before prediction
                show_risk_factor_summary({
                    "Blood Pressure": blood_pressure,
                    "Glucose": glucose,
                    "BMI": bmi
                })
                
                st.markdown("---")
                
                if any(x is None for x in input_data):
                    st.error("Please fill in all fields with valid values")
                else:
                    prediction = diabetes_model.predict([input_data])
                    
                    if prediction[0] == 1:
                        st.warning("⚠️ Based on the provided information, you may be at risk for diabetes.")
                        st.markdown("""
                            ### Recommended Next Steps:
                            1. Consult with a healthcare provider for proper medical evaluation
                            2. Get a comprehensive blood sugar test
                            3. Review your diet and exercise habits
                            4. Monitor your blood sugar regularly
                        """)
                    else:
                        st.success("✅ Based on the provided information, you appear to have a lower risk for diabetes.")
                        st.markdown("""
                            ### Maintaining Good Health:
                            - Continue regular exercise
                            - Maintain a balanced diet
                            - Have regular health check-ups
                            - Monitor your blood sugar periodically
                        """)
            except Exception as e:
                st.error(f"An error occurred during prediction. Please check your inputs and try again.")

    with tab2:
        calculated_bmi = bmi_calculator()
        if calculated_bmi is not None:
            st.button("Use this BMI", on_click=lambda: st.session_state.update({"bmi": calculated_bmi}))

# [Rest of the Heart Disease Prediction code remains the same...]

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    st.write("Enter your cardiac health information for heart disease risk assessment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age', 
                            min_value=0, max_value=120,
                            help="Enter age in years")
        
        sex = st.selectbox('Sex', 
                          options=['Male', 'Female'],
                          help="Select biological sex")
        
        cp = st.selectbox('Chest Pain Type',
                         options=['Typical Angina', 'Atypical Angina', 
                                'Non-anginal Pain', 'Asymptomatic'],
                         help="Select the type of chest pain experienced")
        
        trestbps = st.number_input('Resting Blood Pressure (mm Hg)',
                                  min_value=0, max_value=300,
                                  help="Normal range is 90-200 mm Hg")

    with col2:
        chol = st.number_input('Serum Cholesterol (mg/dl)',
                             min_value=0, max_value=600,
                             help="Normal range is 120-570 mg/dl")
        
        fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl',
                          options=['Yes', 'No'],
                          help="Select if fasting blood sugar is above 120 mg/dl")
        
        restecg = st.selectbox('Resting ECG Results',
                             options=['Normal', 'ST-T Wave Abnormality', 
                                    'Left Ventricular Hypertrophy'],
                             help="Select the resting electrocardiographic results")
        
        thalach = st.number_input('Maximum Heart Rate',
                                min_value=0, max_value=250,
                                help="Normal range is 60-220")

    with col3:
        exang = st.selectbox('Exercise Induced Angina',
                           options=['Yes', 'No'],
                           help="Select if you experience chest pain during exercise")
        
        oldpeak = st.number_input('ST Depression',
                                min_value=0.0, max_value=10.0,
                                help="ST depression induced by exercise relative to rest")
        
        slope = st.selectbox('ST Segment Slope',
                           options=['Upsloping', 'Flat', 'Downsloping'],
                           help="Select the slope of peak exercise ST segment")
        
        ca = st.number_input('Number of Major Vessels',
                           min_value=0, max_value=4,
                           help="Number of major vessels colored by fluoroscopy (0-4)")
        
        thal = st.selectbox('Thalassemia',
                          options=['Normal', 'Fixed Defect', 'Reversible Defect'],
                          help="Select the thalassemia type")

    # Convert categorical inputs to numeric
    sex_dict = {'Male': 1, 'Female': 0}
    cp_dict = {'Typical Angina': 0, 'Atypical Angina': 1, 
               'Non-anginal Pain': 2, 'Asymptomatic': 3}
    fbs_dict = {'Yes': 1, 'No': 0}
    restecg_dict = {'Normal': 0, 'ST-T Wave Abnormality': 1, 
                    'Left Ventricular Hypertrophy': 2}
    exang_dict = {'Yes': 1, 'No': 0}
    slope_dict = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_dict = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

    if st.button('Check Heart Disease Risk'):
        try:
            # Convert inputs to numeric values
            input_data = [
                age, sex_dict[sex], cp_dict[cp], trestbps, chol,
                fbs_dict[fbs], restecg_dict[restecg], thalach,
                exang_dict[exang], oldpeak, slope_dict[slope], ca,
                thal_dict[thal]
            ]
            
            prediction = heart_disease_model.predict([input_data])
            
            st.markdown("---")
            if prediction[0] == 1:
                st.warning("⚠️ Based on the provided information, you may be at risk for heart disease.")
                st.markdown("""
                    ### Recommended Next Steps:
                    1. Schedule an appointment with a cardiologist
                    2. Get a comprehensive cardiac evaluation
                    3. Review your lifestyle habits
                    4. Monitor your blood pressure regularly
                    5. Consider stress management techniques
                """)
            else:
                st.success("✅ Based on the provided information, you appear to have a lower risk for heart disease.")
                st.markdown("""
                    ### Maintaining Heart Health:
                    - Regular cardiovascular exercise
                    - Heart-healthy diet
                    - Regular blood pressure monitoring
                    - Stress management
                    - Regular check-ups
                """)
        except Exception as e:
            st.error("An error occurred during prediction. Please check your inputs and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><small>Disclaimer: This is a preliminary screening tool and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</small></p>
    </div>
""", unsafe_allow_html=True)

