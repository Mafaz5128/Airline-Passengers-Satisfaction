import streamlit as st
import pandas as pd
import joblib
import sklearn

# Load the model pipeline
pipeline_with_preprocessor = joblib.load('full_pipeline.joblib')
preprocessor = pipeline_with_preprocessor.named_steps['preprocessor']
model = pipeline_with_preprocessor.named_steps['classifier']

# Set page config
st.set_page_config(page_title="Airline Satisfaction Predictor", layout="wide")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predict", "About"])

if page == "Home":
    st.title("Airline Passenger Satisfaction Prediction")
    st.write("""
    Welcome to the Airline Passenger Satisfaction Prediction tool!
    
    Use this application to predict whether a passenger will be satisfied with their flight experience
    based on various factors like service ratings, flight details, and passenger information.
    """)
    
    st.image("static/airline_image.jpg", use_column_width=True)  # Replace with your image path

elif page == "Predict":
    st.title("Flight Satisfaction Prediction")
    st.write("Fill in the passenger details to predict satisfaction level")
    
    # Create form for user input
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            flight_distance = st.number_input("Flight Distance (miles)", min_value=0, value=500)
            
        with col2:
            departure_delay = st.number_input("Departure Delay (minutes)", min_value=0, value=0)
            arrival_delay = st.number_input("Arrival Delay (minutes)", min_value=0, value=0)
        
        st.subheader("Service Ratings (1-5)")
        service_cols = st.columns(4)
        
        with service_cols[0]:
            wifi = st.slider("Inflight WiFi", 1, 5, 3)
            departure_convenience = st.slider("Departure/Arrival Convenience", 1, 5, 3)
            online_booking = st.slider("Online Booking Ease", 1, 5, 3)
            
        with service_cols[1]:
            gate_location = st.slider("Gate Location", 1, 5, 3)
            food_drink = st.slider("Food & Drink", 1, 5, 3)
            online_boarding = st.slider("Online Boarding", 1, 5, 3)
            
        with service_cols[2]:
            seat_comfort = st.slider("Seat Comfort", 1, 5, 3)
            inflight_entertainment = st.slider("Inflight Entertainment", 1, 5, 3)
            onboard_service = st.slider("On-board Service", 1, 5, 3)
            
        with service_cols[3]:
            legroom = st.slider("Leg Room Service", 1, 5, 3)
            baggage = st.slider("Baggage Handling", 1, 5, 3)
            checkin = st.slider("Checkin Service", 1, 5, 3)
            
        inflight_service = st.slider("Inflight Service", 1, 5, 3)
        cleanliness = st.slider("Cleanliness", 1, 5, 3)
        
        submitted = st.form_submit_button("Predict Satisfaction")
        
        if submitted:
            # Prepare input data
            user_input = {
                'Gender': gender,
                'Customer Type': customer_type,
                'Age': age,
                'Type of Travel': travel_type,
                'Class': flight_class,
                'Flight Distance': flight_distance,
                'Inflight wifi service': wifi,
                'Departure/Arrival time convenient': departure_convenience,
                'Ease of Online booking': online_booking,
                'Gate location': gate_location,
                'Food and drink': food_drink,
                'Online boarding': online_boarding,
                'Seat comfort': seat_comfort,
                'Inflight entertainment': inflight_entertainment,
                'On-board service': onboard_service,
                'Leg room service': legroom,
                'Baggage handling': baggage,
                'Checkin service': checkin,
                'Inflight service': inflight_service,
                'Cleanliness': cleanliness,
                'Departure Delay in Minutes': departure_delay,
                'Arrival Delay in Minutes': arrival_delay
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([user_input])
            
            # Make prediction
            preprocessed_input = preprocessor.transform(input_df)
            prediction = model.predict(preprocessed_input)[0]
            
            # Display result
            st.subheader("Prediction Result")
            if prediction == 0:
                st.error("Predicted: Neutral or dissatisfied")
            else:
                st.success("Predicted: Satisfied")

elif page == "About":
    st.title("About This Project")
    st.write("""
    ## Airline Passenger Satisfaction Prediction
    
    This application predicts whether an airline passenger will be satisfied with their flight experience
    based on various factors including service ratings, flight details, and passenger information.
    
    ### How it works
    - The model uses a machine learning pipeline with preprocessing and classification
    - It was trained on historical airline passenger data
    - Predictions are made based on the input parameters you provide
    
    ### Technologies Used
    - Python
    - Scikit-learn for machine learning
    - Streamlit for the web interface
    """)

# To run this app, save it as app.py and run: streamlit run app.py
