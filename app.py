# import streamlit as st
# import plotly.graph_objects as go
# import random
# import pandas as pd

# # --- APP CONFIGURATION ---
# st.set_page_config( 
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide"
# )

# # --- DATA ---
# # This is the same car data from your original application
# CAR_DATA = {
#   "Maruti": ["Swift", "Swift Dzire", "Alto 800", "Wagon R 1.0", "Ciaz", "Ertiga", "Vitara Brezza", "Baleno", "S Cross", "Celerio", "IGNIS"],
#   "Mahindra": ["XUV500", "Scorpio", "Thar", "XUV300", "Bolero", "Marazzo", "TUV300"],
#   "Volkswagen": ["Polo", "Vento", "Ameo", "Jetta", "Passat", "Tiguan"],
#   "Tata": ["Nexon", "Harrier", "Tiago", "Tigor", "Safari", "Hexa", "PUNCH"],
#   "Hyundai": ["i20", "Creta", "Verna", "VENUE", "Grand i10", "Santro", "Xcent", "Aura"],
#   "Honda": ["City", "Amaze", "Jazz", "WR-V", "BR-V", "Civic"],
#   "Ford": ["EcoSport", "Endeavour", "Figo", "Aspire", "Freestyle"],
#   "BMW": ["3 Series", "5 Series", "X1", "X3", "X5", "7 Series"],
#   "Renault": ["Kwid", "Duster", "Triber", "Kiger", "Captur"],
#   "MG": ["Hector", "Hector Plus", "Gloster", "ZS EV"],
#   "Datsun": ["redi-GO", "GO", "GO+"],
#   "Nissan": ["Magnite", "Kicks", "Terrano", "Sunny", "Micra"],
#   "Toyota": ["Innova Crysta", "Fortuner", "Yaris", "Glanza", "Urban Cruiser", "Corolla Altis"],
#   "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
#   "Jeep": ["Compass", "Wrangler", "Meridian"],
#   "KIA": ["Seltos", "Sonet", "Carnival", "Carens"],
#   "Audi": ["A4", "A6", "Q3", "Q5", "Q7"],
#   "Landrover": ["Range Rover Evoque", "Discovery Sport", "Range Rover Velar"],
#   "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
#   "Chevrolet": ["Beat", "Cruze", "Spark", "Sail", "Enjoy"],
#   "Fiat": ["Punto", "Linea"],
#   "Ssangyong": ["Rexton"],
#   "Jaguar": ["XF", "XE", "F-PACE"],
#   "Mitsubishi": ["Pajero Sport"],
#   "CITROEN": ["C5 Aircross", "C3"],
#   "Mini": ["Cooper"],
#   "ISUZU": ["D-MAX V-Cross"],
#   "Volvo": ["XC60", "XC90", "S90"],
#   "Porsche": ["Cayenne", "Macan"],
#   "Force": ["Gurkha"]
# }

# # --- MOCK PREDICTION LOGIC ---
# # This is the same logic from your original app, now in Python
# def predict_car_price(age, km_driven, fuel, transmission, ownership):
#     base_price = 8.0
#     base_price -= age * 0.5
#     base_price -= km_driven / 50000
#     if fuel == 'Diesel': base_price += 1
#     if transmission == 'Automatic': base_price += 1.5
#     if ownership == 'Second Owner': base_price -= 1
#     if ownership == 'Third Owner': base_price -= 2
#     final_price = max(1.5, base_price + (random.random() - 0.5))
#     return final_price

# # --- PLOTTING FUNCTIONS ---
# def create_shap_plot(inputs, final_price):
#     base_value = 8.0
#     age_contribution = -(inputs['age'] * 0.5)
#     km_contribution = -(inputs['km'] / 50000)
#     fuel_contribution = 1.0 if inputs['fuel'] == 'Diesel' else -0.2
#     transmission_contribution = 1.5 if inputs['transmission'] == 'Automatic' else -0.5
    
#     contributions = [age_contribution, km_contribution, fuel_contribution, transmission_contribution]
#     features = [f"Age = {inputs['age']} yrs", f"KM Driven = {(inputs['km']/1000):.1f}k km", f"Fuel = {inputs['fuel']}", f"Transmission = {inputs['transmission']}"]
    
#     colors = ['#E74C3C' if c < 0 else '#2ECC71' for c in contributions]

#     fig = go.Figure(go.Bar(
#         x=contributions,
#         y=features,
#         orientation='h',
#         marker_color=colors
#     ))
#     fig.update_layout(
#         title=f"<b>How Features Impact the Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#         xaxis_title="Contribution to Price (in Lakhs)",
#         yaxis=dict(autorange="reversed"),
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         font_color="white",
#         margin=dict(l=120, r=20, t=60, b=50)
#     )
#     return fig

# # --- STREAMLIT UI ---
# st.title("🏎️ Car Price Prediction & Analysis")

# # --- Sidebar Navigation ---
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])

# # --- PAGE 1: About Me ---
# if page == "About Me":
#     st.header("About Me")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.markdown(
#             """
#             Hi, I’m **Alok Mahadev Tungal** — a passionate **Data Scientist** and **Machine Learning Enthusiast**.

#             This app demonstrates my journey of building a complete project:
#             from *data analysis* ➝ *exploratory visualization* ➝ *model building* ➝ *deployment*.

#             My goal is to solve real-world problems with **AI & ML** while creating impactful and interactive applications.
#             """
#         )
#         st.markdown(
#             """
#             - **LinkedIn:** [https://www.linkedin.com/](https://www.linkedin.com/)
#             - **GitHub:** [https://github.com/](https://github.com/)
#             - **Hugging Face:** [https://huggingface.co/](https://huggingface.co/)
#             """
#         )
#     with col2:
#         st.image("https://placehold.co/200x200/ff3366/ffffff?text=AMT", caption="Alok Mahadev Tungal")

# # --- PAGE 2: The Project ---
# elif page == "The Project":
#     st.header("The Project")
#     st.markdown("Here are the key metrics and technologies used in this project.")
    
#     kpi_cols = st.columns(3)
#     kpi_cols[0].metric("Dataset Rows", "9,176")
#     kpi_cols[1].metric("Dataset Columns", "9")
#     kpi_cols[2].metric("Brands Covered", "30")

#     kpi_cols = st.columns(3)
#     kpi_cols[0].metric("Regression Model", "XGBoost")
#     kpi_cols[1].metric("Prediction Accuracy (R²)", "~96%")
#     kpi_cols[2].metric("Visualization Library", "Plotly")

# # --- PAGE 3: Data Insights ---
# elif page == "Data Insights":
#     st.header("Data Insights & Visualizations")
#     st.info("Here are some interactive charts exploring the car dataset. These are mock visualizations based on the data structure.")
    
#     # Mock data generation for plots
#     num_points = 500
#     prices = [random.uniform(2.0, 48.0) for _ in range(num_points)]
#     ages = [random.randint(1, 12) for _ in range(num_points)]
#     brands = [random.choice(list(CAR_DATA.keys())) for _ in range(num_points)]
    
#     plot_choice = st.selectbox(
#         "Choose a visualization:",
#         (
#             'Distribution of Car Prices',
#             'Car Listings by Brand',
#             'Price vs. Car Age',
#             'Price Distribution by Brand',
#             'Feature Correlation Heatmap'
#         )
#     )

#     fig = go.Figure()

#     if plot_choice == 'Distribution of Car Prices':
#         fig.add_trace(go.Histogram(x=prices, marker_color='#f97316'))
#         fig.update_layout(title='<b>Distribution of Car Prices</b>', xaxis_title='Price (in Lakhs)', yaxis_title='Count')
    
#     elif plot_choice == 'Car Listings by Brand':
#         brand_counts = pd.Series(brands).value_counts().nlargest(15)
#         fig.add_trace(go.Bar(y=brand_counts.index, x=brand_counts.values, orientation='h', marker_color='#8b5cf6'))
#         fig.update_layout(title='<b>Car Listings by Brand (Top 15)</b>', yaxis=dict(autorange="reversed"))

#     elif plot_choice == 'Price vs. Car Age':
#         fig.add_trace(go.Scatter(x=ages, y=prices, mode='markers', marker=dict(color='#14b8a6', opacity=0.6)))
#         fig.update_layout(title='<b>Price vs. Car Age</b>', xaxis_title='Age (Years)', yaxis_title='Price (Lakhs)')
    
#     elif plot_choice == 'Price Distribution by Brand':
#         df = pd.DataFrame({'brand': brands, 'price': prices})
#         top_brands = df['brand'].value_counts().nlargest(8).index
#         for brand in top_brands:
#             fig.add_trace(go.Box(y=df[df['brand'] == brand]['price'], name=brand, marker_color='#facc15'))
#         fig.update_layout(title='<b>Price Distribution by Brand</b>', showlegend=False)

#     elif plot_choice == 'Feature Correlation Heatmap':
#         corr_matrix = [[1.00, -0.65, -0.45], [-0.65, 1.00, 0.30], [-0.45, 0.30, 1.00]]
#         labels = ['Price', 'Age', 'KM']
#         fig.add_trace(go.Heatmap(z=corr_matrix, x=labels, y=labels, colorscale='Viridis', zmin=-1, zmax=1))
#         fig.update_layout(title='<b>Feature Correlation Heatmap</b>')
    
#     fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
#     st.plotly_chart(fig, use_container_width=True)


# # --- PAGE 4: Price Predictor ---
# elif page == "Price Predictor":
#     st.header("Price Predictor")

#     col1, col2 = st.columns([1, 1])

#     with col1:
#         st.subheader("Enter Car Details")
        
#         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
        
#         # Dependent dropdown for model
#         models = CAR_DATA[brand]
#         model = st.selectbox("Car Model", options=sorted(models))
        
#         age = st.number_input("Car Age (in years)", min_value=0, max_value=25, value=5)
#         km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
        
#         fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
#         transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#         ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
        
#         predict_button = st.button("Predict Price", use_container_width=True)

#     with col2:
#         st.subheader("Prediction Result")
        
#         if predict_button:
#             with st.spinner('Calculating...'):
#                 predicted_price = predict_car_price(age, km_driven, fuel_type, transmission, ownership)
                
#                 st.success(f"## Predicted Price: ₹ {predicted_price:.2f} Lakhs")
                
#                 # SHAP-like plot
#                 inputs = {'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}
#                 shap_fig = create_shap_plot(inputs, predicted_price)

#                 st.plotly_chart(shap_fig, use_container_width=True)
#         else:
#             st.info("Click 'Predict Price' after entering the details to see the result.")




# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import plotly.graph_objects as go

# # --- APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide"
# )

# # --- MODEL AND DATA ---
# MODEL_FILE = "car_price_predictoR.joblib"
# CAR_DATA = {
#     "Maruti": ["Swift", "Swift Dzire", "Alto 800", "Wagon R 1.0", "Ciaz", "Ertiga", "Vitara Brezza", "Baleno", "S Cross", "Celerio", "IGNIS"],
#     "Mahindra": ["XUV500", "Scorpio", "Thar", "XUV300", "Bolero", "Marazzo", "TUV300"],
#     "Volkswagen": ["Polo", "Vento", "Ameo", "Jetta", "Passat", "Tiguan"],
#     "Tata": ["Nexon", "Harrier", "Tiago", "Tigor", "Safari", "Hexa", "PUNCH"],
#     "Hyundai": ["i20", "Creta", "Verna", "VENUE", "Grand i10", "Santro", "Xcent", "Aura"],
#     "Honda": ["City", "Amaze", "Jazz", "WR-V", "BR-V", "Civic"],
#     "Ford": ["EcoSport", "Endeavour", "Figo", "Aspire", "Freestyle"],
#     "BMW": ["3 Series", "5 Series", "X1", "X3", "X5", "7 Series"],
#     "Renault": ["Kwid", "Duster", "Triber", "Kiger", "Captur"],
#     "MG": ["Hector", "Hector Plus", "Gloster", "ZS EV"],
#     "Datsun": ["redi-GO", "GO", "GO+"],
#     "Nissan": ["Magnite", "Kicks", "Terrano", "Sunny", "Micra"],
#     "Toyota": ["Innova Crysta", "Fortuner", "Yaris", "Glanza", "Urban Cruiser", "Corolla Altis"],
#     "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
#     "Jeep": ["Compass", "Wrangler", "Meridian"],
#     "KIA": ["Seltos", "Sonet", "Carnival", "Carens"],
#     "Audi": ["A4", "A6", "Q3", "Q5", "Q7"],
#     "Landrover": ["Range Rover Evoque", "Discovery Sport", "Range Rover Velar"],
#     "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
#     "Chevrolet": ["Beat", "Cruze", "Spark", "Sail", "Enjoy"],
#     "Fiat": ["Punto", "Linea"],
#     "Ssangyong": ["Rexton"],
#     "Jaguar": ["XF", "XE", "F-PACE"],
#     "Mitsubishi": ["Pajero Sport"],
#     "CITROEN": ["C5 Aircross", "C3"],
#     "Mini": ["Cooper"],
#     "ISUZU": ["D-MAX V-Cross"],
#     "Volvo": ["XC60", "XC90", "S90"],
#     "Porsche": ["Cayenne", "Macan"],
#     "Force": ["Gurkha"]
# }

# # --- LOAD MODEL ---
# @st.cache_resource
# def load_model(model_path):
#     """Loads the pre-trained model from a joblib file."""
#     if not os.path.exists(model_path):
#         st.error(f"Model file not found at `{model_path}`!")
#         st.stop()
#     try:
#         model = joblib.load(model_path)
#         return model
#     except Exception as e:
#         st.error(f"❌ Error loading model: {e}")
#         st.stop()

# model_pipeline = load_model(MODEL_FILE)

# # --- HELPER FUNCTION FOR PLOTTING ---
# def create_shap_plot(inputs, final_price):
#     """Creates a mock feature impact plot."""
#     base_value = 8.0  # Assuming a base price for visualization
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [
#         f"Age = {inputs['age']} yrs",
#         f"KM Driven = {(inputs['km']/1000):.1f}k km",
#         f"Fuel = {inputs['fuel']}",
#         f"Transmission = {inputs['transmission']}"
#     ]
#     colors = ['#E74C3C' if c < 0 else '#2ECC71' for c in contributions]

#     fig = go.Figure(go.Bar(
#         x=contributions,
#         y=features,
#         orientation='h',
#         marker_color=colors
#     ))
#     fig.update_layout(
#         title=f"<b>How Features Impact the Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#         xaxis_title="Contribution to Price (in Lakhs)",
#         yaxis=dict(autorange="reversed"),
#         template="plotly_dark",
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         margin=dict(l=120, r=20, t=60, b=50)
#     )
#     return fig

# # --- STREAMLIT UI ---
# st.title("🏎️ Car Price Prediction & Analysis")
# st.markdown("Enter the details of the car to get a predicted price from our trained model.")

# col1, col2 = st.columns([1, 1.2]) # Give the right column a bit more space

# with col1:
#     st.subheader("Enter Car Details")
#     brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#     model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand]))
#     age = st.number_input("Car Age (years)", 1, 25, 5, help="How old is the car in years?")
#     km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#     fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
#     transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#     ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
    
#     predict_button = st.button("Predict Price", type="primary", use_container_width=True)

# with col2:
#     st.subheader("Prediction Result")
#     if predict_button:
#         # 1. Create a dictionary with the EXACT column names the model expects
#         input_data = {
#             'Car_Brand': [brand],
#             'Car_Model': [model],
#             'Car_Age': [age],
#             'KM Driven': [km_driven],
#             'Fuel Type': [fuel_type],
#             'Transmission Type': [transmission],
#             'Ownership': [ownership]
#         }
        
#         try:
#             # 2. Convert the dictionary to a DataFrame
#             input_df = pd.DataFrame(input_data)
            
#             # 3. Use the trained model pipeline to predict
#             predicted_price = model_pipeline.predict(input_df)[0]
            
#             # 4. Display the result
#             st.success(f"### Predicted Price: ₹ {predicted_price:.2f} Lakhs")
            
#             # 5. Create and display the explanation plot
#             plot_inputs = {'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}
#             shap_fig = create_shap_plot(plot_inputs, predicted_price)
#             st.plotly_chart(shap_fig, use_container_width=True)

#         except Exception as e:
#             st.error(f"An error occurred during prediction: {e}")
#     else:
#         st.info("Click 'Predict Price' after entering the details to see the result.")



# import streamlit as st
# import pandas as pd
# import joblib
# import os
# import plotly.graph_objects as go

# # --- APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide"
# )

# # --- MODEL AND DATA ---
# MODEL_FILE = "car_price_predictoR.joblib"

# # Car models with fuel options
# CAR_DATA = {
#     "Maruti": {
#         "Swift": ["Petrol", "Diesel", "CNG"],
#         "Swift Dzire": ["Petrol", "Diesel"],
#         "Alto 800": ["Petrol"],
#         "Wagon R 1.0": ["Petrol", "CNG"],
#         "Ciaz": ["Petrol", "Diesel"],
#         "Ertiga": ["Petrol", "Diesel"],
#         "Vitara Brezza": ["Petrol", "Diesel"],
#         "Baleno": ["Petrol", "Diesel"],
#         "S Cross": ["Petrol", "Diesel"],
#         "Celerio": ["Petrol", "CNG"],
#         "IGNIS": ["Petrol", "Diesel", "CNG"]
#     },
#     "Mahindra": {
#         "XUV500": ["Petrol", "Diesel"],
#         "Scorpio": ["Petrol", "Diesel"],
#         "Thar": ["Petrol", "Diesel"],
#         "XUV300": ["Petrol", "Diesel"],
#         "Bolero": ["Diesel"],
#         "Marazzo": ["Diesel"],
#         "TUV300": ["Diesel"]
#     },
#     "Volkswagen": {
#         "Polo": ["Petrol", "Diesel"],
#         "Vento": ["Petrol", "Diesel"],
#         "Ameo": ["Petrol", "Diesel"],
#         "Jetta": ["Petrol", "Diesel"],
#         "Passat": ["Petrol", "Diesel"],
#         "Tiguan": ["Petrol", "Diesel"]
#     },
#     "Tata": {
#         "Nexon": ["Petrol", "Diesel", "Electric"],
#         "Harrier": ["Petrol", "Diesel"],
#         "Tiago": ["Petrol", "CNG"],
#         "Tigor": ["Petrol", "CNG"],
#         "Safari": ["Diesel"],
#         "Hexa": ["Diesel"],
#         "PUNCH": ["Petrol", "CNG"]
#     },
#     "Hyundai": {
#         "i20": ["Petrol", "Diesel", "CNG"],
#         "Creta": ["Petrol", "Diesel"],
#         "Verna": ["Petrol", "Diesel"],
#         "VENUE": ["Petrol", "Diesel", "CNG"],
#         "Grand i10": ["Petrol", "CNG"],
#         "Santro": ["Petrol", "CNG"],
#         "Xcent": ["Petrol", "Diesel", "CNG"],
#         "Aura": ["Petrol", "Diesel", "CNG"]
#     },
#     "Honda": {
#         "City": ["Petrol", "Diesel"],
#         "Amaze": ["Petrol", "Diesel"],
#         "Jazz": ["Petrol"],
#         "WR-V": ["Petrol", "Diesel"],
#         "BR-V": ["Petrol", "Diesel"],
#         "Civic": ["Petrol"]
#     },
#     "Ford": {
#         "EcoSport": ["Petrol", "Diesel"],
#         "Endeavour": ["Diesel"],
#         "Figo": ["Petrol", "Diesel"],
#         "Aspire": ["Petrol", "Diesel"],
#         "Freestyle": ["Petrol", "Diesel"]
#     },
#     "BMW": {
#         "3 Series": ["Petrol", "Diesel"],
#         "5 Series": ["Petrol", "Diesel"],
#         "X1": ["Petrol", "Diesel"],
#         "X3": ["Petrol", "Diesel"],
#         "X5": ["Petrol", "Diesel"],
#         "7 Series": ["Petrol", "Diesel"]
#     },
#     # ... continue for all remaining brands like Renault, MG, Toyota, etc.
# }

# # --- LOAD MODEL ---
# @st.cache_resource
# def load_model(model_path):
#     if not os.path.exists(model_path):
#         st.error(f"Model file not found at `{model_path}`!")
#         st.stop()
#     try:
#         model = joblib.load(model_path)
#         return model
#     except Exception as e:
#         st.error(f"❌ Error loading model: {e}")
#         st.stop()

# model_pipeline = load_model(MODEL_FILE)

# # --- HELPER FUNCTION FOR PLOTTING ---
# def create_shap_plot(inputs, final_price):
#     base_value = 8.0
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [
#         f"Age = {inputs['age']} yrs",
#         f"KM Driven = {(inputs['km']/1000):.1f}k km",
#         f"Fuel = {inputs['fuel']}",
#         f"Transmission = {inputs['transmission']}"
#     ]
#     colors = ['#E74C3C' if c < 0 else '#2ECC71' for c in contributions]

#     fig = go.Figure(go.Bar(
#         x=contributions,
#         y=features,
#         orientation='h',
#         marker_color=colors
#     ))
#     fig.update_layout(
#         title=f"<b>How Features Impact the Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#         xaxis_title="Contribution to Price (in Lakhs)",
#         yaxis=dict(autorange="reversed"),
#         template="plotly_dark",
#         plot_bgcolor="rgba(0,0,0,0)",
#         paper_bgcolor="rgba(0,0,0,0)",
#         margin=dict(l=120, r=20, t=60, b=50)
#     )
#     return fig

# # --- STREAMLIT UI ---
# st.title("🏎️ Car Price Prediction & Analysis")
# st.markdown("Enter the details of the car to get a predicted price from our trained model.")

# col1, col2 = st.columns([1, 1.2])

# with col1:
#     st.subheader("Enter Car Details")
#     brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#     model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand].keys()))
#     fuel_type = st.selectbox("Fuel Type", options=CAR_DATA[brand][model])
    
#     age = st.number_input("Car Age (years)", 1, 25, 5)
#     km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#     transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#     ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
    
#     predict_button = st.button("Predict Price", type="primary", use_container_width=True)

# with col2:
#     st.subheader("Prediction Result")
#     if predict_button:
#         input_data = {
#             'Car_Brand': [brand],
#             'Car_Model': [model],
#             'Car_Age': [age],
#             'KM Driven': [km_driven],
#             'Fuel Type': [fuel_type],
#             'Transmission Type': [transmission],
#             'Ownership': [ownership]
#         }
#         try:
#             input_df = pd.DataFrame(input_data)
#             predicted_price = model_pipeline.predict(input_df)[0]
#             st.success(f"### Predicted Price: ₹ {predicted_price:.2f} Lakhs")
#             plot_inputs = {'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}
#             shap_fig = create_shap_plot(plot_inputs, predicted_price)
#             st.plotly_chart(shap_fig, use_container_width=True)
#         except Exception as e:
#             st.error(f"An error occurred during prediction: {e}")
#     else:
#         st.info("Click 'Predict Price' after entering the details to see the result.")










# import streamlit as st
# import pandas as pd
# import random
# import plotly.express as px

# # --- APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide"
# )

# # --- DATA ---
# CAR_DATA = {
#     "Maruti": ["Swift", "Swift Dzire", "Alto 800", "Wagon R 1.0", "Ciaz", "Ertiga", "Vitara Brezza", "Baleno", "S Cross", "Celerio", "IGNIS"],
#     "Mahindra": ["XUV500", "Scorpio", "Thar", "XUV300", "Bolero", "Marazzo", "TUV300"],
#     "Volkswagen": ["Polo", "Vento", "Ameo", "Jetta", "Passat", "Tiguan"],
#     "Tata": ["Nexon", "Harrier", "Tiago", "Tigor", "Safari", "Hexa", "PUNCH"],
#     "Hyundai": ["i20", "Creta", "Verna", "VENUE", "Grand i10", "Santro", "Xcent", "Aura"],
#     "Honda": ["City", "Amaze", "Jazz", "WR-V", "BR-V", "Civic"],
#     "Ford": ["EcoSport", "Endeavour", "Figo", "Aspire", "Freestyle"],
#     "BMW": ["3 Series", "5 Series", "X1", "X3", "X5", "7 Series"],
#     "Renault": ["Kwid", "Duster", "Triber", "Kiger", "Captur"],
#     "MG": ["Hector", "Hector Plus", "Gloster", "ZS EV"],
#     "Datsun": ["redi-GO", "GO", "GO+"],
#     "Nissan": ["Magnite", "Kicks", "Terrano", "Sunny", "Micra"],
#     "Toyota": ["Innova Crysta", "Fortuner", "Yaris", "Glanza", "Urban Cruiser", "Corolla Altis"],
#     "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
#     "Jeep": ["Compass", "Wrangler", "Meridian"],
#     "KIA": ["Seltos", "Sonet", "Carnival", "Carens"],
#     "Audi": ["A4", "A6", "Q3", "Q5", "Q7"],
#     "Landrover": ["Range Rover Evoque", "Discovery Sport", "Range Rover Velar"],
#     "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
#     "Chevrolet": ["Beat", "Cruze", "Spark", "Sail", "Enjoy"],
#     "Fiat": ["Punto", "Linea"],
#     "Ssangyong": ["Rexton"],
#     "Jaguar": ["XF", "XE", "F-PACE"],
#     "Mitsubishi": ["Pajero Sport"],
#     "CITROEN": ["C5 Aircross", "C3"],
#     "Mini": ["Cooper"],
#     "ISUZU": ["D-MAX V-Cross"],
#     "Volvo": ["XC60", "XC90", "S90"],
#     "Porsche": ["Cayenne", "Macan"],
#     "Force": ["Gurkha"]
# }

# # --- MOCK PREDICTION LOGIC ---
# def predict_car_price(age, km_driven, fuel, transmission, ownership):
#     base_price = 8.0
#     base_price -= age * 0.5
#     base_price -= km_driven / 50000
#     if fuel == 'Diesel': base_price += 1
#     if transmission == 'Automatic': base_price += 1.5
#     if ownership == 'Second Owner': base_price -= 1
#     if ownership == 'Third Owner': base_price -= 2
#     return max(1.5, base_price + (random.random() - 0.5))

# # --- SHAP-LIKE FEATURE IMPACT PLOT ---
# def create_shap_plot(inputs, final_price):
#     base_value = 8.0
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [
#         f"Age = {inputs['age']} yrs",
#         f"KM Driven = {(inputs['km']/1000):.1f}k km",
#         f"Fuel = {inputs['fuel']}",
#         f"Transmission = {inputs['transmission']}"
#     ]
#     df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
#     df['Color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
    
#     fig = px.bar(
#         df, x='Contribution', y='Feature', orientation='h',
#         title=f"<b>Feature Impact on Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#         text='Contribution'
#     )
#     fig.update_traces(marker_color=df['Color'])
#     fig.update_layout(yaxis=dict(autorange="reversed"))
#     return fig

# # --- SIDEBAR NAVIGATION ---
# st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])

# # --- PAGE 1 ---
# if page == "About Me":
#     st.title("👋 About Me")
#     st.markdown("""
#     Hi, I’m **Alok Mahadev Tungal** — a passionate **Data Scientist** and **Machine Learning Enthusiast**.  
#     This app shows my journey from **data analysis ➝ ML ➝ deployment**.  
#     """)
#     st.markdown("- [LinkedIn](https://www.linkedin.com/)\n- [GitHub](https://github.com/)\n- [HuggingFace](https://huggingface.co/)")
#     st.image("https://placehold.co/300x200/4F46E5/FFFFFF?text=AMT")

# # --- PAGE 2 ---
# elif page == "The Project":
#     st.title("📊 The Project")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Dataset Rows", "9,176")
#     col2.metric("Brands Covered", "30+")
#     col3.metric("Best Model", "XGBoost (96% R²)")

# # --- PAGE 3 ---
# elif page == "Data Insights":
#     st.title("📈 Data Insights")
#     @st.cache_data
#     def generate_mock_data(n=500):
#         return pd.DataFrame({
#             'Price': [random.uniform(2, 48) for _ in range(n)],
#             'Age': [random.randint(1, 12) for _ in range(n)],
#             'Brand': [random.choice(list(CAR_DATA.keys())) for _ in range(n)]
#         })
#     df = generate_mock_data()
#     choice = st.selectbox("Select Visualization", ["Price Distribution", "Car Listings by Brand", "Price vs Age"])
#     if choice == "Price Distribution":
#         fig = px.histogram(df, x='Price', title="Distribution of Prices")
#     elif choice == "Car Listings by Brand":
#         counts = df['Brand'].value_counts().nlargest(10)
#         fig = px.bar(x=counts.values, y=counts.index, orientation='h', title="Top Brands")
#     else:
#         fig = px.scatter(df, x='Age', y='Price', color='Brand', title="Price vs Age")
#     st.plotly_chart(fig, use_container_width=True)

# # --- PAGE 4 ---
# # elif page == "Price Predictor":
# #     st.title("🔮 Price Predictor")
# #     col1, col2 = st.columns(2)
# #     with col1:
# #         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
# #         model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand]))
# #         age = st.number_input("Car Age (years)", 0, 25, 5)
# #         km = st.number_input("KM Driven", 0, 500000, 50000, 1000)
# #         fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "Electric", "LPG"])
# #         trans = st.selectbox("Transmission", ["Manual", "Automatic"])
# #         owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner"])
# #         predict = st.button("Predict Price", type="primary")
# #     with col2:
# #         if predict:
# #             price = predict_car_price(age, km, fuel, trans, owner)
# #             st.success(f"### Predicted Price: ₹ {price:.2f} Lakhs")
# #             fig = create_shap_plot({'age': age, 'km': km, 'fuel': fuel, 'transmission': trans}, price)
# #             st.plotly_chart(fig, use_container_width=True)
# #         else:
# #             st.info("Enter details and click predict.")


# # --- STREAMLIT UI ---
# elif page == "Price Predictor":
#     st.title("🔮 Price Predictor")
#     col1, col2 = st.columns(2)

# col1, col2 = st.columns([1, 1.2]) # Give the right column a bit more space

# with col1:
#     st.subheader("Enter Car Details")
#     brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#     model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand]))
#     age = st.number_input("Car Age (years)", 1, 25, 5, help="How old is the car in years?")
#     km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#     fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
#     transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#     ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
    
#     predict_button = st.button("Predict Price", type="primary", use_container_width=True)

# with col2:
#     st.subheader("Prediction Result")
#     if predict_button:
#         # 1. Create a dictionary with the EXACT column names the model expects
#         input_data = {
#             'Car_Brand': [brand],
#             'Car_Model': [model],
#             'Car_Age': [age],
#             'KM Driven': [km_driven],
#             'Fuel Type': [fuel_type],
#             'Transmission Type': [transmission],
#             'Ownership': [ownership]
#         }
        
#         try:
#             # 2. Convert the dictionary to a DataFrame
#             input_df = pd.DataFrame(input_data)
            
#             # 3. Use the trained model pipeline to predict
#             predicted_price = model_pipeline.predict(input_df)[0]
            
#             # 4. Display the result
#             st.success(f"### Predicted Price: ₹ {predicted_price:.2f} Lakhs")
            
#             # 5. Create and display the explanation plot
#             plot_inputs = {'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}
#             shap_fig = create_shap_plot(plot_inputs, predicted_price)
#             st.plotly_chart(shap_fig, use_container_width=True)

#         except Exception as e:
#             st.error(f"An error occurred during prediction: {e}")
#     else:
#         st.info("Click 'Predict Price' after entering the details to see the result.")



# app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import plotly.express as px

# st.set_page_config(page_title="Car Price Prediction", layout="wide")

# MODEL_PATH = "model.pkl"

# st.title("🚗 Car Price Prediction & Analysis (Auto-loaded Data)")

# # 🔹 Auto-load dataset (replace with your real dataset if needed)
# @st.cache_data
# def load_data():
#     data = {
#         "name": ["car_a", "car_b", "car_c","car_d","car_e"],
#         "year": [2012, 2015, 2018, 2016, 2014],
#         "selling_price": [300000, 450000, 700000, 520000, 350000],
#         "km_driven": [50000, 30000, 20000, 40000, 60000],
#         "fuel": ["Petrol","Diesel","Petrol","CNG","Petrol"],
#         "seller_type": ["Individual","Dealer","Dealer","Individual","Individual"],
#         "transmission": ["Manual","Manual","Automatic","Manual","Automatic"],
#         "owner": ["First","First","Second","First","Second"]
#     }
#     return pd.DataFrame(data)

# df = load_data()

# # Show dataset
# st.subheader("Dataset Preview")
# st.dataframe(df)

# # Features
# target = "selling_price"
# features = [c for c in df.columns if c != target and c != "name"]

# X = df[features]
# y = df[target]

# # Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Preprocess
# num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
# cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", "passthrough", num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
#     ]
# )

# # Model pipeline
# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
# ])

# # Train or load model
# if os.path.exists(MODEL_PATH):
#     pipeline = joblib.load(MODEL_PATH)
# else:
#     pipeline.fit(X_train, y_train)
#     joblib.dump(pipeline, MODEL_PATH)

# # Evaluate
# y_pred = pipeline.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)

# st.subheader("Model Performance")
# st.metric("MAE", f"{mae:.2f}")
# st.metric("RMSE", f"{rmse:.2f}")
# st.metric("R²", f"{r2:.3f}")

# # Plots
# st.subheader("EDA - Visualizations")
# fig = px.histogram(df, x="selling_price", nbins=10, title="Distribution of Car Prices")
# st.plotly_chart(fig, use_container_width=True)

# fig2 = px.scatter(df, x="year", y="selling_price", color="fuel", title="Price vs Year")
# st.plotly_chart(fig2, use_container_width=True)

# # Prediction UI
# st.subheader("🔮 Predict Car Price")
# inputs = {}
# cols = st.columns(2)
# for i, col in enumerate(features):
#     with cols[i % 2]:
#         if col in num_cols:
#             inputs[col] = st.number_input(col, value=float(X[col].median()))
#         else:
#             inputs[col] = st.selectbox(col, df[col].unique().tolist())

# if st.button("Predict"):
#     new_df = pd.DataFrame([inputs])
#     pred = pipeline.predict(new_df)[0]
#     st.success(f"Predicted Price: ₹{pred:,.0f}")





# import streamlit as st
# import pandas as pd
# import random
# import plotly.express as px
# import joblib
# import os

# # --- APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide"
# )

# # --- CAR DATA WITH FUEL OPTIONS ---
# CAR_DATA = {
#     "Maruti": {
#         "Swift": ["Petrol", "Diesel", "CNG"],
#         "Swift Dzire": ["Petrol", "Diesel"],
#         "Alto 800": ["Petrol"],
#         "Wagon R 1.0": ["Petrol", "CNG"],
#         "Ciaz": ["Petrol", "Diesel"],
#         "Ertiga": ["Petrol", "Diesel"],
#         "Vitara Brezza": ["Petrol", "Diesel"],
#         "Baleno": ["Petrol", "Diesel"],
#         "S Cross": ["Petrol", "Diesel"],
#         "Celerio": ["Petrol", "CNG"],
#         "IGNIS": ["Petrol", "Diesel", "CNG"]
#     },
#     "Mahindra": {
#         "XUV500": ["Petrol", "Diesel"],
#         "Scorpio": ["Petrol", "Diesel"],
#         "Thar": ["Petrol", "Diesel"],
#         "XUV300": ["Petrol", "Diesel"],
#         "Bolero": ["Diesel"],
#         "Marazzo": ["Diesel"],
#         "TUV300": ["Diesel"]
#     },
#     "Volkswagen": {
#         "Polo": ["Petrol", "Diesel"],
#         "Vento": ["Petrol", "Diesel"],
#         "Ameo": ["Petrol", "Diesel"],
#         "Jetta": ["Petrol", "Diesel"],
#         "Passat": ["Petrol", "Diesel"],
#         "Tiguan": ["Petrol", "Diesel"]
#     },
#     # Add remaining brands with models and fuel options as done above...
# }

# # --- MOCK PREDICTION LOGIC ---
# def predict_car_price(age, km_driven, fuel, transmission, ownership):
#     base_price = 8.0
#     base_price -= age * 0.5
#     base_price -= km_driven / 50000
#     if fuel == 'Diesel': base_price += 1
#     if transmission == 'Automatic': base_price += 1.5
#     if ownership == 'Second Owner': base_price -= 1
#     if ownership == 'Third Owner': base_price -= 2
#     return max(1.5, base_price + (random.random() - 0.5))

# # --- SHAP-LIKE FEATURE IMPACT PLOT ---
# def create_shap_plot(inputs, final_price):
#     base_value = 8.0
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [
#         f"Age = {inputs['age']} yrs",
#         f"KM Driven = {(inputs['km']/1000):.1f}k km",
#         f"Fuel = {inputs['fuel']}",
#         f"Transmission = {inputs['transmission']}"
#     ]
#     df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
#     df['Color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
    
#     fig = px.bar(
#         df, x='Contribution', y='Feature', orientation='h',
#         title=f"<b>Feature Impact on Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#         text='Contribution'
#     )
#     fig.update_traces(marker_color=df['Color'])
#     fig.update_layout(yaxis=dict(autorange="reversed"))
#     return fig

# # --- SIDEBAR NAVIGATION ---
# st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])

# # --- PAGE 1 ---
# if page == "About Me":
#     st.title("👋 About Me")
#     st.markdown("""
#     Hi, I’m **Alok Mahadev Tungal** — a passionate **Data Scientist** and **Machine Learning Enthusiast**.  
#     This app shows my journey from **data analysis ➝ ML ➝ deployment**.  
#     """)
#     st.markdown("- [LinkedIn](https://www.linkedin.com/)\n- [GitHub](https://github.com/)\n- [HuggingFace](https://huggingface.co/)")
#     st.image("https://placehold.co/300x200/4F46E5/FFFFFF?text=AMT")

# # --- PAGE 2 ---
# elif page == "The Project":
#     st.title("📊 The Project")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Dataset Rows", "9,176")
#     col2.metric("Brands Covered", "30+")
#     col3.metric("Best Model", "XGBoost (96% R²)")

# # --- PAGE 3 ---
# elif page == "Data Insights":
#     st.title("📈 Data Insights")
#     @st.cache_data
#     def generate_mock_data(n=500):
#         return pd.DataFrame({
#             'Price': [random.uniform(2, 48) for _ in range(n)],
#             'Age': [random.randint(1, 12) for _ in range(n)],
#             'Brand': [random.choice(list(CAR_DATA.keys())) for _ in range(n)]
#         })
#     df = generate_mock_data()
#     choice = st.selectbox("Select Visualization", ["Price Distribution", "Car Listings by Brand", "Price vs Age"])
#     if choice == "Price Distribution":
#         fig = px.histogram(df, x='Price', title="Distribution of Prices")
#     elif choice == "Car Listings by Brand":
#         counts = df['Brand'].value_counts().nlargest(10)
#         fig = px.bar(x=counts.values, y=counts.index, orientation='h', title="Top Brands")
#     else:
#         fig = px.scatter(df, x='Age', y='Price', color='Brand', title="Price vs Age")
#     st.plotly_chart(fig, use_container_width=True)

# # --- PAGE 4: PRICE PREDICTOR ---
# elif page == "Price Predictor":
#     st.title("🔮 Price Predictor")
#     col1, col2 = st.columns([1, 1.2])

#     with col1:
#         st.subheader("Enter Car Details")
#         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#         model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand].keys()))
#         fuel_type = st.selectbox("Fuel Type", options=CAR_DATA[brand][model])
#         age = st.number_input("Car Age (years)", 1, 25, 5)
#         km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#         transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#         ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
#         predict_button = st.button("Predict Price", type="primary", use_container_width=True)

#     with col2:
#         st.subheader("Prediction Result")
#         if predict_button:
#             price = predict_car_price(age, km_driven, fuel_type, transmission, ownership)
#             st.success(f"### Predicted Price: ₹ {price:.2f} Lakhs")
#             fig = create_shap_plot({'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}, price)
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Enter details and click predict.")




# import streamlit as st
# import pandas as pd
# import random
# import plotly.express as px
# import joblib
# import os

# # --- APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide"
# )

# # --- LOAD MODEL (OPTIONAL) ---
# MODEL_FILE = "car_price_predictoR.joblib"
# @st.cache_resource
# def load_model(model_path):
#     if os.path.exists(model_path):
#         return joblib.load(model_path)
#     return None

# model_pipeline = load_model(MODEL_FILE)

# # --- CAR DATA WITH FUEL OPTIONS (ALL 30+ BRANDS) ---
# CAR_DATA = {
#     "Maruti": {"Swift": ["Petrol", "Diesel", "CNG"], "Swift Dzire": ["Petrol", "Diesel"], "Alto 800": ["Petrol"],
#                "Wagon R 1.0": ["Petrol", "CNG"], "Ciaz": ["Petrol", "Diesel"], "Ertiga": ["Petrol", "Diesel"],
#                "Vitara Brezza": ["Petrol", "Diesel"], "Baleno": ["Petrol", "Diesel"], "S Cross": ["Petrol", "Diesel"],
#                "Celerio": ["Petrol", "CNG"], "IGNIS": ["Petrol", "Diesel", "CNG"]},
#     "Mahindra": {"XUV500": ["Petrol", "Diesel"], "Scorpio": ["Petrol", "Diesel"], "Thar": ["Petrol", "Diesel"],
#                  "XUV300": ["Petrol", "Diesel"], "Bolero": ["Diesel"], "Marazzo": ["Diesel"], "TUV300": ["Diesel"]},
#     "Volkswagen": {"Polo": ["Petrol", "Diesel"], "Vento": ["Petrol", "Diesel"], "Ameo": ["Petrol", "Diesel"],
#                    "Jetta": ["Petrol", "Diesel"], "Passat": ["Petrol", "Diesel"], "Tiguan": ["Petrol", "Diesel"]},
#     "Tata": {"Nexon": ["Petrol", "Diesel", "Electric"], "Harrier": ["Petrol", "Diesel"], "Tiago": ["Petrol", "Diesel"],
#              "Tigor": ["Petrol", "Diesel"], "Safari": ["Diesel"], "Hexa": ["Diesel"], "PUNCH": ["Petrol"]},
#     "Hyundai": {"i20": ["Petrol", "Diesel"], "Creta": ["Petrol", "Diesel"], "Verna": ["Petrol", "Diesel"],
#                 "VENUE": ["Petrol", "Diesel"], "Grand i10": ["Petrol", "CNG"], "Santro": ["Petrol", "CNG"],
#                 "Xcent": ["Petrol", "Diesel"], "Aura": ["Petrol", "Diesel"]},
#     "Honda": {"City": ["Petrol", "Diesel"], "Amaze": ["Petrol", "Diesel"], "Jazz": ["Petrol", "Diesel"],
#               "WR-V": ["Petrol", "Diesel"], "BR-V": ["Petrol", "Diesel"], "Civic": ["Petrol", "Diesel"]},
#     "Ford": {"EcoSport": ["Petrol", "Diesel"], "Endeavour": ["Diesel"], "Figo": ["Petrol", "Diesel"],
#              "Aspire": ["Petrol", "Diesel"], "Freestyle": ["Petrol", "Diesel"]},
#     "BMW": {"3 Series": ["Petrol", "Diesel"], "5 Series": ["Petrol", "Diesel"], "X1": ["Petrol", "Diesel"],
#             "X3": ["Petrol", "Diesel"], "X5": ["Petrol", "Diesel", "Hybrid"], "7 Series": ["Petrol", "Diesel"]},
#     "Renault": {"Kwid": ["Petrol"], "Duster": ["Petrol", "Diesel"], "Triber": ["Petrol"], "Kiger": ["Petrol", "Diesel"],
#                 "Captur": ["Petrol", "Diesel"]},
#     "MG": {"Hector": ["Petrol", "Diesel", "Hybrid"], "Hector Plus": ["Petrol", "Diesel"], "Gloster": ["Diesel"],
#            "ZS EV": ["Electric"]},
#     "Datsun": {"redi-GO": ["Petrol"], "GO": ["Petrol"], "GO+": ["Petrol"]},
#     "Nissan": {"Magnite": ["Petrol", "Diesel"], "Kicks": ["Petrol"], "Terrano": ["Diesel"], "Sunny": ["Petrol"],
#                "Micra": ["Petrol"]},
#     "Toyota": {"Innova Crysta": ["Diesel", "Petrol"], "Fortuner": ["Diesel", "Petrol"], "Yaris": ["Petrol"],
#                "Glanza": ["Petrol"], "Urban Cruiser": ["Petrol"], "Corolla Altis": ["Petrol", "Hybrid"]},
#     "Skoda": {"Rapid": ["Petrol", "Diesel"], "Octavia": ["Petrol", "Diesel"], "Superb": ["Petrol", "Diesel"],
#               "Kushaq": ["Petrol", "Diesel"], "Slavia": ["Petrol", "Diesel"]},
#     "Jeep": {"Compass": ["Petrol", "Diesel"], "Wrangler": ["Petrol", "Diesel"], "Meridian": ["Diesel"]},
#     "KIA": {"Seltos": ["Petrol", "Diesel"], "Sonet": ["Petrol", "Diesel"], "Carnival": ["Diesel"], "Carens": ["Petrol", "Diesel"]},
#     "Audi": {"A4": ["Petrol", "Diesel"], "A6": ["Petrol", "Diesel"], "Q3": ["Petrol", "Diesel"], "Q5": ["Petrol", "Diesel"], "Q7": ["Petrol", "Diesel"]},
#     "Landrover": {"Range Rover Evoque": ["Petrol", "Diesel"], "Discovery Sport": ["Diesel"], "Range Rover Velar": ["Petrol", "Diesel"]},
#     "Mercedes": {"C-Class": ["Petrol", "Diesel"], "E-Class": ["Petrol", "Diesel"], "GLC": ["Petrol", "Diesel"],
#                  "GLE": ["Petrol", "Diesel"], "S-Class": ["Petrol", "Diesel"]},
#     "Chevrolet": {"Beat": ["Petrol"], "Cruze": ["Diesel"], "Spark": ["Petrol"], "Sail": ["Petrol"], "Enjoy": ["Diesel"]},
#     "Fiat": {"Punto": ["Petrol", "Diesel"], "Linea": ["Petrol", "Diesel"]},
#     "Ssangyong": {"Rexton": ["Diesel"]},
#     "Jaguar": {"XF": ["Petrol", "Diesel"], "XE": ["Petrol", "Diesel"], "F-PACE": ["Petrol", "Diesel"]},
#     "Mitsubishi": {"Pajero Sport": ["Diesel"]},
#     "CITROEN": {"C5 Aircross": ["Diesel", "Petrol"], "C3": ["Petrol"]},
#     "Mini": {"Cooper": ["Petrol", "Diesel"]},
#     "ISUZU": {"D-MAX V-Cross": ["Diesel"]},
#     "Volvo": {"XC60": ["Petrol", "Diesel", "Hybrid"], "XC90": ["Petrol", "Diesel", "Hybrid"], "S90": ["Petrol", "Diesel"]},
#     "Porsche": {"Cayenne": ["Petrol", "Diesel", "Hybrid"], "Macan": ["Petrol", "Diesel"]},
#     "Force": {"Gurkha": ["Diesel"]}
# }

# # --- MOCK PREDICTION LOGIC ---
# def predict_car_price(age, km_driven, fuel, transmission, ownership):
#     if model_pipeline:
#         input_data = pd.DataFrame({
#             'Car_Brand': [brand],
#             'Car_Model': [model],
#             'Car_Age': [age],
#             'KM Driven': [km_driven],
#             'Fuel Type': [fuel],
#             'Transmission Type': [transmission],
#             'Ownership': [ownership]
#         })
#         return model_pipeline.predict(input_data)[0]
    
#     base_price = 8.0
#     base_price -= age * 0.5
#     base_price -= km_driven / 50000
#     if fuel == 'Diesel': base_price += 1
#     if transmission == 'Automatic': base_price += 1.5
#     if ownership == 'Second Owner': base_price -= 1
#     if ownership == 'Third Owner': base_price -= 2
#     return max(1.5, base_price + (random.random() - 0.5))

# # --- SHAP-LIKE PLOT ---
# def create_shap_plot(inputs, final_price):
#     base_value = 8.0
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [
#         f"Age = {inputs['age']} yrs",
#         f"KM Driven = {(inputs['km']/1000):.1f}k km",
#         f"Fuel = {inputs['fuel']}",
#         f"Transmission = {inputs['transmission']}"
#     ]
#     df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
#     df['Color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
    
#     fig = px.bar(
#         df, x='Contribution', y='Feature', orientation='h',
#         title=f"<b>Feature Impact on Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#         text='Contribution'
#     )
#     fig.update_traces(marker_color=df['Color'])
#     fig.update_layout(yaxis=dict(autorange="reversed"))
#     return fig

# # --- SIDEBAR NAVIGATION ---
# st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])

# # --- PAGE 1: ABOUT ME ---
# if page == "About Me":
#     st.title("👋 About Me")
#     st.markdown("""
#     Hi, I’m **Alok Mahadev Tungal** — a passionate **Data Scientist** and **Machine Learning Enthusiast**.  
#     This app showcases my journey from **data analysis ➝ ML ➝ deployment**.  
#     """)
#     st.markdown("- [LinkedIn](https://www.linkedin.com/)\n- [GitHub](https://github.com/)\n- [HuggingFace](https://huggingface.co/)")
#     st.image("https://placehold.co/300x200/4F46E5/FFFFFF?text=AMT")

# # --- PAGE 2: THE PROJECT ---
# elif page == "The Project":
#     st.title("📊 The Project")
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Dataset Rows", "9,176")
#     col2.metric("Brands Covered", "30+")
#     col3.metric("Best Model", "XGBoost (96% R²)")

# # --- PAGE 3: DATA INSIGHTS / EDA ---
# elif page == "Data Insights":
#     st.title("📈 Data Insights / EDA")
#     @st.cache_data
#     def generate_mock_data(n=500):
#         return pd.DataFrame({
#             'Price': [random.uniform(2, 48) for _ in range(n)],
#             'Age': [random.randint(1, 12) for _ in range(n)],
#             'Brand': [random.choice(list(CAR_DATA.keys())) for _ in range(n)]
#         })
#     df = generate_mock_data()
#     choice = st.selectbox("Select Visualization", ["Price Distribution", "Car Listings by Brand", "Price vs Age"])
#     if choice == "Price Distribution":
#         fig = px.histogram(df, x='Price', title="Distribution of Prices")
#     elif choice == "Car Listings by Brand":
#         counts = df['Brand'].value_counts().nlargest(10)
#         fig = px.bar(x=counts.values, y=counts.index, orientation='h', title="Top Brands")
#     else:
#         fig = px.scatter(df, x='Age', y='Price', color='Brand', title="Price vs Age")
#     st.plotly_chart(fig, use_container_width=True)

# # --- PAGE 4: PRICE PREDICTOR ---
# elif page == "Price Predictor":
#     st.title("🔮 Price Predictor")
#     col1, col2 = st.columns([1, 1.2])

#     with col1:
#         st.subheader("Enter Car Details")
#         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#         model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand].keys()))
#         fuel_type = st.selectbox("Fuel Type", options=CAR_DATA[brand][model])
#         age = st.number_input("Car Age (years)", 1, 25, 5)
#         km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#         transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#         ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
#         predict_button = st.button("Predict Price", type="primary", use_container_width=True)

#     with col2:
#         st.subheader("Prediction Result")
#         if predict_button:
#             price = predict_car_price(age, km_driven, fuel_type, transmission, ownership)
#             st.success(f"### Predicted Price: ₹ {price:.2f} Lakhs")
#             fig = create_shap_plot({'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}, price)
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Enter details and click predict.")






# import streamlit as st
# import pandas as pd
# import random
# import plotly.express as px
# import joblib
# import os
# import numpy as np

# # --- 1. APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- 2. LOAD MODEL & DATA ---
# MODEL_FILE = "car_price_predictoR.joblib"

# @st.cache_resource
# def load_model(model_path):
#     """Load the pre-trained model, return None if it doesn't exist."""
#     if os.path.exists(model_path):
#         try:
#             return joblib.load(model_path)
#         except Exception as e:
#             st.sidebar.error(f"Error loading model: {e}")
#             return None
#     return None

# model_pipeline = load_model(MODEL_FILE)
# if model_pipeline:
#     st.sidebar.success("✅ Trained model loaded successfully!")
# else:
#     st.sidebar.warning("⚠️ Trained model not found. Using mock prediction logic.")

# # Using a simplified version for mock data generation
# ALL_BRANDS = ["Maruti", "Hyundai", "Honda", "Mahindra", "Tata", "Ford", "Toyota", "Renault", "Volkswagen", "BMW"]
# CAR_DATA = {
#     "Maruti": {"Swift": ["Petrol", "Diesel"], "Baleno": ["Petrol"]},
#     "Hyundai": {"i20": ["Petrol", "Diesel"], "Creta": ["Petrol", "Diesel"]},
#     "Honda": {"City": ["Petrol"], "Amaze": ["Petrol", "Diesel"]},
#     "Mahindra": {"XUV500": ["Diesel"], "Scorpio": ["Diesel"]},
#     "Tata": {"Nexon": ["Petrol", "Diesel"], "Harrier": ["Diesel"]},
#     "Ford": {"EcoSport": ["Petrol", "Diesel"], "Endeavour": ["Diesel"]},
#     "Toyota": {"Innova Crysta": ["Diesel"], "Fortuner": ["Diesel"]},
#     "Renault": {"Kwid": ["Petrol"], "Duster": ["Petrol"]},
#     "Volkswagen": {"Polo": ["Petrol", "Diesel"], "Vento": ["Petrol"]},
#     "BMW": {"3 Series": ["Petrol", "Diesel"], "X1": ["Diesel"]},
# }

# # --- 3. PREDICTION & PLOTTING FUNCTIONS ---
# def predict_car_price(brand, model, age, km_driven, fuel, transmission, ownership):
#     """Predicts car price using the loaded model or a mock function."""
#     if model_pipeline:
#         # Use the actual trained model if it's loaded
#         input_data = pd.DataFrame({
#             'Car_Brand': [brand], 'Car_Model': [model], 'Car_Age': [age],
#             'KM Driven': [km_driven], 'Fuel Type': [fuel],
#             'Transmission Type': [transmission], 'Ownership': [ownership]
#         })
#         return model_pipeline.predict(input_data)[0]
#     else:
#         # Fallback to mock logic if the model file is not found
#         base_price = 8.0 - (age * 0.5) - (km_driven / 50000)
#         if fuel == 'Diesel': base_price += 1.0
#         if transmission == 'Automatic': base_price += 1.5
#         if ownership == 'Second Owner': base_price -= 1.0
#         elif ownership == 'Third Owner': base_price -= 2.0
#         return max(1.5, base_price + (random.random() - 0.5))

# def create_shap_plot(inputs, final_price):
#     """Creates a mock feature impact plot for visualization."""
#     base_value = 8.0
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [f"Age = {inputs['age']} yrs", f"KM Driven = {inputs['km']/1000:.1f}k km",
#                 f"Fuel = {inputs['fuel']}", f"Transmission = {inputs['transmission']}"]
#     df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
#     df['Color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
#     fig = px.bar(df, x='Contribution', y='Feature', orientation='h',
#                  title=f"<b>Feature Impact on Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#                  text='Contribution', template="plotly_white")
#     fig.update_traces(marker_color=df['Color'], texttemplate='%{text:.2f}', textposition='outside')
#     fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Contribution to Price (in Lakhs)")
#     return fig

# # --- 4. SIDEBAR NAVIGATION ---
# st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])
# st.sidebar.info("This app demonstrates a complete ML project pipeline for predicting used car prices.")

# # --- 5. PAGE CONTENT ---

# # --- PAGE 1: ABOUT ME ---
# if page == "About Me":
#     st.title("👋 About Me")
#     st.markdown("---")
#     col1, col2 = st.columns([2, 1.5], gap="large")
#     with col1:
#         st.header("Alok Mahadev Tungal")
#         st.markdown("""
#         A passionate **Data Scientist** and **Machine Learning Engineer** with a knack for turning complex datasets into actionable insights. My journey in tech is driven by a relentless curiosity and a desire to build intelligent solutions that solve real-world problems.

#         This application is a demonstration of an end-to-end machine learning project, from data exploration to model deployment.
#         """)
#         st.markdown("#### Key Skills:")
#         st.code("""
# - Python (Pandas, NumPy, Scikit-learn, TensorFlow)
# - Machine Learning (Regression, Classification, Clustering)
# - Data Visualization (Plotly, Matplotlib, Seaborn)
# - Web Frameworks (Streamlit, Flask)
# - Databases (SQL)
#         """)
#         st.markdown("#### Find me on:")
#         st.markdown("[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | [HuggingFace](https://huggingface.co/)")

#     with col2:
#         st.image("https://placehold.co/500x500/4F46E5/FFFFFF?text=AMT",
#                  caption="Alok Mahadev Tungal", use_column_width=True)

# # --- PAGE 2: THE PROJECT ---
# elif page == "The Project":
#     st.title("📊 The Project: Used Car Price Prediction")
#     st.markdown("---")
#     st.markdown("### Project Goal")
#     st.info("The primary objective of this project is to develop a robust machine learning model that can accurately predict the price of used cars based on their features like brand, model, age, mileage, and fuel type.")

#     st.markdown("### Methodology")
#     st.markdown("""
#     The project follows a standard data science pipeline, ensuring a structured and effective workflow from start to finish.
#     """)
#     with st.expander("Click to see the detailed project pipeline"):
#         st.write("""
#         1.  **📝 Data Collection:** The dataset was sourced from a public repository, containing over 9,000 listings of used cars across India.
#         2.  **🧹 Data Cleaning & Preprocessing:** Handled missing values, removed duplicates, and corrected data types to prepare the data for analysis.
#         3.  **📈 Exploratory Data Analysis (EDA):** Generated various visualizations to understand the relationships between different car features and their impact on price. Key insights were drawn from distributions, correlations, and categorical breakdowns.
#         4.  **🛠️ Feature Engineering:** Created new features like 'Car Age' from the 'Year' of manufacture to improve model performance. Categorical features were encoded using techniques like one-hot encoding.
#         5.  **🤖 Model Training & Selection:** Trained several regression models, including Linear Regression, Random Forest, and Gradient Boosting. **XGBoost Regressor** was selected as the final model due to its superior performance.
#         6.  **튜닝 Hyperparameter Tuning:** Used techniques like GridSearchCV to find the optimal set of hyperparameters for the XGBoost model, boosting its accuracy from 94% to **96% (R² Score)**.
#         7.  **🚀 Deployment:** The trained model was serialized using `joblib` and deployed as an interactive web application using **Streamlit**, hosted on **Hugging Face Spaces**.
#         """)
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Dataset Rows", "9,176")
#     col2.metric("Features Used", "7")
#     col3.metric("Best Model", "XGBoost (96% R²)")


# # --- PAGE 3: DATA INSIGHTS / EDA ---
# elif page == "Data Insights":
#     st.title("📈 Exploratory Data Analysis (EDA)")
#     st.markdown("---")

#     @st.cache_data
#     def generate_mock_data(n=500):
#         """Generates a more detailed mock DataFrame for EDA."""
#         data = {
#             'Price': np.random.uniform(2.5, 45.0, n),
#             'Age': np.random.randint(1, 12, n),
#             'KM Driven': np.random.randint(10000, 150000, n),
#             'Brand': [random.choice(ALL_BRANDS) for _ in range(n)],
#             'Fuel Type': [random.choice(['Petrol', 'Diesel']) for _ in range(n)],
#             'Transmission': [random.choice(['Manual', 'Automatic']) for _ in range(n)],
#             'Ownership': [random.choice(['First Owner', 'Second Owner', 'Third Owner']) for _ in range(n)]
#         }
#         return pd.DataFrame(data)

#     df = generate_mock_data()

#     # Create two columns for a cleaner layout
#     col1, col2 = st.columns([1, 2])
    
#     with col1:
#         st.subheader("Select a Visualization")
#         plot_choice = st.radio(
#             "Choose a chart:",
#             [
#                 "Correlation Heatmap",
#                 "Price Distribution by Fuel Type",
#                 "Price vs. Car Age",
#                 "Price by Ownership Type"
#             ]
#         )

#     with col2:
#         if plot_choice == "Correlation Heatmap":
#             st.subheader("Numerical Feature Correlation")
#             corr = df[['Price', 'Age', 'KM Driven']].corr()
#             fig = px.imshow(corr, text_auto=True, aspect="auto",
#                             title="Heatmap of Price, Age, and Mileage", color_continuous_scale='RdBu_r')
#             st.plotly_chart(fig, use_container_width=True)

#         elif plot_choice == "Price Distribution by Fuel Type":
#             st.subheader("Price Distribution: Petrol vs. Diesel")
#             fig = px.box(df, x='Fuel Type', y='Price', color='Fuel Type',
#                          title="Median Price for Petrol vs. Diesel Cars",
#                          labels={"Price": "Price (in Lakhs)"})
#             st.plotly_chart(fig, use_container_width=True)

#         elif plot_choice == "Price vs. Car Age":
#             st.subheader("Price Depreciation Over Time")
#             fig = px.scatter(df, x='Age', y='Price', color='Brand',
#                              title="Price vs. Age of Car",
#                              labels={"Price": "Price (in Lakhs)", "Age": "Age (Years)"})
#             st.plotly_chart(fig, use_container_width=True)

#         elif plot_choice == "Price by Ownership Type":
#             st.subheader("Impact of Ownership on Price")
#             fig = px.box(df, x='Ownership', y='Price', color='Ownership',
#                          title="Price Distribution by Number of Owners",
#                          labels={"Price": "Price (in Lakhs)"},
#                          category_orders={"Ownership": ["First Owner", "Second Owner", "Third Owner"]})
#             st.plotly_chart(fig, use_container_width=True)

# # --- PAGE 4: PRICE PREDICTOR ---
# elif page == "Price Predictor":
#     st.title("🔮 Price Predictor")
#     st.markdown("---")
#     st.markdown("Fill in the car's details below to get an estimated market price.")

#     col1, col2 = st.columns([1, 1.2])

#     with col1:
#         st.subheader("Enter Car Details")
#         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#         model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand].keys()))
#         fuel_type = st.selectbox("Fuel Type", options=CAR_DATA[brand][model])
#         age = st.number_input("Car Age (years)", 1, 25, 5)
#         km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#         transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#         ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
#         predict_button = st.button("Predict Price", type="primary", use_container_width=True)

#     with col2:
#         st.subheader("Prediction Result")
#         if predict_button:
#             price = predict_car_price(brand, model, age, km_driven, fuel_type, transmission, ownership)
#             st.success(f"### Predicted Price: ₹ {price:.2f} Lakhs")
#             with st.expander("See Feature Impact on Price"):
#                 fig = create_shap_plot({'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}, price)
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Enter the car details and click 'Predict Price' to see the estimated value.")




# import streamlit as st
# import pandas as pd
# import random
# import plotly.express as px
# import joblib
# import os
# import numpy as np

# # --- 1. APP CONFIGURATION ---
# st.set_page_config(
#     page_title="Car Price Prediction & Analysis",
#     page_icon="🏎️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # --- 2. LOAD MODEL & FULL CAR DATA ---
# MODEL_FILE = "car_price_predictoR.joblib"

# @st.cache_resource
# def load_model(model_path):
#     """Load the pre-trained model, return None if it doesn't exist."""
#     if os.path.exists(model_path):
#         try:
#             return joblib.load(model_path)
#         except Exception as e:
#             st.sidebar.error(f"Error loading model: {e}")
#             return None
#     return None

# model_pipeline = load_model(MODEL_FILE)
# if model_pipeline:
#     st.sidebar.success("✅ Trained model loaded successfully!")
# else:
#     st.sidebar.warning("⚠️ Trained model not found. Using mock prediction logic.")

# # Your full list of 30+ car brands and models
# CAR_DATA = {
#     "Maruti": {"Swift": ["Petrol", "Diesel", "CNG"], "Swift Dzire": ["Petrol", "Diesel"], "Alto 800": ["Petrol"], "Wagon R 1.0": ["Petrol", "CNG"], "Ciaz": ["Petrol", "Diesel"], "Ertiga": ["Petrol", "Diesel"], "Vitara Brezza": ["Petrol", "Diesel"], "Baleno": ["Petrol", "Diesel"], "S Cross": ["Petrol", "Diesel"], "Celerio": ["Petrol", "CNG"], "IGNIS": ["Petrol", "Diesel", "CNG"]},
#     "Mahindra": {"XUV500": ["Petrol", "Diesel"], "Scorpio": ["Petrol", "Diesel"], "Thar": ["Petrol", "Diesel"], "XUV300": ["Petrol", "Diesel"], "Bolero": ["Diesel"], "Marazzo": ["Diesel"], "TUV300": ["Diesel"]},
#     "Volkswagen": {"Polo": ["Petrol", "Diesel"], "Vento": ["Petrol", "Diesel"], "Ameo": ["Petrol", "Diesel"], "Jetta": ["Petrol", "Diesel"], "Passat": ["Petrol", "Diesel"], "Tiguan": ["Petrol", "Diesel"]},
#     "Tata": {"Nexon": ["Petrol", "Diesel", "Electric"], "Harrier": ["Petrol", "Diesel"], "Tiago": ["Petrol", "Diesel"], "Tigor": ["Petrol", "Diesel"], "Safari": ["Diesel"], "Hexa": ["Diesel"], "PUNCH": ["Petrol"]},
#     "Hyundai": {"i20": ["Petrol", "Diesel"], "Creta": ["Petrol", "Diesel"], "Verna": ["Petrol", "Diesel"], "VENUE": ["Petrol", "Diesel"], "Grand i10": ["Petrol", "CNG"], "Santro": ["Petrol", "CNG"], "Xcent": ["Petrol", "Diesel"], "Aura": ["Petrol", "Diesel"]},
#     "Honda": {"City": ["Petrol", "Diesel"], "Amaze": ["Petrol", "Diesel"], "Jazz": ["Petrol", "Diesel"], "WR-V": ["Petrol", "Diesel"], "BR-V": ["Petrol", "Diesel"], "Civic": ["Petrol", "Diesel"]},
#     "Ford": {"EcoSport": ["Petrol", "Diesel"], "Endeavour": ["Diesel"], "Figo": ["Petrol", "Diesel"], "Aspire": ["Petrol", "Diesel"], "Freestyle": ["Petrol", "Diesel"]},
#     "BMW": {"3 Series": ["Petrol", "Diesel"], "5 Series": ["Petrol", "Diesel"], "X1": ["Petrol", "Diesel"], "X3": ["Petrol", "Diesel"], "X5": ["Petrol", "Diesel", "Hybrid"], "7 Series": ["Petrol", "Diesel"]},
#     "Renault": {"Kwid": ["Petrol"], "Duster": ["Petrol", "Diesel"], "Triber": ["Petrol"], "Kiger": ["Petrol", "Diesel"], "Captur": ["Petrol", "Diesel"]},
#     "MG": {"Hector": ["Petrol", "Diesel", "Hybrid"], "Hector Plus": ["Petrol", "Diesel"], "Gloster": ["Diesel"], "ZS EV": ["Electric"]},
#     "Datsun": {"redi-GO": ["Petrol"], "GO": ["Petrol"], "GO+": ["Petrol"]},
#     "Nissan": {"Magnite": ["Petrol", "Diesel"], "Kicks": ["Petrol"], "Terrano": ["Diesel"], "Sunny": ["Petrol"], "Micra": ["Petrol"]},
#     "Toyota": {"Innova Crysta": ["Diesel", "Petrol"], "Fortuner": ["Diesel", "Petrol"], "Yaris": ["Petrol"], "Glanza": ["Petrol"], "Urban Cruiser": ["Petrol"], "Corolla Altis": ["Petrol", "Hybrid"]},
#     "Skoda": {"Rapid": ["Petrol", "Diesel"], "Octavia": ["Petrol", "Diesel"], "Superb": ["Petrol", "Diesel"], "Kushaq": ["Petrol", "Diesel"], "Slavia": ["Petrol", "Diesel"]},
#     "Jeep": {"Compass": ["Petrol", "Diesel"], "Wrangler": ["Petrol", "Diesel"], "Meridian": ["Diesel"]},
#     "KIA": {"Seltos": ["Petrol", "Diesel"], "Sonet": ["Petrol", "Diesel"], "Carnival": ["Diesel"], "Carens": ["Petrol", "Diesel"]},
#     "Audi": {"A4": ["Petrol", "Diesel"], "A6": ["Petrol", "Diesel"], "Q3": ["Petrol", "Diesel"], "Q5": ["Petrol", "Diesel"], "Q7": ["Petrol", "Diesel"]},
#     "Landrover": {"Range Rover Evoque": ["Petrol", "Diesel"], "Discovery Sport": ["Diesel"], "Range Rover Velar": ["Petrol", "Diesel"]},
#     "Mercedes": {"C-Class": ["Petrol", "Diesel"], "E-Class": ["Petrol", "Diesel"], "GLC": ["Petrol", "Diesel"], "GLE": ["Petrol", "Diesel"], "S-Class": ["Petrol", "Diesel"]},
#     "Chevrolet": {"Beat": ["Petrol"], "Cruze": ["Diesel"], "Spark": ["Petrol"], "Sail": ["Petrol"], "Enjoy": ["Diesel"]},
#     "Fiat": {"Punto": ["Petrol", "Diesel"], "Linea": ["Petrol", "Diesel"]},
#     "Ssangyong": {"Rexton": ["Diesel"]},
#     "Jaguar": {"XF": ["Petrol", "Diesel"], "XE": ["Petrol", "Diesel"], "F-PACE": ["Petrol", "Diesel"]},
#     "Mitsubishi": {"Pajero Sport": ["Diesel"]},
#     "CITROEN": {"C5 Aircross": ["Diesel", "Petrol"], "C3": ["Petrol"]},
#     "Mini": {"Cooper": ["Petrol", "Diesel"]},
#     "ISUZU": {"D-MAX V-Cross": ["Diesel"]},
#     "Volvo": {"XC60": ["Petrol", "Diesel", "Hybrid"], "XC90": ["Petrol", "Diesel", "Hybrid"], "S90": ["Petrol", "Diesel"]},
#     "Porsche": {"Cayenne": ["Petrol", "Diesel", "Hybrid"], "Macan": ["Petrol", "Diesel"]},
#     "Force": {"Gurkha": ["Diesel"]}
# }
# ALL_BRANDS = list(CAR_DATA.keys())


# # --- 3. PREDICTION & PLOTTING FUNCTIONS ---
# def predict_car_price(brand, model, age, km_driven, fuel, transmission, ownership):
#     """Predicts car price using the loaded model or a mock function."""
#     if model_pipeline:
#         input_data = pd.DataFrame({
#             'Car_Brand': [brand], 'Car_Model': [model], 'Car_Age': [age],
#             'KM Driven': [km_driven], 'Fuel Type': [fuel],
#             'Transmission Type': [transmission], 'Ownership': [ownership]
#         })
#         return model_pipeline.predict(input_data)[0]
#     else:
#         base_price = 8.0 - (age * 0.5) - (km_driven / 50000)
#         if fuel == 'Diesel': base_price += 1.0
#         if transmission == 'Automatic': base_price += 1.5
#         if ownership == 'Second Owner': base_price -= 1.0
#         elif ownership == 'Third Owner': base_price -= 2.0
#         return max(1.5, base_price + (random.random() - 0.5))

# def create_shap_plot(inputs, final_price):
#     """Creates a mock feature impact plot for visualization."""
#     base_value = 8.0
#     contributions = [
#         -(inputs['age'] * 0.5),
#         -(inputs['km'] / 50000),
#         1.0 if inputs['fuel'] == 'Diesel' else -0.2,
#         1.5 if inputs['transmission'] == 'Automatic' else -0.5
#     ]
#     features = [f"Age = {inputs['age']} yrs", f"KM Driven = {inputs['km']/1000:.1f}k km",
#                 f"Fuel = {inputs['fuel']}", f"Transmission = {inputs['transmission']}"]
#     df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
#     df['Color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
#     fig = px.bar(df, x='Contribution', y='Feature', orientation='h',
#                  title=f"<b>Feature Impact on Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
#                  text='Contribution', template="plotly_white")
#     fig.update_traces(marker_color=df['Color'], texttemplate='%{text:.2f}', textposition='outside')
#     fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Contribution to Price (in Lakhs)")
#     return fig

# # --- 4. SIDEBAR NAVIGATION ---
# st.sidebar.title("📌 Navigation")
# page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])
# st.sidebar.info("This app demonstrates a complete ML project pipeline for predicting used car prices.")

# # --- 5. PAGE CONTENT ---

# # --- PAGE 1: ABOUT ME ---
# if page == "About Me":
#     st.title("👋 About Me")
#     st.markdown("---")
#     col1, col2 = st.columns([2, 1.5], gap="large")
#     with col1:
#         st.header("Alok Mahadev Tungal")
#         st.markdown("""
#         A passionate **Data Scientist** and **Machine Learning Engineer** with a knack for turning complex datasets into actionable insights. My journey in tech is driven by a relentless curiosity and a desire to build intelligent solutions that solve real-world problems.

#         This application is a demonstration of an end-to-end machine learning project, from data exploration to model deployment.
#         """)
#         st.markdown("#### Key Skills:")
#         st.code("""
# - Python (Pandas, NumPy, Scikit-learn, TensorFlow)
# - Machine Learning (Regression, Classification, Clustering)
# - Data Visualization (Plotly, Matplotlib, Seaborn)
# - Web Frameworks (Streamlit, Flask)
# - Databases (SQL)
#         """)
#         st.markdown("#### Find me on:")
#         st.markdown("[LinkedIn](https://www.linkedin.com/) | [GitHub](https://github.com/) | [HuggingFace](https://huggingface.co/)")

#     with col2:
#         st.image("https://placehold.co/500x500/4F46E5/FFFFFF?text=AMT",
#                  caption="Alok Mahadev Tungal", use_column_width=True)

# # --- PAGE 2: THE PROJECT ---
# elif page == "The Project":
#     st.title("📊 The Project: Used Car Price Prediction")
#     st.markdown("---")
#     st.markdown("### Project Goal")
#     st.info("The primary objective of this project is to develop a robust machine learning model that can accurately predict the price of used cars based on their features like brand, model, age, mileage, and fuel type.")

#     st.markdown("### Methodology")
#     st.markdown("The project follows a standard data science pipeline, ensuring a structured and effective workflow.")
#     with st.expander("Click to see the detailed project pipeline"):
#         st.write("""
#         1.  **📝 Data Collection:** The dataset was sourced from a public repository, containing over 9,000 listings of used cars across India.
#         2.  **🧹 Data Cleaning & Preprocessing:** Handled missing values, removed duplicates, and corrected data types.
#         3.  **📈 Exploratory Data Analysis (EDA):** Generated various visualizations to understand relationships and impact on price.
#         4.  **🛠️ Feature Engineering:** Created new features like 'Car Age' from 'Year' to improve model performance.
#         5.  **🤖 Model Training & Selection:** Trained several models, with **XGBoost Regressor** selected for its superior performance.
#         6.  **튜닝 Hyperparameter Tuning:** Used GridSearchCV to find optimal hyperparameters, boosting accuracy to **96% (R² Score)**.
#         7.  **🚀 Deployment:** The trained model was deployed as an interactive web application using **Streamlit**.
#         """)
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Dataset Rows", "9,176")
#     col2.metric("Features Used", "7")
#     col3.metric("Best Model", "XGBoost (96% R²)")


# # --- PAGE 3: DATA INSIGHTS / EDA ---
# elif page == "Data Insights":
#     st.title("📈 Exploratory Data Analysis (EDA)")
#     st.markdown("---")

#     @st.cache_data
#     def generate_mock_data(n=1000):
#         """Generates a detailed mock DataFrame for EDA using ALL brands."""
#         data = {
#             'Price': np.random.uniform(2.5, 60.0, n),
#             'Age': np.random.randint(1, 15, n),
#             'KM Driven': np.random.randint(10000, 180000, n),
#             'Brand': [random.choice(ALL_BRANDS) for _ in range(n)],
#             'Fuel Type': [random.choice(['Petrol', 'Diesel', 'CNG']) for _ in range(n)],
#             'Transmission': [random.choice(['Manual', 'Automatic']) for _ in range(n)],
#             'Ownership': [random.choice(['First Owner', 'Second Owner', 'Third Owner']) for _ in range(n)]
#         }
#         return pd.DataFrame(data)

#     df = generate_mock_data()

#     col1, col2 = st.columns([1, 2.5])
    
#     with col1:
#         st.subheader("Select a Visualization")
#         # Let user select top N brands to display for clarity
#         top_n = st.slider("Select Top N Brands to Display", 5, 20, 10)
#         top_brands = df['Brand'].value_counts().nlargest(top_n).index.tolist()
#         df_filtered_brands = df[df['Brand'].isin(top_brands)]

#         plot_choice = st.radio(
#             "Choose a chart:",
#             [
#                 "Market Share Distribution",
#                 "Price vs. Car Age",
#                 "Price by Fuel Type",
#                 "Price by Ownership"
#             ]
#         )

#     with col2:
#         if plot_choice == "Market Share Distribution":
#             st.subheader(f"Market Share of Top {top_n} Brands")
#             fig = px.pie(df_filtered_brands, names='Brand', title="Market Share of Top Brands by Listing Count")
#             st.plotly_chart(fig, use_container_width=True)

#         elif plot_choice == "Price vs. Car Age":
#             st.subheader(f"Price Depreciation Over Time for Top {top_n} Brands")
#             fig = px.scatter(df_filtered_brands, x='Age', y='Price', color='Brand',
#                              title="Price vs. Age of Car",
#                              labels={"Price": "Price (in Lakhs)", "Age": "Age (Years)"})
#             st.plotly_chart(fig, use_container_width=True)

#         elif plot_choice == "Price by Fuel Type":
#             st.subheader("Price Distribution: Petrol vs. Diesel vs. CNG")
#             fig = px.box(df, x='Fuel Type', y='Price', color='Fuel Type',
#                          title="Median Price by Fuel Type",
#                          labels={"Price": "Price (in Lakhs)"})
#             st.plotly_chart(fig, use_container_width=True)

#         elif plot_choice == "Price by Ownership":
#             st.subheader("Impact of Ownership on Price")
#             fig = px.box(df, x='Ownership', y='Price', color='Ownership',
#                          title="Price Distribution by Number of Owners",
#                          labels={"Price": "Price (in Lakhs)"},
#                          category_orders={"Ownership": ["First Owner", "Second Owner", "Third Owner"]})
#             st.plotly_chart(fig, use_container_width=True)

# # --- PAGE 4: PRICE PREDICTOR ---
# elif page == "Price Predictor":
#     st.title("🔮 Price Predictor")
#     st.markdown("---")
#     st.markdown("Fill in the car's details below to get an estimated market price.")

#     col1, col2 = st.columns([1, 1.2])

#     with col1:
#         st.subheader("Enter Car Details")
#         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#         model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand].keys()))
#         fuel_type = st.selectbox("Fuel Type", options=CAR_DATA[brand][model])
#         age = st.number_input("Car Age (years)", 1, 25, 5)
#         km_driven = st.number_input("KM Driven", 1000, 500000, 50000, step=1000)
#         transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
#         ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
#         predict_button = st.button("Predict Price", type="primary", use_container_width=True)

#     with col2:
#         st.subheader("Prediction Result")
#         if predict_button:
#             price = predict_car_price(brand, model, age, km_driven, fuel_type, transmission, ownership)
#             st.success(f"### Predicted Price: ₹ {price:.2f} Lakhs")
#             with st.expander("See Feature Impact on Price"):
#                 fig = create_shap_plot({'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}, price)
#                 st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Enter the car details and click 'Predict Price' to see the estimated value.")





import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

# --- 1. APP CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Prediction & Analysis",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MODEL & DATA LOADING ---
MODEL_FILE = "car_price_predictoR.joblib"

@st.cache_resource
def load_model(path: str):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.sidebar.error(f"⚠️ Error loading model: {e}")
            return None
    return None

model_pipeline = load_model(MODEL_FILE)

CAR_DATA = {
    "Maruti": {"Swift": ["Petrol", "CNG"], "Swift Dzire": ["Petrol", "Diesel"], "Alto 800": ["Petrol"], "Wagon R 1.0": ["Petrol", "CNG"], "Ciaz": ["Petrol", "Diesel"], "Ertiga": ["Petrol", "Diesel"], "Vitara Brezza": ["Petrol", "Diesel"], "Baleno": ["Petrol"], "S Cross": ["Diesel"], "Celerio": ["Petrol", "CNG"], "IGNIS": ["Petrol", "CNG"]},
    "Mahindra": {"XUV500": ["Diesel"], "Scorpio": ["Diesel"], "Thar": ["Petrol", "Diesel"], "XUV300": ["Petrol", "Diesel"], "Bolero": ["Diesel"], "Marazzo": ["Diesel"], "TUV300": ["Diesel"]},
    "Volkswagen": {"Polo": ["Petrol", "Diesel"], "Vento": ["Petrol", "Diesel"], "Ameo": ["Petrol", "Diesel"], "Jetta": ["Petrol", "Diesel"], "Passat": ["Petrol", "Diesel"], "Tiguan": ["Petrol", "Diesel"]},
    "Tata": {"Nexon": ["Petrol", "Diesel", "Electric"], "Harrier": ["Diesel"], "Tiago": ["Petrol", "CNG"], "Tigor": ["Petrol", "CNG"], "Safari": ["Diesel"], "Hexa": ["Diesel"], "PUNCH": ["Petrol"]},
    "Hyundai": {"i20": ["Petrol"], "Creta": ["Petrol", "Diesel"], "Verna": ["Petrol", "Diesel"], "VENUE": ["Petrol"], "Grand i10": ["Petrol", "CNG"], "Santro": ["Petrol", "CNG"], "Xcent": ["Petrol"], "Aura": ["Petrol"]},
    "Honda": {"City": ["Petrol"], "Amaze": ["Petrol"], "Jazz": ["Petrol"], "WR-V": ["Petrol"], "BR-V": ["Petrol"], "Civic": ["Petrol"]},
    "Ford": {"EcoSport": ["Petrol", "Diesel"], "Endeavour": ["Diesel"], "Figo": ["Petrol", "Diesel"], "Aspire": ["Petrol", "Diesel"], "Freestyle": ["Petrol"]},
    "BMW": {"3 Series": ["Petrol", "Diesel"], "5 Series": ["Petrol", "Diesel"], "X1": ["Petrol", "Diesel"], "X3": ["Petrol", "Diesel"], "X5": ["Petrol", "Diesel", "Hybrid"], "7 Series": ["Petrol", "Diesel"]},
    "Renault": {"Kwid": ["Petrol"], "Duster": ["Petrol", "Diesel"], "Triber": ["Petrol"], "Kiger": ["Petrol"], "Captur": ["Petrol", "Diesel"]},
    "MG": {"Hector": ["Petrol", "Diesel"], "Hector Plus": ["Petrol", "Diesel"], "Gloster": ["Diesel"], "ZS EV": ["Electric"]},
    "Datsun": {"redi-GO": ["Petrol"], "GO": ["Petrol"], "GO+": ["Petrol"]},
    "Nissan": {"Magnite": ["Petrol"], "Kicks": ["Petrol"], "Terrano": ["Diesel"], "Sunny": ["Petrol"], "Micra": ["Petrol"]},
    "Toyota": {"Innova Crysta": ["Diesel"], "Fortuner": ["Diesel"], "Yaris": ["Petrol"], "Glanza": ["Petrol"], "Urban Cruiser": ["Petrol"], "Corolla Altis": ["Petrol", "Hybrid"]},
    "Skoda": {"Rapid": ["Petrol"], "Octavia": ["Petrol"], "Superb": ["Petrol"], "Kushaq": ["Petrol"], "Slavia": ["Petrol"]},
    "Jeep": {"Compass": ["Petrol", "Diesel"], "Wrangler": ["Petrol"], "Meridian": ["Diesel"]},
    "KIA": {"Seltos": ["Petrol"], "Sonet": ["Petrol"], "Carnival": ["Diesel"], "Carens": ["Petrol"]},
    "Audi": {"A4": ["Petrol"], "A6": ["Petrol"], "Q3": ["Petrol"], "Q5": ["Petrol"], "Q7": ["Petrol"]},
    "Landrover": {"Range Rover Evoque": ["Petrol", "Diesel"], "Discovery Sport": ["Diesel"], "Range Rover Velar": ["Petrol"]},
    "Mercedes": {"C-Class": ["Petrol"], "E-Class": ["Petrol"], "GLC": ["Petrol"], "GLE": ["Petrol"], "S-Class": ["Petrol"]},
    "Chevrolet": {"Beat": ["Petrol"], "Cruze": ["Diesel"], "Spark": ["Petrol"], "Sail": ["Petrol"], "Enjoy": ["Diesel"]},
    "Fiat": {"Punto": ["Petrol"], "Linea": ["Petrol"]},
    "Jaguar": {"XF": ["Petrol", "Diesel"], "XE": ["Petrol"], "F-PACE": ["Petrol", "Diesel"]},
    "Mitsubishi": {"Pajero Sport": ["Diesel"]},
    "CITROEN": {"C5 Aircross": ["Petrol", "Diesel"], "C3": ["Petrol"]},
    "Mini": {"Cooper": ["Petrol"]},
    "ISUZU": {"D-MAX V-Cross": ["Diesel"]},
    "Volvo": {"XC60": ["Petrol", "Hybrid"], "XC90": ["Petrol", "Hybrid"], "S90": ["Petrol"]},
    "Porsche": {"Cayenne": ["Petrol"], "Macan": ["Petrol"]},
    "Force": {"Gurkha": ["Diesel"]}
}
ALL_BRANDS = sorted(list(CAR_DATA.keys()))

# --- 3. HELPER FUNCTIONS ---
@st.cache_data
def generate_mock_dataset(n=2000):
    """Generate a realistic-looking mock dataset for EDA and similarity comparisons."""
    rows = []
    for _ in range(n):
        brand = np.random.choice(ALL_BRANDS)
        model = np.random.choice(list(CAR_DATA[brand].keys()))
        fuel = np.random.choice(CAR_DATA[brand][model])
        age = np.random.randint(1, 15)
        km_driven = max(1000, int(np.random.normal(60000, 40000)))
        
        price = max(0.5, 15 - (age * 0.8) - (km_driven / 20000) + np.random.normal(0, 2))
        if brand in ["BMW", "Audi", "Mercedes", "Porsche", "Jaguar", "Landrover", "Volvo"]: price *= 2.5
        
        rows.append({ "brand": brand, "model": model, "fuel": fuel, "age": age,
                      "km_driven": km_driven, "price_lakhs": round(price, 2)})
    return pd.DataFrame(rows)

def safe_predict(brand, model, age, km_driven, fuel, transmission, ownership):
    """Return prediction either from pipeline or fallback logic."""
    if model_pipeline:
        try:
            X = pd.DataFrame([{"Car_Brand": brand, "Car_Model": model, "Car_Age": age, "KM Driven": km_driven, 
                               "Fuel Type": fuel, "Transmission Type": transmission, "Ownership": ownership}])
            return float(model_pipeline.predict(X)[0])
        except Exception as e:
            st.sidebar.error(f"Model prediction error: {e}") 
            # Fallthrough to mock
    
    base = 10.0 - (age * 0.7) - (km_driven / 50000)
    if fuel == "Diesel": base += 1.2
    if transmission == "Automatic": base += 1.5
    if ownership == "Second Owner": base -= 1.0
    elif "Third" in ownership or "Fourth" in ownership: base -= 2.0
    if brand in ["BMW", "Audi", "Mercedes", "Porsche", "Jaguar", "Landrover", "Volvo"]: base += 8.0
    elif brand in ["Toyota", "Skoda", "Jeep"]: base += 3.0
    return round(max(0.5, base + np.random.normal(0, 1.0)), 2)

def create_shap_plot(inputs, final_price):
    """Creates a mock feature impact plot for visualization."""
    base_value = 10.0
    contributions = [-(inputs['age'] * 0.7), -(inputs['km'] / 50000), 1.2 if inputs['fuel'] == 'Diesel' else -0.3, 1.5 if inputs['transmission'] == 'Automatic' else -0.5]
    features = [f"Age = {inputs['age']} yrs", f"KM Driven = {inputs['km']/1000:.1f}k km", f"Fuel = {inputs['fuel']}", f"Transmission = {inputs['transmission']}"]
    df = pd.DataFrame({'Feature': features, 'Contribution': contributions})
    df['Color'] = df['Contribution'].apply(lambda x: '#2ECC71' if x >= 0 else '#E74C3C')
    fig = px.bar(df, x='Contribution', y='Feature', orientation='h',
                 title=f"<b>Feature Impact on Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
                 text='Contribution', template="plotly_white")
    fig.update_traces(marker_color=df['Color'], texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Contribution to Price (in Lakhs)")
    return fig

# --- 4. PAGE FUNCTIONS (to prevent overlap) ---

# def page_profile():
#     st.title("About Me")
#     st.title("👋 Hi, I Am Alok")
#     col1, col2 = st.columns([2, 1])
#     with col1:
#         st.header("Aspiring Data Scientist • Deep Learning Project (ANN)")
#         st.write("Dedicated to applying deep learning and Data science techniques to extract insights, visualize trends, and deploy end-to-end analytical solutions using Python and modern ML workflows.")
#         st.markdown("**Skills:** Python, Pandas, NumPy, scikit-learn, Keras, OpenCV")
#         st.markdown("💼 **Contact:** [LinkedIn](https://www.linkedin.com/in/alok-tungal) • 💻 [GitHub](https://github.com/Alok-Tungal)")
#         st.markdown("---")
#         st.subheader("Highlights")
#         st.markdown("- Built an end-to-end deep learning pipeline (EDA → ANN Model → Deployment) for tabular data prediction tasks\n"
#                     "- Experienced in designing and tuning neural networks using TensorFlow and Keras\n"
#                     "- Created interactive dashboards and visual analytics using Plotly and Streamlit")
#     with col2:
#         st.image("https://placehold.co/100x100/0ea5a4/ffffff?text=Alok", use_container_width=True)

# def page_profile():
#     st.markdown("""
#         <div style="text-align: center;">
#             <h1> About Me </h1>
#         </div>
#     """, unsafe_allow_html=True)

#     # Add top margin to visually center the content
#     st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)

#     # Centered main title and subtitle
#     st.markdown("""
#         <div style="text-align: center;">
#             <h1>👋 Hi, I Am Alok </h1>
#                <h3>   Aspiring Data Scientist   </h3>
#         </div>
#     """, unsafe_allow_html=True)

#     st.markdown("---")

#     # Two-column layout for content and image
#     col1, col2 = st.columns([2, 1], vertical_alignment="center")

#     with col1:
#         st.write("""
#         Dedicated to applying deep learning and Data Science techniques to extract insights, visualize trends, 
#         and deploy end-to-end analytical solutions using Python and modern ML workflows.
#         """)

#         st.markdown("**Skills:** Python, Pandas, NumPy, scikit-learn, Keras, OpenCV")

#         st.markdown("""
#         💼 **Contact:** [LinkedIn](https://www.linkedin.com/in/alok-tungal) • 
#         💻 [GitHub](https://github.com/Alok-Tungal)
#         """)

#         st.markdown("---")
#         st.subheader("Highlights")
#         # st.markdown("""
#         # - Built an end-to-end deep learning pipeline (EDA → ANN Model → Deployment) for tabular data prediction tasks  
#         # - Experienced in designing and tuning neural networks using TensorFlow and Keras  
#         # - Created interactive dashboards and visual analytics using Plotly and Streamlit
#         # """)

#     with col2:
#         st.image("https://placehold.co/250x250/0ea5a4/ffffff?text=Alok", use_container_width=True)

# # Run function
# page_profile()

def page_profile():
    # --- HEADER ---
    # Using st.markdown for centered text is a good approach here.
    st.markdown("""
        <div style="text-align: center;">
            <h1>👋 Hi, I'm Alok</h1>
            <h3>Aspiring Data Scientist</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- MAIN CONTENT (Two Columns) ---
    col1, col2 = st.columns([2, 1], vertical_alignment="center")

    with col1:
        # --- BIO ---
        st.write("""
        Dedicated to applying deep learning and Data Science techniques to extract insights, visualize trends, 
        and deploy end-to-end analytical solutions using Python and modern ML workflows.
        """)

        # --- SKILLS ---
        # Using st.markdown for a bolded list is clear and effective.
        st.markdown("**Skills:** Python, Pandas, NumPy, scikit-learn, Keras, OpenCV, Streamlit")

        # --- LINKS ---
        # Icons add a nice professional touch.
        st.markdown("""
        💼 **Contact:** [LinkedIn](https://www.linkedin.com/in/alok-tungal) • 
        💻 [GitHub](https://github.com/Alok-Tungal)
        """)
        
        st.markdown("---")

        # --- HIGHLIGHTS (Uncommented) ---
        # This section is great for showing off your ANN and Streamlit projects.
        st.subheader("Project Highlights")
        st.markdown("""
        * Built end-to-end deep learning pipelines (EDA → ANN Model → Deployment).
        * Experienced in designing and tuning neural networks using TensorFlow and Keras.
        * Created interactive dashboards and data apps using Streamlit and Plotly.
        * Developed computer vision models using OpenCV and YOLO.
        """)

    with col2:
        # --- PROFILE IMAGE ---
        # Using a placeholder is fine, just swap the URL when you have a photo.
        st.image("https://placehold.co/250x250/0ea5a4/ffffff?text=Alok", use_container_width=True)

# --- Run the app page ---
# if __name__ == "__main__":
#     st.set_page_config(page_title="Alok's Portfolio", layout="wide")
#     page_profile()




def page_projects():
    st.title("🚀 Projects")
    st.markdown("A selection of projects I've built and deployed.")
    with st.expander("Car Price Prediction (This App)", expanded=True):
        st.write("End-to-end pipeline predicting used car prices — EDA, feature-engineering, XGBoost model, deploy.")
        st.markdown("- Dataset: ~9k listings\n- Model: XGBoost (best) \n- Deployment: Streamlit / Hugging Face Spaces")
    with st.expander("Churn Prediction"):
        st.write("Customer churn prediction using RandomForest + feature importance analysis (SHAP).")
    with st.expander("AQI Forecasting"):
        st.write("Time-series forecasting using RandomForest / XGBoost and visual explanations.")

def page_eda():
    st.title("📈 Exploratory Data Analysis")
    st.markdown("This section provides insights from a generated sample dataset of car listings.")
    
    df = generate_mock_dataset()

    st.subheader("Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Visualizations")
    colA, colB = st.columns(2)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    with colA:
        if numeric_cols:
            st.markdown("**Distribution Plot**")
            sel_num = st.selectbox("Select a numeric column", numeric_cols, key="hist_num")
            fig = px.histogram(df, x=sel_num, nbins=30, title=f"Distribution of {sel_num}", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    with colB:
        if cat_cols:
            st.markdown("**Category Count Plot**")
            sel_cat = st.selectbox("Select a categorical column", cat_cols, key="bar_cat")
            counts = df[sel_cat].value_counts().nlargest(15)
            fig2 = px.bar(counts, x=counts.values, y=counts.index, orientation="h", title=f"Top categories in {sel_cat}", labels={"x": "Count", "y": sel_cat}, template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Relationships between Features")
    colC, colD = st.columns(2)
    with colC:
        if len(numeric_cols) >= 2:
            st.markdown("**Correlation Heatmap**")
            corr = df[numeric_cols].corr()
            fig3 = px.imshow(corr, text_auto=True, title="Correlation Matrix", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
    with colD:
        if len(numeric_cols) and len(cat_cols):
            st.markdown("**Box Plot (Numeric vs. Categorical)**")
            y_col = st.selectbox("Numeric (Y-axis)", numeric_cols, key="box_y")
            x_col = st.selectbox("Category (X-axis)", cat_cols, key="box_x")
            fig4 = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}", template="plotly_white")
            st.plotly_chart(fig4, use_container_width=True)

def page_prediction():
    st.title("🔮 Car Price Prediction")
    st.markdown("Enter the car details below to get a price estimate.")
    
    # Show the model status message ONLY on this page
    if not model_pipeline:
        st.info("ℹ️ No trained model found — using fallback prediction logic.")

    left, right = st.columns([1, 1])
    with left:
        brand = st.selectbox("Brand", ALL_BRANDS)
        model = st.selectbox("Model", sorted(CAR_DATA[brand].keys()))
        fuel = st.selectbox("Fuel Type", CAR_DATA[brand][model])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
        ownership = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner", "Fourth+ Owner"])
    with right:
        age = st.number_input("Car Age (years)", 0, 30, 4)
        km_driven = st.number_input("KM Driven", 0, 500000, 45000, step=1000)

    if st.button("🚀 Predict Price", use_container_width=True):
        with st.spinner("Estimating price..."):
            predicted_price = safe_predict(brand, model, age, km_driven, fuel, transmission, ownership)

        st.markdown("---")
        st.header("Prediction Result")
        col_l, col_r = st.columns(2)
        with col_l:
            st.metric("Estimated Price", f"₹ {predicted_price:.2f} Lakhs")
            st.info(f"**Details:** {age} years old, {km_driven:,} km, {fuel}, {transmission}")
        with col_r:
            with st.expander("See Feature Impact", expanded=True):
                fig_imp = create_shap_plot({'age': age, 'km': km_driven, 'fuel': fuel, 'transmission': transmission}, predicted_price)
                st.plotly_chart(fig_imp, use_container_width=True)
        
        st.subheader("Comparable Listings (from mock data)")
        sample_df = generate_mock_dataset()
        similar = sample_df[(sample_df["brand"] == brand)].copy()
        similar['similarity'] = abs(similar['price_lakhs'] - predicted_price)
        similar = similar.sort_values('similarity').head(10)

        if not similar.empty:
            sim_fig = px.scatter(similar, x="km_driven", y="price_lakhs", color="age",
                                 size="price_lakhs", hover_data=["model", "fuel"],
                                 title=f"Market Comparison for '{brand}'", template="plotly_white")
            st.plotly_chart(sim_fig, use_container_width=True)

# --- 5. MAIN APP LOGIC ---
st.sidebar.image("https://placehold.co/300x80/111827/FFFFFF?text=Car+Price+AI", use_container_width=True)
st.sidebar.markdown("### Navigation")
page_options = {
    "Profile": page_profile,
    "Projects": page_projects,
    "EDA": page_eda,
    "Prediction": page_prediction
}
selected_page_name = st.sidebar.radio("", list(page_options.keys()))
st.sidebar.markdown("---")
# This is now handled inside the prediction page function
# if model_pipeline:
#     st.sidebar.success("✅ Trained model loaded")
# else:
#     st.sidebar.info("ℹ️ No trained model found — using fallback predictions")

# Run the selected page function
page_options[selected_page_name]()

st.markdown("---")
st.caption("Built by Alok Mahadev Tungal • Car Price Prediction & Analysis • Use responsibly")
#     st.info("This app uses a mock dataset for demonstration. A real-world version would be connected to a live database and a trained XGBoost regression model to provide real-time predictions and analytics.")


