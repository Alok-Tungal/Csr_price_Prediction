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



import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Interactive Car Price Analyzer",
    page_icon="🚗",
    layout="wide"
)

# --- DATA GENERATION ---
@st.cache_data
def generate_mock_data(num_cars=500):
    """Generates a realistic mock dataset for the car market."""
    brands = ["Maruti", "Hyundai", "Honda", "Tata", "Mahindra", "Ford", "Toyota", "BMW"]
    data = {
        'Brand': np.random.choice(brands, num_cars),
        'Age': np.random.randint(1, 12, num_cars),
        'KM_Driven': np.random.randint(5000, 150000, num_cars),
        'Fuel_Type': np.random.choice(['Petrol', 'Diesel'], num_cars),
        'Transmission': np.random.choice(['Manual', 'Automatic'], num_cars),
        'Ownership': np.random.choice(['First Owner', 'Second Owner', 'Third Owner'], num_cars)
    }
    df = pd.DataFrame(data)

    # Create a mock price with logical dependencies
    base_price = 20.0 - (df['Age'] * 1.5) - (df['KM_Driven'] / 20000)
    base_price += df['Fuel_Type'].apply(lambda x: 1.5 if x == 'Diesel' else 0)
    base_price += df['Transmission'].apply(lambda x: 2.0 if x == 'Automatic' else 0)
    df['Price'] = np.clip(base_price + np.random.normal(0, 1.5, num_cars), 1.5, 50.0) # Add noise
    
    return df

df = generate_mock_data()

# --- SIDEBAR FOR CONTROLS ---
st.sidebar.title("🔎 Market Filters")
st.sidebar.markdown("Use the filters below to explore the car market data.")

price_range = st.sidebar.slider(
    'Filter by Price Range (Lakhs)',
    min_value=float(df['Price'].min()),
    max_value=float(df['Price'].max()),
    value=(float(df['Price'].min()), float(df['Price'].max()))
)

brand_selection = st.sidebar.multiselect(
    'Filter by Brand',
    options=sorted(df['Brand'].unique()),
    default=sorted(df['Brand'].unique())[:3] # Default to first 3 brands
)

fuel_selection = st.sidebar.multiselect(
    'Filter by Fuel Type',
    options=df['Fuel_Type'].unique(),
    default=df['Fuel_Type'].unique()
)

# --- FILTER DATA BASED ON CONTROLS ---
filtered_df = df[
    (df['Price'].between(price_range[0], price_range[1])) &
    (df['Brand'].isin(brand_selection)) &
    (df['Fuel_Type'].isin(fuel_selection))
]

# --- MAIN APP LAYOUT ---
st.title("🚗 Interactive Car Price Analyzer")
st.markdown("An interactive dashboard to explore market trends and predict used car prices.")
st.markdown("---")

# --- TOP KPI ROW ---
avg_price_total = df['Price'].mean()
avg_price_filtered = filtered_df['Price'].mean() if not filtered_df.empty else 0

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Cars Listed", value=len(df))
with col2:
    st.metric(label="Cars Matching Filters", value=len(filtered_df))
with col3:
    st.metric(
        label="Average Price (Filtered)",
        value=f"₹{avg_price_filtered:.2f} L",
        delta=f"₹{avg_price_filtered - avg_price_total:.2f} L vs. market avg",
    )

st.markdown("---")

# --- TABS FOR DIFFERENT VIEWS ---
tab1, tab2 = st.tabs(["📊 Market Analysis", "🔮 Price Predictor"])

with tab1:
    st.header("Explore Market Trends")
    if not filtered_df.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Price Depreciation by Age")
            fig_scatter = px.scatter(
                filtered_df, x='Age', y='Price',
                color='Brand',
                title="Price vs. Age for Selected Brands",
                labels={"Price": "Price (Lakhs)", "Age": "Age (Years)"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        with col2:
            st.subheader("Price by Transmission & Fuel Type")
            fig_box = px.box(
                filtered_df, x='Transmission', y='Price',
                color='Fuel_Type',
                title="Price Distribution by Drivetrain",
                labels={"Price": "Price (Lakhs)"}
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.warning("No cars match the current filter settings. Please adjust the filters in the sidebar.")

with tab2:
    st.header("Get a Price Estimate")
    st.info("Enter the details of a car to get a quick price prediction.")

    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("Car Details")
        pred_brand = st.selectbox("Brand", options=sorted(df['Brand'].unique()))
        pred_age = st.number_input("Age (Years)", 1, 20, 5)
        pred_km = st.number_input("Kilometers Driven", 1000, 300000, 50000, step=1000)
        pred_fuel = st.selectbox("Fuel Type", options=df['Fuel_Type'].unique())
        pred_trans = st.selectbox("Transmission", options=df['Transmission'].unique())
        predict_button = st.button("Predict Price", type="primary")

    with col2:
        st.subheader("Prediction Result")
        if predict_button:
            # Simple mock prediction logic (replace with your actual model's logic if available)
            predicted_price = 20.0 - (pred_age * 1.5) - (pred_km / 20000)
            predicted_price += 1.5 if pred_fuel == 'Diesel' else 0
            predicted_price += 2.0 if pred_trans == 'Automatic' else 0
            final_price = max(1.5, predicted_price + np.random.normal(0, 0.5))

            st.success(f"### Estimated Price: ₹ {final_price:.2f} Lakhs")
            st.markdown("This is an estimate based on market data. Actual price may vary.")
        else:
            st.write("Click the 'Predict Price' button to see the result.")
