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







import streamlit as st
import plotly.express as px
import pandas as pd
import random

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Car Price Prediction & Analysis",
    page_icon="🏎️",
    layout="wide"
)

# --- DATA ---
# This is the same car data from your original application
CAR_DATA = {
    "Maruti": ["Swift", "Swift Dzire", "Alto 800", "Wagon R 1.0", "Ciaz", "Ertiga", "Vitara Brezza", "Baleno", "S Cross", "Celerio", "IGNIS"],
    "Mahindra": ["XUV500", "Scorpio", "Thar", "XUV300", "Bolero", "Marazzo", "TUV300"],
    "Volkswagen": ["Polo", "Vento", "Ameo", "Jetta", "Passat", "Tiguan"],
    "Tata": ["Nexon", "Harrier", "Tiago", "Tigor", "Safari", "Hexa", "PUNCH"],
    "Hyundai": ["i20", "Creta", "Verna", "VENUE", "Grand i10", "Santro", "Xcent", "Aura"],
    "Honda": ["City", "Amaze", "Jazz", "WR-V", "BR-V", "Civic"],
    "Ford": ["EcoSport", "Endeavour", "Figo", "Aspire", "Freestyle"],
    "BMW": ["3 Series", "5 Series", "X1", "X3", "X5", "7 Series"],
    "Renault": ["Kwid", "Duster", "Triber", "Kiger", "Captur"],
    "MG": ["Hector", "Hector Plus", "Gloster", "ZS EV"],
    "Datsun": ["redi-GO", "GO", "GO+"],
    "Nissan": ["Magnite", "Kicks", "Terrano", "Sunny", "Micra"],
    "Toyota": ["Innova Crysta", "Fortuner", "Yaris", "Glanza", "Urban Cruiser", "Corolla Altis"],
    "Skoda": ["Rapid", "Octavia", "Superb", "Kushaq", "Slavia"],
    "Jeep": ["Compass", "Wrangler", "Meridian"],
    "KIA": ["Seltos", "Sonet", "Carnival", "Carens"],
    "Audi": ["A4", "A6", "Q3", "Q5", "Q7"],
    "Landrover": ["Range Rover Evoque", "Discovery Sport", "Range Rover Velar"],
    "Mercedes": ["C-Class", "E-Class", "GLC", "GLE", "S-Class"],
    "Chevrolet": ["Beat", "Cruze", "Spark", "Sail", "Enjoy"],
    "Fiat": ["Punto", "Linea"],
    "Ssangyong": ["Rexton"],
    "Jaguar": ["XF", "XE", "F-PACE"],
    "Mitsubishi": ["Pajero Sport"],
    "CITROEN": ["C5 Aircross", "C3"],
    "Mini": ["Cooper"],
    "ISUZU": ["D-MAX V-Cross"],
    "Volvo": ["XC60", "XC90", "S90"],
    "Porsche": ["Cayenne", "Macan"],
    "Force": ["Gurkha"]
}

# --- MOCK PREDICTION LOGIC ---
def predict_car_price(age, km_driven, fuel, transmission, ownership):
    base_price = 8.0
    base_price -= age * 0.5
    base_price -= km_driven / 50000
    if fuel == 'Diesel': base_price += 1
    if transmission == 'Automatic': base_price += 1.5
    if ownership == 'Second Owner': base_price -= 1
    if ownership == 'Third Owner': base_price -= 2
    final_price = max(1.5, base_price + (random.random() - 0.5))
    return final_price

# --- PLOTTING FUNCTIONS ---
def create_shap_plot(inputs, final_price):
    base_value = 8.0
    age_contribution = -(inputs['age'] * 0.5)
    km_contribution = -(inputs['km'] / 50000)
    fuel_contribution = 1.0 if inputs['fuel'] == 'Diesel' else -0.2
    transmission_contribution = 1.5 if inputs['transmission'] == 'Automatic' else -0.5
    
    contributions = [age_contribution, km_contribution, fuel_contribution, transmission_contribution]
    features = [f"Age = {inputs['age']} yrs", f"KM Driven = {(inputs['km']/1000):.1f}k km", f"Fuel = {inputs['fuel']}", f"Transmission = {inputs['transmission']}"]
    
    plot_data = pd.DataFrame({
        'contribution': contributions,
        'feature': features
    })
    
    plot_data['color'] = plot_data['contribution'].apply(lambda c: '#2ECC71' if c >= 0 else '#E74C3C')
    
    fig = px.bar(
        plot_data,
        x='contribution',
        y='feature',
        orientation='h',
        title=f"<b>How Features Impact the Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
        labels={'contribution': 'Contribution to Price (in Lakhs)', 'feature': ''}
    )
    
    fig.update_traces(marker_color=plot_data['color'])
    
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#333",
        margin=dict(l=120, r=20, t=60, b=50)
    )
    return fig

# --- STREAMLIT UI ---
st.title("🏎️ Car Price Prediction & Analysis")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Me", "The Project", "Data Insights", "Price Predictor"])

# --- PAGE 1: About Me ---
if page == "About Me":
    st.header("About Me")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            Hi, I’m **Alok Mahadev Tungal** — a passionate **Data Scientist** and **Machine Learning Enthusiast**.

            This app demonstrates my journey of building a complete project:
            from *data analysis* ➝ *exploratory visualization* ➝ *model building* ➝ *deployment*.

            My goal is to solve real-world problems with **AI & ML** while creating impactful and interactive applications.
            """
        )
        st.markdown(
            """
            - **LinkedIn:** [https://www.linkedin.com/](https://www.linkedin.com/)
            - **GitHub:** [https://github.com/](https://github.com/)
            - **Hugging Face:** [https://huggingface.co/](https://huggingface.co/)
            """
        )
    with col2:
        st.image("https://placehold.co/200x200/4F46E5/FFFFFF?text=AMT", caption="Alok Mahadev Tungal")

# --- PAGE 2: The Project ---
elif page == "The Project":
    st.header("The Project")
    st.markdown("Here are the key metrics and technologies used in this project.")
    
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Dataset Rows", "9,176")
    kpi_cols[1].metric("Dataset Columns", "9")
    kpi_cols[2].metric("Brands Covered", "30")

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Regression Model", "XGBoost")
    kpi_cols[1].metric("Prediction Accuracy (R²)", "~96%")
    kpi_cols[2].metric("Visualization Library", "Plotly Express")

# --- PAGE 3: Data Insights ---
elif page == "Data Insights":
    st.header("Data Insights & Visualizations")
    st.info("Here are some interactive charts exploring the car dataset. These are mock visualizations based on the data structure.")
    
    # Mock data generation for plots
    @st.cache_data
    def generate_mock_data(num_points=500):
        prices = [random.uniform(2.0, 48.0) for _ in range(num_points)]
        ages = [random.randint(1, 12) for _ in range(num_points)]
        brands = [random.choice(list(CAR_DATA.keys())) for _ in range(num_points)]
        return pd.DataFrame({'price': prices, 'age': ages, 'brand': brands})

    df = generate_mock_data()
    
    plot_choice = st.selectbox(
        "Choose a visualization:",
        (
            'Distribution of Car Prices',
            'Car Listings by Brand',
            'Price vs. Car Age',
            'Price Distribution by Brand',
            'Feature Correlation Heatmap'
        )
    )

    fig = None

    if plot_choice == 'Distribution of Car Prices':
        fig = px.histogram(df, x='price', title='<b>Distribution of Car Prices</b>', 
                           labels={'price': 'Price (in Lakhs)', 'count': 'Number of Cars'},
                           color_discrete_sequence=['#f97316'])
    
    elif plot_choice == 'Car Listings by Brand':
        brand_counts = df['brand'].value_counts().nlargest(15)
        fig = px.bar(x=brand_counts.values, y=brand_counts.index, orientation='h', 
                     title='<b>Car Listings by Brand (Top 15)</b>',
                     labels={'x': 'Number of Listings', 'y': 'Brand'},
                     color_discrete_sequence=['#8b5cf6'])
        fig.update_layout(yaxis=dict(autorange="reversed"))

    elif plot_choice == 'Price vs. Car Age':
        fig = px.scatter(df, x='age', y='price', title='<b>Price vs. Car Age</b>',
                         labels={'age': 'Age (Years)', 'price': 'Price (Lakhs)'},
                         color_discrete_sequence=['#14b8a6'])
        fig.update_traces(marker=dict(opacity=0.6))
    
    elif plot_choice == 'Price Distribution by Brand':
        top_brands = df['brand'].value_counts().nlargest(8).index
        df_top_brands = df[df['brand'].isin(top_brands)]
        fig = px.box(df_top_brands, x='brand', y='price', title='<b>Price Distribution by Brand (Top 8)</b>',
                     labels={'brand': 'Brand', 'price': 'Price (Lakhs)'},
                     color_discrete_sequence=['#facc15'])
        
    elif plot_choice == 'Feature Correlation Heatmap':
        corr_matrix = pd.DataFrame({
            'Price': [1.00, -0.65, -0.45],
            'Age': [-0.65, 1.00, 0.30],
            'KM': [-0.45, 0.30, 1.00]
        }, index=['Price', 'Age', 'KM'])
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title='<b>Feature Correlation Heatmap</b>',
                        color_continuous_scale='Viridis', zmin=-1, zmax=1)

    if fig:
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#333")
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 4: Price Predictor ---
elif page == "Price Predictor":
    st.header("Price Predictor")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Enter Car Details")
        
        brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
        
        models = CAR_DATA[brand]
        model = st.selectbox("Car Model", options=sorted(models))
        
        age = st.number_input("Car Age (in years)", min_value=0, max_value=25, value=5)
        km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
        
        fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
        transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
        ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
        
        predict_button = st.button("Predict Price", use_container_width=True, type="primary")

    with col2:
        st.subheader("Prediction Result")
        
        if predict_button:
            with st.spinner('Calculating...'):
                predicted_price = predict_car_price(age, km_driven, fuel_type, transmission, ownership)
                
                st.success(f"## Predicted Price: ₹ {predicted_price:.2f} Lakhs")
                
                inputs = {'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}
                shap_fig = create_shap_plot(inputs, predicted_price)
                st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.info("Click 'Predict Price' after entering the details to see the result.")


