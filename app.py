import streamlit as st
import plotly.graph_objects as go
import random
import pandas as pd

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
# This is the same logic from your original app, now in Python
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
    
    colors = ['#E74C3C' if c < 0 else '#2ECC71' for c in contributions]

    fig = go.Figure(go.Bar(
        x=contributions,
        y=features,
        orientation='h',
        marker_color=colors
    ))
    fig.update_layout(
        title=f"<b>How Features Impact the Price</b><br>Base: ₹{base_value:.2f}L | Final: ₹{final_price:.2f}L",
        xaxis_title="Contribution to Price (in Lakhs)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
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
        st.image("https://placehold.co/200x200/ff3366/ffffff?text=AMT", caption="Alok Mahadev Tungal")

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
    kpi_cols[2].metric("Visualization Library", "Plotly")

# --- PAGE 3: Data Insights ---
elif page == "Data Insights":
    st.header("Data Insights & Visualizations")
    st.info("Here are some interactive charts exploring the car dataset. These are mock visualizations based on the data structure.")
    
    # Mock data generation for plots
    num_points = 500
    prices = [random.uniform(2.0, 48.0) for _ in range(num_points)]
    ages = [random.randint(1, 12) for _ in range(num_points)]
    brands = [random.choice(list(CAR_DATA.keys())) for _ in range(num_points)]
    
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

    fig = go.Figure()

    if plot_choice == 'Distribution of Car Prices':
        fig.add_trace(go.Histogram(x=prices, marker_color='#f97316'))
        fig.update_layout(title='<b>Distribution of Car Prices</b>', xaxis_title='Price (in Lakhs)', yaxis_title='Count')
    
    elif plot_choice == 'Car Listings by Brand':
        brand_counts = pd.Series(brands).value_counts().nlargest(15)
        fig.add_trace(go.Bar(y=brand_counts.index, x=brand_counts.values, orientation='h', marker_color='#8b5cf6'))
        fig.update_layout(title='<b>Car Listings by Brand (Top 15)</b>', yaxis=dict(autorange="reversed"))

    elif plot_choice == 'Price vs. Car Age':
        fig.add_trace(go.Scatter(x=ages, y=prices, mode='markers', marker=dict(color='#14b8a6', opacity=0.6)))
        fig.update_layout(title='<b>Price vs. Car Age</b>', xaxis_title='Age (Years)', yaxis_title='Price (Lakhs)')
    
    elif plot_choice == 'Price Distribution by Brand':
        df = pd.DataFrame({'brand': brands, 'price': prices})
        top_brands = df['brand'].value_counts().nlargest(8).index
        for brand in top_brands:
            fig.add_trace(go.Box(y=df[df['brand'] == brand]['price'], name=brand, marker_color='#facc15'))
        fig.update_layout(title='<b>Price Distribution by Brand</b>', showlegend=False)

    elif plot_choice == 'Feature Correlation Heatmap':
        corr_matrix = [[1.00, -0.65, -0.45], [-0.65, 1.00, 0.30], [-0.45, 0.30, 1.00]]
        labels = ['Price', 'Age', 'KM']
        fig.add_trace(go.Heatmap(z=corr_matrix, x=labels, y=labels, colorscale='Viridis', zmin=-1, zmax=1))
        fig.update_layout(title='<b>Feature Correlation Heatmap</b>')
    
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig, use_container_width=True)


# --- PAGE 4: Price Predictor ---
elif page == "Price Predictor":
    st.header("Price Predictor")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Enter Car Details")
        
        brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
        
        # Dependent dropdown for model
        models = CAR_DATA[brand]
        model = st.selectbox("Car Model", options=sorted(models))
        
        age = st.number_input("Car Age (in years)", min_value=0, max_value=25, value=5)
        km_driven = st.number_input("KM Driven", min_value=0, max_value=500000, value=50000, step=1000)
        
        fuel_type = st.selectbox("Fuel Type", options=['Petrol', 'Diesel', 'CNG', 'Electric', 'LPG'])
        transmission = st.selectbox("Transmission Type", options=['Manual', 'Automatic'])
        ownership = st.selectbox("Ownership", options=['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'])
        
        predict_button = st.button("Predict Price", use_container_width=True)

    with col2:
        st.subheader("Prediction Result")
        
        if predict_button:
            with st.spinner('Calculating...'):
                predicted_price = predict_car_price(age, km_driven, fuel_type, transmission, ownership)
                
                st.success(f"## Predicted Price: ₹ {predicted_price:.2f} Lakhs")
                
                # SHAP-like plot
                inputs = {'age': age, 'km': km_driven, 'fuel': fuel_type, 'transmission': transmission}
                shap_fig = create_shap_plot(inputs, predicted_price)

                st.plotly_chart(shap_fig, use_container_width=True)
        else:
            st.info("Click 'Predict Price' after entering the details to see the result.")







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
# elif page == "Price Predictor":
#     st.title("🔮 Price Predictor")
#     col1, col2 = st.columns(2)
#     with col1:
#         brand = st.selectbox("Car Brand", options=sorted(CAR_DATA.keys()))
#         model = st.selectbox("Car Model", options=sorted(CAR_DATA[brand]))
#         age = st.number_input("Car Age (years)", 0, 25, 5)
#         km = st.number_input("KM Driven", 0, 500000, 50000, 1000)
#         fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG", "Electric", "LPG"])
#         trans = st.selectbox("Transmission", ["Manual", "Automatic"])
#         owner = st.selectbox("Ownership", ["First Owner", "Second Owner", "Third Owner"])
#         predict = st.button("Predict Price", type="primary")
#     with col2:
#         if predict:
#             price = predict_car_price(age, km, fuel, trans, owner)
#             st.success(f"### Predicted Price: ₹ {price:.2f} Lakhs")
#             fig = create_shap_plot({'age': age, 'km': km, 'fuel': fuel, 'transmission': trans}, price)
#             st.plotly_chart(fig, use_container_width=True)
#         else:
#             st.info("Enter details and click predict.")



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
