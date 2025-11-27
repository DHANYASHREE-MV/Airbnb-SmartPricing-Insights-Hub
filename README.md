# Airbnb-SmartPricing-Insights-Hub

### EDA • Interactive Market Dashboard • Machine Learning Price Predictor (FastAPI)

This project is a full-stack data science application built to analyze the **New York City Airbnb market**, explore price trends, and **predict nightly stay prices** using a machine learning model served via **FastAPI**.  
The user interface is built using **Streamlit** with a modern dark-glass UI theme.

---

## 🌟 Project Highlights

### 🔍 **1. Exploratory Data Analysis (EDA)**
Understand the NYC Airbnb dataset through:
- Dataset preview & shape  
- Summary statistics  
- Price distribution  
- Room type distribution  
- Neighbourhood group comparison  
- Visual insights using Plotly  

---

### 📊 **2. Market Dashboard – Interactive Explorer**
A full analytics dashboard where users can:

- Filter by **Neighbourhood Group**, **Room Type**, **Price Range**, and **Minimum Nights**
- View:
  - KPI Cards (Total Listings, Avg Price, Median Price, Avg Reviews)
  - Filtered price distribution  
  - Room type share  
  - Top neighbourhoods by price  
  - Geo-map (Mapbox) of listings with pricing and reviews  

This creates a real **product-like analytics interface**.

---

### 💰 **3. Machine Learning Price Predictor (FastAPI Backend)**
A dedicated module to estimate the **nightly price** of an Airbnb listing.

- Inputs collected via Streamlit UI (categorical + numeric)
- Sent to FastAPI endpoint via POST request
- FastAPI loads ML model (`price_model.pkl`)
- Returns predicted price instantly

This demonstrates a **true ML deployment workflow**.

---

## 🧠 Machine Learning Pipeline

| Component | Description |
|----------|-------------|
| Model Type -- RandomForestRegressor |
| Preprocessing -- OneHotEncoding + StandardScaler + Imputation |
| Framework -- scikit-learn |
| Exported Model --`price_model.pkl` |
| Served Through -- FastAPI endpoint `/predict` |
| Metrics Used  -- MAE • RMSE • R² |

Pipeline built using `ColumnTransformer` + `Pipeline`.

---

## 🗂️ Project Structure
project/
│
├── backend/
│ └── main.py # FastAPI backend (prediction API)
│
├── notebook/
│ ├── phase1_eda.ipynb
│ ├── phase2_feature_engineering.ipynb
│ ├── phase3.py # Model training script
│ └── price_model.pkl # Saved model used in FastAPI
│
├── data/
│ └── AB_NYC_2019.csv # Dataset
│
├── frontend
    └── frontend.py # Streamlit application
├── requirements.txt
└── README.md

---

## ⚙️ Setup & Run Locally

### 1️⃣ Install Dependencies**
bash/terminal/powershell
pip install -r requirements.txt

### 2️⃣ Start FastAPI Backend
cd backend
python -muvicorn main:app --reload

 ->FastAPI will run at:

API Docs → http://127.0.0.1:8000/docs
Predict Endpoint → http://127.0.0.1:8000/predict

** if FastAPI fails,then return to your main project directory,and establish python environment by activating scripts : [.\.venv\Scripts\activate] 
    and then excecute commands to get your backend working.**
    
3️⃣ Start Streamlit Frontend
python -m streamlit run frontend.py
 ** For verification just check the price predicted in the streamlit frontend which was given by FastAPI,is same as backend server.** [find attached screenshots]
  
🧾 Dataset Details
Dataset used: AB_NYC_2019.csv
Source includes:

*48,895 listings
*16 columns
*Attributes like neighbourhood, price, room type, availability, reviews count, coordinates, etc.

✨ Technologies Used:

| Category          | Tools                  |
| ----------------- | ---------------------- |
| **Frontend**      | Streamlit, Plotly      |
| **Backend**       | FastAPI                |
| **ML Pipeline**   | scikit-learn           |
| **Model Serving** | Uvicorn                |
| **Data**          | pandas, numpy          |
| **Visualization** | Plotly Express, Mapbox |


🎯 Key Learning Outcomes

End-to-end ML workflow (EDA → Model → Deployment)
REST API integration with ML models
Full-stack dashboard development
Data visualization & storytelling
Real-world deployment architecture

⭐ Show Your Support!

If this project helped or impressed you, please ⭐ star the repo — it means a lot!






