
````markdown
#  Women Safety Risk Predictor

A simple machine learning app that predicts the **risk level** for women in Indian districts using crime data from 2017–2022. Built with Python, scikit-learn, and Streamlit.

---

## 📊 What It Does

- Accepts any Indian district as input (via dropdown).
- Predicts safety **risk level**: High, Medium, or Low.
- Shows top 5 reported crimes in that district.
- Displays a visual bar chart using Plotly.

---

## 🧠 How It Works

1. **Model Training (Offline)**  
   A supervised learning model (RandomForestClassifier) was trained using district-wise crime data from 2017 to 2022.

2. **Input from User**  
   You select a district — the app fetches relevant data, calculates the mean stats, and uses the trained model to predict the risk.

3. **Results**  
   - Risk level (color-coded).
   - Top crimes with counts.
   - Bar graph of top 5 crimes.

---


