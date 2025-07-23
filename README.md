# 🛡️ Women Safety Risk Predictor

A machine learning-based app to **predict the safety risk level** for women in any Indian district using real government crime data (2017–2022).

---

## 🔍 What This App Does

- Accepts any Indian district as input.
- Predicts **Risk Level**: High, Medium, or Low.
- Displays the **Top 5 crimes reported** in that district.
- Visualizes the data with an **interactive bar chart** (using Plotly).
- Built with **Streamlit** for a fast and clean UI.

---

## 🧠 How It Works

### 1. Model Training (Offline)

- The model used is a `RandomForestClassifier`.
- Trained on district-wise crime data from 2017 to 2022.
- Data source: NCRB official datasets, preprocessed to group relevant crime categories.

### 2. User Interaction

- You choose a district from the dropdown.
- The app calculates the average crime data for that district.
- The trained ML model predicts the safety **risk level**.
- Top 5 crimes are shown with their case counts.

---

## 🖥️ Tech Stack

- **Frontend**: Streamlit
- **Backend ML**: Scikit-learn (Random Forest)
- **Data Handling**: Pandas
- **Visualization**: Plotly
- **Packaging**: Joblib for model + encoder

---

streamlit run app.py
