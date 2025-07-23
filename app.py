import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


model = joblib.load('women_safety_model.pkl')
le = joblib.load('label_encoder.pkl')

df = pd.read_csv('districtwise-crime-against-women-2017-onwards.csv')
df['district_name_clean'] = df['district_name'].str.strip().str.lower()
crime_columns = df.columns[8:-1]
district_names = sorted(df['district_name'].unique())


crime_label_map = {
    'rape_girls_below_18': 'Rape of Girls Below 18',
    'rape_women_above_18': 'Rape of Women Above 18',
    'attempt_to_rape': 'Attempt to Rape',
    'assault_on_womenintent_to_outrage_modesty': 'Assault with Intent to Outrage Modesty',
    'insult_to_modesty_of_women': 'Insult to the Modesty of Women',
    'kidnp_and_abductn_of_women_and_girls': 'Kidnapping and Abduction of Women and Girls',
    'kidnp_and_abductn_of_women_for_illicit_interc': 'Kidnapping of Women for Illicit Intercourse',
    'kidnp_and_abductn_of_women_above_18_for_marrg': 'Kidnapping of Women (18+) for Marriage',
    'kidnp_and_abductn_of_girls_below_18_for_marrg': 'Kidnapping of Girls (<18) for Marriage',
    'kidnp_and_abductn_of_others': 'Other Kidnapping and Abduction',
    'cyber_crimes_against_women': 'Cyber Crimes Against Women',
    'importation_of_girls': 'Importation of Girls',
    'cruelty_by_husband_or_his_relatives': 'Cruelty by Husband or His Relatives',
    'dowry_deaths': 'Dowry Deaths',
    'dowry_prohibition': 'Dowry Prohibition Violations',
    'immoral_traffic': 'Immoral Trafficking',
    'procuration_of_minor_girls': 'Procuration of Minor Girls',
    'human_trafficking': 'Human Trafficking',
    'abetment_of_suicide': 'Abetment of Suicide',
    'causing_death_by_neglect': 'Causing Death by Negligence',
    'acid_attack': 'Acid Attack',
    'attempt_to_commit_acid_attack': 'Attempt to Commit Acid Attack',
    'unclassifiable_crimes': 'Unclassified Crimes',
    'total_crimes_against_women': 'Total Crimes Against Women',
    'assault_on_womenabove_18': 'Assault on Women (Above 18)',
    'kidnapping_and_abduction': 'Kidnapping and Abduction',
    'kidnapping_and_abduction_of_women_others': 'Kidnapping and Abduction of Women (Other)',
    'child_rape': 'Child Rape'
}


def beautify(crime):
    return crime_label_map.get(crime, crime.replace('_', ' ').title())

def predict_risk_level(district_name):
    name_clean = district_name.strip().lower()
    district_rows = df[df['district_name_clean'] == name_clean]

    if district_rows.empty:
        return None, []

    district_mean = district_rows[crime_columns].mean().values.reshape(1, -1)
    prediction_encoded = model.predict(district_mean)[0]
    prediction_label = le.inverse_transform([prediction_encoded])[0]

    crime_sums = district_rows[crime_columns].sum().sort_values(ascending=False)
    top_crimes = crime_sums.head(5)

    return prediction_label, top_crimes

# Streamlit UI
st.set_page_config(page_title="Women Safety Predictor", layout="centered")

st.title("Women Safety Risk Predictor")
st.write("Check safety risk level and top reported crimes for any Indian district based on crime data from 2017 to 2022.")

selected_district = st.selectbox("📍 Select District", district_names)

if st.button("🚨 Predict Risk Level"):
    with st.spinner("Analyzing district data..."):
        risk_level, top_crimes = predict_risk_level(selected_district)

    if risk_level is None:
        st.error("District not found. Please try again.")
    else:
        if risk_level == "High":
            st.error(f"🔴 Predicted Risk Level: **{risk_level}**")
        elif risk_level == "Medium":
            st.warning(f"🟡 Predicted Risk Level: **{risk_level}**")
        else:
            st.success(f"🟢 Predicted Risk Level: **{risk_level}**")

        st.subheader("🔎 Top Reported Crimes")
        for crime, count in top_crimes.items():
            st.markdown(f"- {beautify(crime)} ({int(count)} cases)")

        fig = px.bar(
            top_crimes.sort_values(),
            x=top_crimes.values,
            y=[beautify(c) for c in top_crimes.index],
            orientation='h',
            color=top_crimes.values,
            color_continuous_scale='Reds',
            labels={'x': 'Number of Cases', 'y': 'Crime Type'},
            height=400
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
