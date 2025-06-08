# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("malaria_africa_cleaned.csv")

df = load_data()

st.title("🛡️ MalariaShield Dashboard")
st.subheader("Predicting and Visualizing Malaria Cases in Africa (SDG 3)")

# Sidebar Filters
country_list = sorted(df["Country"].unique())
selected_country = st.sidebar.selectbox("Select Country", country_list)
year_range = st.sidebar.slider("Select Year Range", 
                               int(df["Year"].min()), 
                               int(df["Year"].max()), 
                               (2010, 2017))

# Filter Data
filtered_df = df[(df["Country"] == selected_country) & 
                 (df["Year"].between(year_range[0], year_range[1]))]

# Show Summary Stats
st.write(f"### 📊 Malaria Data for {selected_country} ({year_range[0]} - {year_range[1]})")
st.dataframe(filtered_df)

# Line Plot: Estimated vs Reported Cases
st.write("#### 📈 Estimated vs Reported Malaria Cases Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_df["Year"], filtered_df["No. of cases_median"], label="Estimated Median Cases", marker="o")
ax.plot(filtered_df["Year"], filtered_df["No. of cases_reported"], label="Reported Cases", marker="x")
ax.set_xlabel("Year")
ax.set_ylabel("Cases")
ax.set_title(f"Malaria Cases Trend in {selected_country}")
ax.legend()
st.pyplot(fig)

# Bar Chart: Deaths
st.write("#### 💀 Estimated vs Reported Deaths")
fig2, ax2 = plt.subplots()
width = 0.35
x = filtered_df["Year"]
ax2.bar(x - width/2, filtered_df["No. of deaths_median"], width, label="Estimated Median Deaths")
ax2.bar(x + width/2, filtered_df["No. of deaths_reported"], width, label="Reported Deaths")
ax2.set_xlabel("Year")
ax2.set_ylabel("Deaths")
ax2.set_title(f"Malaria Deaths in {selected_country}")
ax2.legend()
st.pyplot(fig2)

# Future Work Section
st.markdown("#### 🔮 Future Predictions")
st.info("Model predictions will be added in a future version using machine learning. This version focuses on historical analysis.")

st.markdown("---")
st.markdown("✅ *Dashboard built for PLP AI4SE Week 2 — MalariaShield by Group 39*")
