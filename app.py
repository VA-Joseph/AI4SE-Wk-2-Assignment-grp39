 # app.py 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.linear_model import LinearRegression
import numpy as np

uploaded_file = st.sidebar.file_uploader(" Upload your own CSV file", type=["csv"])

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

# Pie Chart: Shpwcasing the Latest Year Reported vs Estimated
st.write("#### 🧬 Reported vs Estimated Cases Distribution (Latest Year)")
latest_year = filtered_df["Year"].max()
latest_data = filtered_df[filtered_df["Year"] == latest_year]
if not latest_data.empty:
    values = [latest_data["No. of cases_reported"].values[0], latest_data["No. of cases_median"].values[0]]
    labels = ['Reported', 'Estimated']
    fig4, ax4 = plt.subplots()
    ax4.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99'])
    ax4.axis('equal')
    st.pyplot(fig4)

#Stacked Bar Chart: Cases + Deaths
st.write("#### 🧱 Combined Cases and Deaths (Stacked Bar)")
fig5, ax5 = plt.subplots()
ax5.bar(filtered_df["Year"], filtered_df["No. of cases_reported"], label='Cases Reported', color='skyblue')
ax5.bar(filtered_df["Year"], filtered_df["No. of deaths_reported"], bottom=filtered_df["No. of cases_reported"], 
        label='Deaths Reported', color='salmon')
ax5.set_xlabel("Year")
ax5.set_ylabel("Count")
ax5.set_title("Reported Cases and Deaths (Stacked)")
ax5.legend()
st.pyplot(fig5)

st.write("#### 🔥 Correlation Heatmap (Numerical Features)")
fig6, ax6 = plt.subplots()
sns.heatmap(filtered_df.select_dtypes(include='number').corr(), annot=True, cmap="YlGnBu", ax=ax6)
st.pyplot(fig6)

# 🔮 ML Predictions
st.markdown("#### 🔮 Predict Future Malaria Cases")
if len(filtered_df) >= 5:
    model = LinearRegression()
    X = filtered_df[["Year"]]
    y = filtered_df["No. of cases_reported"]

    model.fit(X, y)

    future_years = np.array(range(year_range[1] + 1, year_range[1] + 6)).reshape(-1, 1)
    future_preds = model.predict(future_years)

    pred_df = pd.DataFrame({
        "Year": future_years.flatten(),
        "Predicted Reported Cases": future_preds.astype(int)
    })

    st.write("##### 📅 Predicted Reported Malaria Cases (Next 5 Years)")
    st.dataframe(pred_df)

    # Combine for plot
    combined_years = pd.concat([X, pd.DataFrame(future_years, columns=["Year"])])
    combined_cases = pd.concat([y, pd.Series(future_preds, name="No. of cases_reported")])

    fig7, ax7 = plt.subplots()
    ax7.plot(combined_years, combined_cases, label="Predicted Trend", linestyle="--", color="green")
    ax7.scatter(X, y, label="Historical Reported Cases", color="blue")
    ax7.set_xlabel("Year")
    ax7.set_ylabel("Reported Cases")
    ax7.set_title("Reported Malaria Cases + Future Predictions")
    ax7.legend()
    st.pyplot(fig7)
else:
    st.warning("Not enough historical data to build a prediction model. Need at least 5 years.")

# Future Work Section
st.markdown("#### 🔮 Future Predictions")
st.info("Model predictions will be added in a future version using machine learning. This version focuses on historical analysis.")

st.markdown("---")
st.markdown("✅ *Dashboard built for PLP AI4SE Week 2 — MalariaShield by Group 39*")

