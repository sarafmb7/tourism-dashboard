import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Tourism Dashboard", layout="wide")

# =====================
# 📥 تحميل البيانات
# =====================
df = pd.read_csv("Tourism.csv")

df = df.drop(columns=['Unnamed: 3'], errors='ignore')
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])

st.title("🌍 Tourism Flow Dashboard")

# =====================
# 📊 البيانات
# =====================
st.subheader("📄 Data Preview")
st.write(df.head())

# =====================
# 🏙️ تحليل المدن
# =====================
st.subheader("🏙️ Visitors by City")

city_visitors = df.groupby('City')['Visitors'].sum().sort_values(ascending=False)

fig1, ax1 = plt.subplots()
city_visitors.plot(kind='bar', ax=ax1)
st.pyplot(fig1)

# =====================
# 📅 التدفق الزمني
# =====================
st.subheader("📅 Tourist Flow Over Time")

daily = df.groupby('Date')['Visitors'].sum()

fig2, ax2 = plt.subplots()
ax2.plot(daily.index, daily.values)
st.pyplot(fig2)

# =====================
# 🔮 التنبؤ
# =====================
st.subheader("🔮 Forecast (30 Days)")

prophet_df = df.groupby('Date')['Visitors'].sum().reset_index()
prophet_df.columns = ['ds', 'y']

model = Prophet()
model.fit(prophet_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig3 = model.plot(forecast)
st.pyplot(fig3)
