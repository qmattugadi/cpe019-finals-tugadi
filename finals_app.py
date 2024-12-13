import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import calendar

st.title("NYC Taxi Rides Analysis")
st.header("Final Exam: Model Deployment in the Cloud")
st.header("Submitted by Tugadi, Marlv Andrei T.")

# Load dataset directly from the given path
data_path = 'data\dataset.csv'
df = pd.read_csv(data_path, index_col=0)

st.write("### Dataset Preview")
st.dataframe(df.head())

# Data preparation
st.markdown("## Data Preparation")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')
st.write("Converted `timestamp` to datetime and set it as the index.")
st.write("### Dataset Info")
st.text(df.info())

# Null check
st.write("### Null Values")
st.write(df.isna().sum())

# Data Visualization
st.markdown("## Visualizing Data")
st.line_chart(df['value'])

# Analyzing Specific Events
st.markdown("## Specific Events Analysis")

events = {
    "New York City Marathon": ("2014-10-30", "2014-11-03"),
    "Thanksgiving": ("2014-11-25", "2014-11-30"),
    "Snow Storm": ("2014-11-22", "2014-11-30"),
    "Christmas and New Year": ("2014-12-22", "2015-01-02")
}

for event, dates in events.items():
    st.markdown(f"### {event}")
    start_date, end_date = dates
    fig, ax = plt.subplots()
    df.loc[start_date:end_date]['value'].plot(ax=ax, title=f"{event}")
    st.pyplot(fig)

# Adding time attributes
st.markdown("## Adding Time Attributes")
df['day'] = df.index.day
df['hour'] = df.index.hour
df['weekday'] = df.index.weekday
df['month'] = df.index.month

df_day = df['value'].resample('D').mean()
st.line_chart(df_day)

# Month visualization
df['month'] = df['month'].apply(lambda x: calendar.month_name[x])
fig, ax = plt.subplots()
sns.lineplot(x="day", y="value", hue="month", data=df, ax=ax)
ax.set_title("Day Rides by Month")
st.pyplot(fig)

# Weekday visualization
df['weekday'] = df['weekday'].apply(lambda x: calendar.day_name[x])
fig, ax = plt.subplots()
sns.lineplot(x="hour", y="value", hue="weekday", data=df, ax=ax)
ax.set_title("Hour Rides by Weekday")
st.pyplot(fig)

# Time series decomposition
st.markdown("## Time Series Decomposition")
df = df[['value', 'day', 'hour']].resample('D').mean()
decomposed = seasonal_decompose(df['value'], model='additive')

fig, axes = plt.subplots(4, 1, figsize=(10, 8))
decomposed.observed.plot(ax=axes[0], title="Observed")
decomposed.trend.plot(ax=axes[1], title="Trend")
decomposed.seasonal.plot(ax=axes[2], title="Seasonal")
decomposed.resid.plot(ax=axes[3], title="Residual")
plt.tight_layout()
st.pyplot(fig)

# Stationarity check
st.markdown("## Stationarity Check")
result = adfuller(df['value'].dropna())
st.write(f"Test Statistic: {result[0]}")
st.write(f"p-value: {result[1]}")

# Autocorrelation and Partial Autocorrelation
st.markdown("## Autocorrelation Analysis")
fig, ax = plt.subplots()
plot_acf(df['value'].dropna(), lags=30, ax=ax)
st.pyplot(fig)

fig, ax = plt.subplots()
plot_pacf(df['value'].dropna(), lags=30, ax=ax)
st.pyplot(fig)

st.success("Analysis completed!")
