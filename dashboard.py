import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page config
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")

# Title
st.title("📊 Customer Churn Analysis Dashboard")

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Basic Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# -------------------------
# 🔹 Top Metrics Section
# -------------------------
col1, col2, col3 = st.columns(3)

total_customers = df.shape[0]
churn_count = df[df['Churn'] == 'Yes'].shape[0]
churn_rate = round((churn_count / total_customers) * 100, 2)

col1.metric("Total Customers", total_customers)
col2.metric("Churned Customers", churn_count)
col3.metric("Churn Rate (%)", churn_rate)

st.markdown("---")

# -------------------------
# 🔹 Sidebar Filters
# -------------------------
st.sidebar.header("Filter Data")

contract_filter = st.sidebar.selectbox(
    "Select Contract Type",
    options=["All"] + list(df['Contract'].unique())
)

gender_filter = st.sidebar.selectbox(
    "Select Gender",
    options=["All"] + list(df['gender'].unique())
)

# Apply filters
filtered_df = df.copy()

if contract_filter != "All":
    filtered_df = filtered_df[filtered_df['Contract'] == contract_filter]

if gender_filter != "All":
    filtered_df = filtered_df[filtered_df['gender'] == gender_filter]

# -------------------------
# 🔹 Plots Section
# -------------------------

col4, col5 = st.columns(2)

# Small and neat figure size
fig1, ax1 = plt.subplots(figsize=(4,3))
sns.countplot(data=filtered_df, x='Churn', palette='Set2', ax=ax1)
ax1.set_title("Churn Distribution", fontsize=10)
ax1.set_xlabel("")
ax1.set_ylabel("Count")
col4.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(4,3))
sns.countplot(data=filtered_df, x='Contract', hue='Churn', palette='Set1', ax=ax2)
ax2.set_title("Churn by Contract Type", fontsize=10)
ax2.set_xlabel("")
ax2.set_ylabel("Count")
plt.xticks(rotation=30)
col5.pyplot(fig2)

# -------------------------
# 🔹 Tenure Analysis
# -------------------------
st.markdown("### 📈 Tenure vs Churn")

fig3, ax3 = plt.subplots(figsize=(6,3))
sns.boxplot(data=filtered_df, x='Churn', y='tenure', palette='coolwarm', ax=ax3)
ax3.set_title("Tenure Distribution by Churn", fontsize=10)
st.pyplot(fig3)
