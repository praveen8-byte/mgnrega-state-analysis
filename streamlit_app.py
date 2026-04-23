import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

st.set_page_config(page_title="ABA Assignment Dashboard", layout="wide")
st.title("📊 MGNREGA State-Level Analysis Dashboard")

def safe_divide(a, b):
    return np.where(b == 0, 0, a / b)

@st.cache_data
def load_data():
    df = pd.read_csv("final_state_dataset.csv")
    if 'Remarks' in df.columns:
        df = df.drop(columns=['Remarks'])
    df = df.drop_duplicates()
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    df['Expenditure_per_Person'] = safe_divide(df['Total_Exp'], df['Total_Individuals_Worked'])
    df['Women_Share'] = safe_divide(df['Women_Persondays'], df['Total_Individuals_Worked']) * 100
    df['Work_Completion_Rate'] = safe_divide(df['Number_of_Completed_Works'], df['Total_No_of_Works_Takenup']) * 100

    state_df = df.groupby('state_name', as_index=False).agg({
        'Total_Exp': 'sum',
        'Total_Individuals_Worked': 'sum',
        'Total_Households_Worked': 'sum',
        'Average_days_of_employment_provided_per_Household': 'mean',
        'Women_Persondays': 'sum',
        'Number_of_Completed_Works': 'sum',
        'Total_No_of_Works_Takenup': 'sum'
    })
    state_df['Women_Share'] = safe_divide(state_df['Women_Persondays'], state_df['Total_Individuals_Worked']) * 100
    state_df['Work_Completion_Rate'] = safe_divide(state_df['Number_of_Completed_Works'], state_df['Total_No_of_Works_Takenup']) * 100
    state_df['Expenditure_per_Person'] = safe_divide(state_df['Total_Exp'], state_df['Total_Individuals_Worked'])
    return state_df

state_df = load_data()

st.sidebar.header("Filters")
all_states = state_df['state_name'].tolist()
selected_states = st.sidebar.multiselect("Select States", all_states, default=all_states)
filtered_df = state_df[state_df['state_name'].isin(selected_states)]

st.subheader("📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Expenditure", f"₹{filtered_df['Total_Exp'].sum():,.0f}")
col2.metric("Total Individuals Worked", f"{filtered_df['Total_Individuals_Worked'].sum():,.0f}")
col3.metric("Avg Employment Days", f"{filtered_df['Average_days_of_employment_provided_per_Household'].mean():.2f}")
col4.metric("Avg Work Completion Rate", f"{filtered_df['Work_Completion_Rate'].mean():.2f}%")

st.divider()

with st.expander("🗂️ View Raw Data"):
    st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)

st.divider()

st.subheader("💰 Top 10 States by Total Expenditure")
top10 = filtered_df.sort_values('Total_Exp', ascending=False).head(10)
fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.barplot(data=top10, x='Total_Exp', y='state_name', palette='Blues_r', ax=ax1)
st.pyplot(fig1)
plt.close(fig1)

st.divider()

st.subheader("🔗 Correlation Heatmap")
corr_cols = ['Total_Exp', 'Total_Individuals_Worked',
             'Average_days_of_employment_provided_per_Household',
             'Women_Share', 'Work_Completion_Rate', 'Expenditure_per_Person']
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(filtered_df[corr_cols].corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax2)
st.pyplot(fig2)
plt.close(fig2)

st.divider()

st.subheader("📈 Expenditure vs Average Employment Days")
fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=filtered_df, x='Total_Exp',
                y='Average_days_of_employment_provided_per_Household',
                hue='state_name', legend=False, ax=ax3)
for _, row in filtered_df.iterrows():
    ax3.annotate(row['state_name'],
                 (row['Total_Exp'], row['Average_days_of_employment_provided_per_Household']),
                 fontsize=6, alpha=0.7)
st.pyplot(fig3)
plt.close(fig3)

st.divider()

st.subheader("📐 OLS Regression: Predicting Average Employment Days")
X = filtered_df[['Total_Exp', 'Women_Share']].copy()
y = filtered_df['Average_days_of_employment_provided_per_Household']
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
st.text(model.summary().as_text())

st.divider()

st.subheader("👩 Women Share by State")
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.barplot(data=filtered_df.sort_values('Women_Share', ascending=False),
            x='Women_Share', y='state_name', palette='PuRd_r', ax=ax4)
st.pyplot(fig4)
plt.close(fig4)