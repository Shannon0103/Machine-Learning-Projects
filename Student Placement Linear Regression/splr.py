import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Set page configuration
st.set_page_config(page_title="Student Placement Analysis", layout="centered")

st.title("Student Placement Dashboard")
st.markdown("""
A simple and interactive dashboard to explore student placement data, predict outcomes, and view statistics.
""")

# Load the data
df = pd.read_csv("placement.csv")
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Load pre-trained logistic regression model
with open("lrmodel_placement.pkl", "rb") as f:
    model = pickle.load(f)

# Sidebar options
st.sidebar.header("Options")
show_data = st.sidebar.checkbox("Show raw data", value=True)
show_stats = st.sidebar.checkbox("Show summary statistics", value=True)
show_plot = st.sidebar.checkbox("Show scatter plot (CGPA vs IQ)", value=True)
show_predict = st.sidebar.checkbox("Predict placement", value=True)

# Data preview
if show_data:
    st.subheader("Raw Placement Data")
    st.dataframe(df.head(10))

# Statistics
if show_stats:
    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

# Scatter plot
if show_plot:
    st.subheader("CGPA vs IQ Scatter Plot")
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df, x="cgpa", y="iq", hue="placement", palette="Set2", s=80, ax=ax
    )
    ax.set_title("CGPA vs IQ by Placement Outcome")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    st.pyplot(fig)

# Placement prediction input and output
if show_predict:
    st.subheader("Predict Student Placement")
    cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=6.0, step=0.01)
    iq = st.number_input("Enter IQ", min_value=50.0, max_value=250.0, value=120.0, step=1.0)
    input_df = pd.DataFrame({"cgpa": [cgpa], "iq": [iq]})

    pred_outcome = model.predict(input_df)[0]
    result_str = "Placed" if pred_outcome == 1 else "Not Placed"
    st.success(f"Prediction: {result_str}")

# Footer
st.markdown("""
---
Made by Shannon Dsouza
""")
