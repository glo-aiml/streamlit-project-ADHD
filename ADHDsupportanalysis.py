import pandas as pd
import streamlit as st
import plotly.express as px


features = pd.read_csv("features.csv", sep=";")
patient_info = pd.read_csv("patient_info.csv", sep=";")

data = features.merge(patient_info, on="ID", how="inner")

# Create readable group labels
data["Group"] = data["ADHD"].map({
    1: "ADHD",
    0: "Control"
})

# App Title
st.title("ADHD Symptoms, Anxiety, and Depression Analysis")

st.write("""
This dashboard explores relationships between ADHD symptoms,
anxiety, and depression indicators within the HYPERAKTIV dataset.
The goal is to investigate factors associated with ADHD symptom
severity and identify areas that may warrant additional support.
""")

# Dataset Overview

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Participants", len(data))

with col2:
    st.metric("ADHD Participants", int(data["ADHD"].sum()))

with col3:
    st.metric("Control Participants", int((data["ADHD"] == 0).sum()))

# Score Distributions
st.subheader("Distribution of ADHD Symptom Scores")

fig = px.histogram(
    data,
    x="ASRS",
    color="Group",
    nbins=20,
    title="Distribution of ADHD Self-Report Scale (ASRS) Scores"
)

st.plotly_chart(fig, use_container_width=True)

# ADHD Symptoms vs Anxiety
st.subheader("ADHD Symptoms vs Anxiety")

st.caption("""
ASRS measures ADHD symptom severity.
HADS-A measures anxiety symptoms.
""")

fig = px.scatter(
    data,
    x="ASRS",
    y="HADS_A",
    color="Group",
    trendline="ols",
    hover_data=["ID"],
    title="ASRS Score vs Anxiety (HADS-A)",
    labels={
        "ASRS": "ADHD Symptom Score (ASRS)",
        "HADS_A": "Anxiety Score (HADS-A)"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ADHD Symptoms vs Depression
st.subheader("ADHD Symptoms vs Depression")

st.caption("""
HADS-D measures depression symptoms.
""")

fig = px.scatter(
    data,
    x="ASRS",
    y="HADS_D",
    color="Group",
    trendline="ols",
    hover_data=["ID"],
    title="ASRS Score vs Depression (HADS-D)",
    labels={
        "ASRS": "ADHD Symptom Score (ASRS)",
        "HADS_D": "Depression Score (HADS-D)"
    }
)

st.plotly_chart(fig, use_container_width=True)

# ADHD Symptoms vs Depression Severity
st.subheader("ADHD Symptoms vs Depression Severity")

st.caption("""
MADRS is a clinical measure of depression severity.
""")

fig = px.scatter(
    data,
    x="ASRS",
    y="MADRS",
    color="Group",
    trendline="ols",
    hover_data=["ID"],
    title="ASRS Score vs MADRS ",
    labels={
        "ASRS": "ADHD Symptom Score (ASRS)",
        "MADRS": "Depression Severity Score (MADRS)"
    }
)

st.plotly_chart(fig, use_container_width=True)

# Correlation Summary
st.subheader("Correlation Summary")

correlations = pd.DataFrame({
    "Measure": [
        "Anxiety (HADS-A)",
        "Depression (HADS-D)",
        "Depression Severity (MADRS)"
    ],
    "Correlation with ASRS": [
        data["ASRS"].corr(data["HADS_A"]),
        data["ASRS"].corr(data["HADS_D"]),
        data["ASRS"].corr(data["MADRS"])
    ]
})

correlations["Correlation with ASRS"] = (
    correlations["Correlation with ASRS"]
    .round(3)
)

st.dataframe(
    correlations,
    use_container_width=True
)

# Key Findings
st.subheader("Key Findings")

st.info("""
This dashboard explores associations between ADHD symptom scores,
anxiety, and depression measures within the HYPERAKTIV dataset.

The visualizations can help identify whether higher ADHD symptom
scores tend to occur alongside higher anxiety or depression scores.

These findings are exploratory and should not be interpreted as
evidence of causation.
""")

# Dataset Acknowledgment
st.subheader("Dataset Acknowledgment")

st.markdown("""
**HYPERAKTIV Dataset**

Hicks et al. (2021)

*HYPERAKTIV: An Activity Dataset from Adult Patients with
Attention-Deficit/Hyperactivity Disorder (ADHD)*

Proceedings of the 12th ACM Multimedia Systems Conference

DOI: 10.1145/3458305.3478454
""")
