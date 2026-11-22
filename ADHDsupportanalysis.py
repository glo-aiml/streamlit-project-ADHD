import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px

# Load data
features = pd.read_csv("features.csv", sep=";")
patient_info = pd.read_csv("patient_info.csv", sep=";")

# Merge datasets
data = features.merge(patient_info, on="ID", how="inner")

# Features to explore
selected_features = [
    "ACC__mean",
    "ACC__standard_deviation",
    "ACC__root_mean_square",
    "ACC__maximum",
    "ACC__mean_abs_change",
    "ACC__sample_entropy"
]

# Friendly display names
feature_labels = {
    "ACC__mean": "Average Activity",
    "ACC__standard_deviation": "Movement Variability",
    "ACC__root_mean_square": "Overall Activity Level",
    "ACC__maximum": "Peak Activity",
    "ACC__mean_abs_change": "Activity Change Rate",
    "ACC__sample_entropy": "Movement Complexity"
}

st.title("Exploring Activity Patterns as Potential ADHD Support Signals")

st.write(
    """
    This dashboard explores whether activity-based measurements may help
    identify when people with ADHD could benefit from additional support.
    """
)

# Dataset overview
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Participants", len(data))

with col2:
    st.metric("ADHD Participants", int(data["ADHD"].sum()))

with col3:
    st.metric("Controls", int((data["ADHD"] == 0).sum()))

# Feature selector
st.subheader("Activity Feature Comparison")

selected_metric = st.selectbox(
    "Select Activity Metric",
    selected_features,
    format_func=lambda x: feature_labels[x]
)

# ADHD labels
plot_data = data.copy()
plot_data["Group"] = plot_data["ADHD"].map(
    {1: "ADHD", 0: "Control"}
)

# Boxplot
fig = px.box(
    plot_data,
    x="Group",
    y=selected_metric,
    color="Group",
    title=f"{feature_labels[selected_metric]} by Group",
    labels={
        selected_metric: feature_labels[selected_metric],
        "Group": "Participant Group"
    }
)
st.plotly_chart(fig, use_container_width=True)

# Summary statistics
st.subheader("Summary Statistics")

summary = (
    plot_data
    .groupby("Group")[selected_metric]
    .agg(["mean", "median", "std"])
    .round(2)
)

st.dataframe(summary)

# Symptom relationship
st.subheader("Relationship with ADHD Symptoms")

fig2 = px.scatter(
    plot_data,
    x="ASRS",
    y=selected_metric,
    color="Group",
    hover_data=["ID"],
    title=f"ASRS Score vs {feature_labels[selected_metric]}"
)

st.plotly_chart(fig2, use_container_width=True)

# Potential Support Signals
st.subheader("Signals That Could Inform Future ADHD Support Systems")

st.caption(
    "This section highlights measurable signals that may help identify periods of increased ADHD-related difficulty before challenges become disruptive."
)

support_signals = pd.DataFrame({
    "Signal": [
        "Sleep",
        "Stress",
        "Physical Activity",
        "HRV"
    ],
    "Evidence Strength": [
        "Very High",
        "High",
        "High",
        "Moderate-High"
    ],
    "Potential Impact": [
        "Attention & Executive Function",
        "Emotional Regulation",
        "Focus & Mood",
        "Stress Resilience"
    ]
})

st.dataframe(support_signals)

st.info(
    """
    Important: The activity measurements shown in this dashboard are not intended
    to diagnose ADHD or predict behavior with certainty.

    The purpose of this analysis is to explore whether objective signals such as
    movement patterns, sleep, stress, heart rate variability (HRV), and other
    physiological measures could help identify when an individual with ADHD may
    benefit from additional support.

    Future assistive technologies could potentially use these signals to provide
    timely reminders, task support, routine guidance, and emotional regulation
    interventions when they are most needed.
    """
)

# Dataset Citation:
st.subheader("Dataset Acknowledgment")

st.markdown("""
This project uses the HYPERAKTIV dataset.

**Citation:**

Hicks, S., Stautland, A., Fasmer, O. B., Førland, W.,
Hammer, H. L., Halvorsen, P., Mjeldheim, K., Oedegaard,
K. J., Osnes, B., Syrstad, V. E. G., Riegler, M.,
& Jakobsen, P. (2021).

*HYPERAKTIV: An Activity Dataset from Adult Patients with
Attention-Deficit/Hyperactivity Disorder (ADHD).*

Proceedings of the 12th ACM Multimedia Systems Conference.

DOI: 10.1145/3458305.3478454
""")
