#!/usr/bin/env python3
"""
Main entry point for the Divorce Prediction System
This file should be used when deploying to platforms like Streamlit Cloud, Heroku, Railway
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import pickle

# The dual perspective predictor is loaded from the saved model file

# Initialize session state for navigation
if 'current_screen' not in st.session_state:
    st.session_state.current_screen = 1

# Screen configuration
SCREENS = {
    1: {"title": "👨 Man's Personal Characteristics", "type": "man_personal"},
    2: {"title": "👨 Man's Relationship Perspective", "type": "man_relationship"},
    3: {"title": "👩 Woman's Personal Characteristics", "type": "woman_personal"},
    4: {"title": "👩 Woman's Relationship Perspective", "type": "woman_relationship"},
    5: {"title": "💑 Shared Relationship Factors", "type": "shared"},
    6: {"title": "📊 Prediction Results", "type": "results"}
}

# Load dual-perspective model and configuration
with open("model/divorce_model.pkl", "rb") as f:
    predictor = pickle.load(f)  # This is already a DualPerspectivePredictor object

with open("model/divorce_features.pkl", "rb") as f:
    feature_list = pickle.load(f)

with open("model/feature_categories.pkl", "rb") as f:
    feature_categories = pickle.load(f)

# Get feature categories
individual_features = feature_categories['individual']
shared_features = feature_categories['shared']
perspective_dependent_features = feature_categories['perspective_dependent']

# Streamlit app
st.set_page_config(page_title="Divorce Prediction System", page_icon="💔", layout="wide")

# Initialize input dictionaries in session state
if 'man_inputs' not in st.session_state:
    st.session_state.man_inputs = {}
if 'woman_inputs' not in st.session_state:
    st.session_state.woman_inputs = {}
if 'shared_inputs' not in st.session_state:
    st.session_state.shared_inputs = {}

def navigate_to_screen(screen_number):
    st.session_state.current_screen = screen_number
    st.rerun()

def create_navigation_buttons():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.session_state.current_screen > 1:
            if st.button("⬅️ Previous", key="prev_btn"):
                navigate_to_screen(st.session_state.current_screen - 1)
    
    with col2:
        st.markdown(f"<h4 style='text-align: center;'>Screen {st.session_state.current_screen} of {len(SCREENS)}</h4>", unsafe_allow_html=True)
    
    with col3:
        if st.session_state.current_screen < len(SCREENS):
            if st.button("Next ➡️", key="next_btn"):
                navigate_to_screen(st.session_state.current_screen + 1)

# Education levels for reuse
education_levels = [
    "Completed Primary School",
    "Completed Secondary School", 
    "Completed High School",
    "HND",
    "Bachelor or Associate degree",
    "Masters",
    "PhD"
]
education_values = [10, 20, 30, 40, 50, 60, 70]

# Mental health levels for reuse
mental_health_levels = [
    "Stable (Gets sick normally)",
    "Unstable (Gets sick abnormally)", 
    "Health condition (Lives with a medical condition)"
]
mental_health_values = [80, 50, 30]

# Income levels for reuse
income_levels = [
    "Bad Income Earner",
    "Average Income Earner",
    "Good Income Earner", 
    "High Income Earner"
]
income_values = [25, 50, 75, 90]

# Addiction levels for reuse
addiction_levels = [
    "No addiction",
    "Mild addiction (occasional use)",
    "Moderate addiction (regular use)",
    "Severe addiction (dependency)"
]
addiction_values = [90, 70, 40, 15]

# Independence levels for reuse
independence_levels = [
    "Completely dependent (cannot function alone)",
    "Mostly dependent (needs significant help)",
    "Somewhat independent (needs some help)",
    "Mostly independent (minimal help needed)",
    "Completely independent (fully self-sufficient)"
]
independence_values = [20, 35, 55, 75, 90]

# Socializing age levels for reuse
socializing_age_levels = [
    "Very early (before 13)",
    "Early teens (13-15)",
    "Mid teens (16-17)",
    "Late teens (18-19)",
    "Early twenties (20-22)",
    "Late start (23+)"
]
socializing_age_values = [30, 50, 70, 80, 75, 60]

def render_man_personal_screen():
    st.markdown("**Enter the man's personal characteristics:**")

    for feature in individual_features:
        if feature == "Education":
            selected_education = st.selectbox(
                "Man's Education Level",
                education_levels,
                index=4,
                key="man_education"
            )
            st.session_state.man_inputs[feature] = education_values[education_levels.index(selected_education)]

        elif feature == "Mental Health":
            selected_mental_health = st.selectbox(
                "Man's Mental Health",
                mental_health_levels,
                index=0,
                key="man_mental_health"
            )
            st.session_state.man_inputs[feature] = mental_health_values[mental_health_levels.index(selected_mental_health)]

        elif feature == "Self Confidence":
            confidence_rating = st.slider(
                "Man's Self Confidence (0-5 stars)",
                min_value=0.0, max_value=5.0, value=3.5, step=0.1,
                key="man_confidence"
            )
            st.session_state.man_inputs[feature] = confidence_rating * 20

        elif feature == "Good Income":
            selected_income = st.selectbox(
                "Man's Income Level",
                income_levels,
                index=2,
                key="man_income"
            )
            st.session_state.man_inputs[feature] = income_values[income_levels.index(selected_income)]

        elif feature == "Addiction":
            selected_addiction = st.selectbox(
                "Man's Addiction Level",
                addiction_levels,
                index=0,
                key="man_addiction"
            )
            st.session_state.man_inputs[feature] = addiction_values[addiction_levels.index(selected_addiction)]

        elif feature == "Independency":
            selected_independence = st.selectbox(
                "Man's Independence Level",
                independence_levels,
                index=3,
                key="man_independence"
            )
            st.session_state.man_inputs[feature] = independence_values[independence_levels.index(selected_independence)]

        elif feature == "Start Socializing with the Opposite Sex Age ":
            selected_socializing_age = st.selectbox(
                "Man's Age When Started Dating",
                socializing_age_levels,
                index=3,
                key="man_socializing_age"
            )
            st.session_state.man_inputs[feature] = socializing_age_values[socializing_age_levels.index(selected_socializing_age)]

        else:
            value = st.slider(
                f"Man's {feature}",
                min_value=0.0, max_value=100.0, value=50.0,
                key=f"man_{feature.replace(' ', '_').lower()}"
            )
            st.session_state.man_inputs[feature] = value

def render_man_relationship_screen():
    st.markdown("**Enter the man's relationship perspective:**")

    # Social gap levels
    social_gap_levels = [
        "No social gap (same social status)",
        "Small social gap (minor differences)",
        "Moderate social gap (noticeable differences)",
        "Large social gap (significant differences)",
        "Very large social gap (major class differences)"
    ]
    social_gap_values = [85, 70, 55, 40, 25]

    # Desire to marry levels
    desire_marry_levels = [
        "No desire to marry",
        "Little desire to marry",
        "Moderate desire to marry",
        "Strong desire to marry",
        "Very strong desire to marry"
    ]
    desire_marry_values = [20, 40, 60, 80, 95]

    for feature in perspective_dependent_features:
        if feature == "Social Gap":
            selected_social_gap = st.selectbox(
                "Man's view of social gap with woman",
                social_gap_levels,
                index=1,
                key="man_social_gap"
            )
            st.session_state.man_inputs[feature] = social_gap_values[social_gap_levels.index(selected_social_gap)]

        elif feature == "Desire to Marry":
            selected_desire_marry = st.selectbox(
                "Man's desire to marry",
                desire_marry_levels,
                index=3,
                key="man_desire_marry"
            )
            st.session_state.man_inputs[feature] = desire_marry_values[desire_marry_levels.index(selected_desire_marry)]

        # Add other perspective-dependent features here...
        else:
            value = st.slider(
                f"Man's view: {feature}",
                min_value=0.0, max_value=100.0, value=50.0,
                key=f"man_persp_{feature.replace(' ', '_').lower()}"
            )
            st.session_state.man_inputs[feature] = value

def render_woman_personal_screen():
    st.markdown("**Enter the woman's personal characteristics:**")

    for feature in individual_features:
        if feature == "Education":
            selected_education = st.selectbox(
                "Woman's Education Level",
                education_levels,
                index=4,
                key="woman_education"
            )
            st.session_state.woman_inputs[feature] = education_values[education_levels.index(selected_education)]

        elif feature == "Mental Health":
            selected_mental_health = st.selectbox(
                "Woman's Mental Health",
                mental_health_levels,
                index=0,
                key="woman_mental_health"
            )
            st.session_state.woman_inputs[feature] = mental_health_values[mental_health_levels.index(selected_mental_health)]

        elif feature == "Self Confidence":
            confidence_rating = st.slider(
                "Woman's Self Confidence (0-5 stars)",
                min_value=0.0, max_value=5.0, value=3.5, step=0.1,
                key="woman_confidence"
            )
            st.session_state.woman_inputs[feature] = confidence_rating * 20

        elif feature == "Good Income":
            selected_income = st.selectbox(
                "Woman's Income Level",
                income_levels,
                index=2,
                key="woman_income"
            )
            st.session_state.woman_inputs[feature] = income_values[income_levels.index(selected_income)]

        elif feature == "Addiction":
            selected_addiction = st.selectbox(
                "Woman's Addiction Level",
                addiction_levels,
                index=0,
                key="woman_addiction"
            )
            st.session_state.woman_inputs[feature] = addiction_values[addiction_levels.index(selected_addiction)]

        elif feature == "Independency":
            selected_independence = st.selectbox(
                "Woman's Independence Level",
                independence_levels,
                index=3,
                key="woman_independence"
            )
            st.session_state.woman_inputs[feature] = independence_values[independence_levels.index(selected_independence)]

        elif feature == "Start Socializing with the Opposite Sex Age ":
            selected_socializing_age = st.selectbox(
                "Woman's Age When Started Dating",
                socializing_age_levels,
                index=3,
                key="woman_socializing_age"
            )
            st.session_state.woman_inputs[feature] = socializing_age_values[socializing_age_levels.index(selected_socializing_age)]

        else:
            value = st.slider(
                f"Woman's {feature}",
                min_value=0.0, max_value=100.0, value=50.0,
                key=f"woman_{feature.replace(' ', '_').lower()}"
            )
            st.session_state.woman_inputs[feature] = value

def render_woman_relationship_screen():
    st.markdown("**Enter the woman's relationship perspective:**")

    # Social gap levels
    social_gap_levels = [
        "No social gap (same social status)",
        "Small social gap (minor differences)",
        "Moderate social gap (noticeable differences)",
        "Large social gap (significant differences)",
        "Very large social gap (major class differences)"
    ]
    social_gap_values = [85, 70, 55, 40, 25]

    # Desire to marry levels
    desire_marry_levels = [
        "No desire to marry",
        "Little desire to marry",
        "Moderate desire to marry",
        "Strong desire to marry",
        "Very strong desire to marry"
    ]
    desire_marry_values = [20, 40, 60, 80, 95]

    for feature in perspective_dependent_features:
        if feature == "Social Gap":
            selected_social_gap = st.selectbox(
                "Woman's view of social gap with man",
                social_gap_levels,
                index=1,
                key="woman_social_gap"
            )
            st.session_state.woman_inputs[feature] = social_gap_values[social_gap_levels.index(selected_social_gap)]

        elif feature == "Desire to Marry":
            selected_desire_marry = st.selectbox(
                "Woman's desire to marry",
                desire_marry_levels,
                index=3,
                key="woman_desire_marry"
            )
            st.session_state.woman_inputs[feature] = desire_marry_values[desire_marry_levels.index(selected_desire_marry)]

        # Add other perspective-dependent features here...
        else:
            value = st.slider(
                f"Woman's view: {feature}",
                min_value=0.0, max_value=100.0, value=50.0,
                key=f"woman_persp_{feature.replace(' ', '_').lower()}"
            )
            st.session_state.woman_inputs[feature] = value

def render_shared_screen():
    st.markdown("**Enter shared relationship characteristics:**")

    for feature in shared_features:
        if feature == "Age Gap":
            age_gap = st.slider(
                "Age Gap (years)",
                min_value=0, max_value=50, value=3,
                key="age_gap"
            )
            st.session_state.shared_inputs[feature] = age_gap

        elif feature == "Economic Similarity":
            economic_levels = [
                "Similar",
                "Slightly similar",
                "Not similar",
                "Completely different"
            ]
            economic_values = [85, 70, 45, 25]

            selected_economic = st.selectbox(
                "Economic Similarity",
                economic_levels,
                index=0,
                key="economic_similarity"
            )
            st.session_state.shared_inputs[feature] = economic_values[economic_levels.index(selected_economic)]

        else:
            value = st.slider(
                f"{feature}",
                min_value=0.0, max_value=100.0, value=50.0,
                key=f"shared_{feature.replace(' ', '_').lower()}"
            )
            st.session_state.shared_inputs[feature] = value

def render_results_screen():
    st.markdown("**Divorce Prediction Results:**")

    if st.button("🔍 Generate Predictions", type="primary"):
        try:
            # Get predictions from both perspectives
            results = predictor.predict_from_perspective(
                st.session_state.man_inputs,
                st.session_state.woman_inputs,
                st.session_state.shared_inputs
            )

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("👨 Man's Perspective")
                man_risk = "High Risk" if results['man_prediction'] == 1 else "Low Risk"
                man_color = "red" if results['man_prediction'] == 1 else "green"
                st.markdown(f"**Divorce Risk:** <span style='color: {man_color}'>{man_risk}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {results['man_probability']:.1%}")

            with col2:
                st.subheader("👩 Woman's Perspective")
                woman_risk = "High Risk" if results['woman_prediction'] == 1 else "Low Risk"
                woman_color = "red" if results['woman_prediction'] == 1 else "green"
                st.markdown(f"**Divorce Risk:** <span style='color: {woman_color}'>{woman_risk}</span>", unsafe_allow_html=True)
                st.markdown(f"**Confidence:** {results['woman_probability']:.1%}")

        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")

    # Show summary of inputs
    with st.expander("📋 Input Summary"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Man's Inputs:**")
            for key, value in st.session_state.man_inputs.items():
                st.write(f"{key}: {value}")

        with col2:
            st.write("**Woman's Inputs:**")
            for key, value in st.session_state.woman_inputs.items():
                st.write(f"{key}: {value}")

        with col3:
            st.write("**Shared Inputs:**")
            for key, value in st.session_state.shared_inputs.items():
                st.write(f"{key}: {value}")

# Main app title
st.title("💔 Dual-Perspective Divorce Prediction System")
current_screen = SCREENS[st.session_state.current_screen]
st.markdown(f"### {current_screen['title']}")

# Progress bar
progress = (st.session_state.current_screen - 1) / (len(SCREENS) - 1)
st.progress(progress)

st.markdown("---")

# Render current screen
if current_screen['type'] == 'man_personal':
    render_man_personal_screen()
elif current_screen['type'] == 'man_relationship':
    render_man_relationship_screen()
elif current_screen['type'] == 'woman_personal':
    render_woman_personal_screen()
elif current_screen['type'] == 'woman_relationship':
    render_woman_relationship_screen()
elif current_screen['type'] == 'shared':
    render_shared_screen()
elif current_screen['type'] == 'results':
    render_results_screen()

st.markdown("---")
create_navigation_buttons()

# Main execution block for running directly
if __name__ == "__main__":
    # This allows the script to be run directly with: streamlit run streamlit_app.py
    pass
