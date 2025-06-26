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

    # Relationship with spouse family levels
    spouse_family_levels = [
        "Strongly opposed (family actively against the spouse choice)",
        "Disapproval (family doesn't approve but tolerates)",
        "Neutral/Mixed (family has mixed feelings or indifferent)",
        "Approval (family likes and supports the spouse choice)",
        "Strong endorsement (family enthusiastically supports and celebrates)"
    ]
    spouse_family_values = [15, 35, 50, 75, 90]

    # Loyalty levels
    loyalty_levels = [
        "Very disloyal (frequently unfaithful or betrays trust)",
        "Somewhat disloyal (occasional betrayals or unfaithfulness)",
        "Neutral/Uncertain (mixed loyalty, situational faithfulness)",
        "Mostly loyal (generally faithful with minor lapses)",
        "Completely loyal (absolutely faithful and trustworthy)"
    ]
    loyalty_values = [10, 30, 50, 75, 95]

    # Relation with non-spouse before marriage levels
    non_spouse_relation_levels = [
        "Very problematic (multiple serious relationships with unresolved issues)",
        "Somewhat problematic (few relationships with some lingering complications)",
        "Neutral/Mixed (normal dating history with minor past relationship effects)",
        "Mostly positive (healthy past relationships that ended well)",
        "Very positive (excellent relationship history, learned and grew from experiences)"
    ]
    non_spouse_relation_values = [20, 40, 60, 80, 95]

    # Spouse confirmed by family levels
    spouse_confirmed_levels = [
        "Strongly rejected (family completely opposes and refuses to accept spouse)",
        "Not confirmed (family disapproves but doesn't actively oppose)",
        "Neutral/Uncertain (family has mixed feelings or is undecided)",
        "Confirmed (family approves and accepts the spouse choice)",
        "Strongly confirmed (family enthusiastically endorses and celebrates the choice)"
    ]
    spouse_confirmed_values = [10, 30, 50, 75, 95]

    # Love levels
    love_levels = [
        "No love (feels no romantic love or affection)",
        "Little love (minimal romantic feelings, mostly companionship)",
        "Moderate love (some romantic feelings but not overwhelming)",
        "Strong love (deep romantic feelings and emotional connection)",
        "Passionate love (intense, overwhelming romantic love and devotion)"
    ]
    love_values = [15, 35, 55, 80, 95]

    # Love levels
    love_levels = [
        "No love (feels no romantic love or affection)",
        "Little love (minimal romantic feelings, mostly companionship)",
        "Moderate love (some romantic feelings but not overwhelming)",
        "Strong love (deep romantic feelings and emotional connection)",
        "Passionate love (intense, overwhelming romantic love and devotion)"
    ]
    love_values = [15, 35, 55, 80, 95]

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

        elif feature == "Relationship with the Spouse Family":
            selected_spouse_family = st.selectbox(
                "Man's relationship with woman's family",
                spouse_family_levels,
                index=2,
                key="man_spouse_family"
            )
            st.session_state.man_inputs[feature] = spouse_family_values[spouse_family_levels.index(selected_spouse_family)]

        elif feature == "Loyalty":
            selected_loyalty = st.selectbox(
                "Man's loyalty in the relationship",
                loyalty_levels,
                index=3,
                key="man_loyalty"
            )
            st.session_state.man_inputs[feature] = loyalty_values[loyalty_levels.index(selected_loyalty)]

        elif feature == "Relation with Non-spouse Before Marriage":
            selected_non_spouse_relation = st.selectbox(
                "Man's past relationships before marriage",
                non_spouse_relation_levels,
                index=2,
                key="man_non_spouse_relation"
            )
            st.session_state.man_inputs[feature] = non_spouse_relation_values[non_spouse_relation_levels.index(selected_non_spouse_relation)]

        elif feature == "Spouse Confirmed by Family":
            selected_spouse_confirmed = st.selectbox(
                "How much man's family confirms/accepts the woman",
                spouse_confirmed_levels,
                index=3,
                key="man_spouse_confirmed"
            )
            st.session_state.man_inputs[feature] = spouse_confirmed_values[spouse_confirmed_levels.index(selected_spouse_confirmed)]

        elif feature == "Love":
            selected_love = st.selectbox(
                "Man's level of love for the woman",
                love_levels,
                index=3,
                key="man_love"
            )
            st.session_state.man_inputs[feature] = love_values[love_levels.index(selected_love)]

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

    # Relationship with spouse family levels
    spouse_family_levels = [
        "Strongly opposed (family actively against the spouse choice)",
        "Disapproval (family doesn't approve but tolerates)",
        "Neutral/Mixed (family has mixed feelings or indifferent)",
        "Approval (family likes and supports the spouse choice)",
        "Strong endorsement (family enthusiastically supports and celebrates)"
    ]
    spouse_family_values = [15, 35, 50, 75, 90]

    # Loyalty levels
    loyalty_levels = [
        "Very disloyal (frequently unfaithful or betrays trust)",
        "Somewhat disloyal (occasional betrayals or unfaithfulness)",
        "Neutral/Uncertain (mixed loyalty, situational faithfulness)",
        "Mostly loyal (generally faithful with minor lapses)",
        "Completely loyal (absolutely faithful and trustworthy)"
    ]
    loyalty_values = [10, 30, 50, 75, 95]

    # Relation with non-spouse before marriage levels
    non_spouse_relation_levels = [
        "Very problematic (multiple serious relationships with unresolved issues)",
        "Somewhat problematic (few relationships with some lingering complications)",
        "Neutral/Mixed (normal dating history with minor past relationship effects)",
        "Mostly positive (healthy past relationships that ended well)",
        "Very positive (excellent relationship history, learned and grew from experiences)"
    ]
    non_spouse_relation_values = [20, 40, 60, 80, 95]

    # Spouse confirmed by family levels
    spouse_confirmed_levels = [
        "Strongly rejected (family completely opposes and refuses to accept spouse)",
        "Not confirmed (family disapproves but doesn't actively oppose)",
        "Neutral/Uncertain (family has mixed feelings or is undecided)",
        "Confirmed (family approves and accepts the spouse choice)",
        "Strongly confirmed (family enthusiastically endorses and celebrates the choice)"
    ]
    spouse_confirmed_values = [10, 30, 50, 75, 95]

    # Love levels
    love_levels = [
        "No love (feels no romantic love or affection)",
        "Little love (minimal romantic feelings, mostly companionship)",
        "Moderate love (some romantic feelings but not overwhelming)",
        "Strong love (deep romantic feelings and emotional connection)",
        "Passionate love (intense, overwhelming romantic love and devotion)"
    ]
    love_values = [15, 35, 55, 80, 95]

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

        elif feature == "Relationship with the Spouse Family":
            selected_spouse_family = st.selectbox(
                "Woman's relationship with man's family",
                spouse_family_levels,
                index=2,
                key="woman_spouse_family"
            )
            st.session_state.woman_inputs[feature] = spouse_family_values[spouse_family_levels.index(selected_spouse_family)]

        elif feature == "Loyalty":
            selected_loyalty = st.selectbox(
                "Woman's loyalty in the relationship",
                loyalty_levels,
                index=3,
                key="woman_loyalty"
            )
            st.session_state.woman_inputs[feature] = loyalty_values[loyalty_levels.index(selected_loyalty)]

        elif feature == "Relation with Non-spouse Before Marriage":
            selected_non_spouse_relation = st.selectbox(
                "Woman's past relationships before marriage",
                non_spouse_relation_levels,
                index=2,
                key="woman_non_spouse_relation"
            )
            st.session_state.woman_inputs[feature] = non_spouse_relation_values[non_spouse_relation_levels.index(selected_non_spouse_relation)]

        elif feature == "Spouse Confirmed by Family":
            selected_spouse_confirmed = st.selectbox(
                "How much woman's family confirms/accepts the man",
                spouse_confirmed_levels,
                index=3,
                key="woman_spouse_confirmed"
            )
            st.session_state.woman_inputs[feature] = spouse_confirmed_values[spouse_confirmed_levels.index(selected_spouse_confirmed)]

        elif feature == "Love":
            selected_love = st.selectbox(
                "Woman's level of love for the man",
                love_levels,
                index=3,
                key="woman_love"
            )
            st.session_state.woman_inputs[feature] = love_values[love_levels.index(selected_love)]

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

        elif feature == "Cultural Similarities":
            cultural_similarity_levels = [
                "Very different cultures (completely different traditions, values, customs)",
                "Somewhat different (different cultural backgrounds with some overlap)",
                "Moderately similar (some shared cultural values and traditions)",
                "Very similar (same cultural background and shared values)",
                "Identical cultural backgrounds (same ethnicity, traditions, and customs)"
            ]
            cultural_similarity_values = [15, 35, 55, 80, 95]

            selected_cultural_similarity = st.selectbox(
                "Cultural Similarities",
                cultural_similarity_levels,
                index=2,
                key="cultural_similarities"
            )
            st.session_state.shared_inputs[feature] = cultural_similarity_values[cultural_similarity_levels.index(selected_cultural_similarity)]

        elif feature == "Common Interests":
            common_interests_levels = [
                "No common interests (completely different hobbies and activities)",
                "Few common interests (minimal overlap in activities and hobbies)",
                "Some common interests (moderate overlap in leisure activities)",
                "Many common interests (significant shared hobbies and activities)",
                "Almost identical interests (love doing the same things together)"
            ]
            common_interests_values = [20, 40, 60, 80, 95]

            selected_common_interests = st.selectbox(
                "Common Interests",
                common_interests_levels,
                index=2,
                key="common_interests"
            )
            st.session_state.shared_inputs[feature] = common_interests_values[common_interests_levels.index(selected_common_interests)]

        elif feature == "Religion Compatibility":
            religion_compatibility_levels = [
                "Completely incompatible (conflicting religious beliefs and practices)",
                "Mostly incompatible (different religions with some tolerance)",
                "Somewhat compatible (different beliefs but mutual respect)",
                "Very compatible (same religion or complementary spiritual beliefs)",
                "Perfectly compatible (identical religious beliefs and practices)"
            ]
            religion_compatibility_values = [15, 35, 55, 80, 95]

            selected_religion_compatibility = st.selectbox(
                "Religion Compatibility",
                religion_compatibility_levels,
                index=3,
                key="religion_compatibility"
            )
            st.session_state.shared_inputs[feature] = religion_compatibility_values[religion_compatibility_levels.index(selected_religion_compatibility)]

        elif feature == "No of Children from Previous Marriage":
            children_options = [
                "0 children",
                "1 child",
                "2 children",
                "3 children",
                "4 children",
                "5+ children"
            ]
            children_values = [0, 1, 2, 3, 4, 5]

            selected_children = st.selectbox(
                "Number of Children from Previous Marriage (combined total)",
                children_options,
                index=0,
                key="no_children_previous_marriage"
            )
            st.session_state.shared_inputs[feature] = children_values[children_options.index(selected_children)]

        elif feature == "Engagement Time":
            engagement_time_levels = [
                "Very short (less than 6 months - rushed decision)",
                "Short (6 months to 1 year - quick engagement)",
                "Moderate (1-2 years - adequate time to know each other)",
                "Long (2-3 years - thorough preparation and planning)",
                "Very long (3+ years - extensive courtship and preparation)"
            ]
            engagement_time_values = [30, 50, 75, 85, 70]

            selected_engagement_time = st.selectbox(
                "Engagement Time Duration",
                engagement_time_levels,
                index=2,
                key="engagement_time"
            )
            st.session_state.shared_inputs[feature] = engagement_time_values[engagement_time_levels.index(selected_engagement_time)]

        elif feature == "Commitment":
            commitment_levels = [
                "Very low commitment (one or both partners show little dedication)",
                "Low commitment (minimal effort to maintain the relationship)",
                "Moderate commitment (average dedication with some effort)",
                "High commitment (strong dedication and effort from both partners)",
                "Very high commitment (exceptional dedication and unwavering devotion)"
            ]
            commitment_values = [15, 35, 55, 80, 95]

            selected_commitment = st.selectbox(
                "Overall Commitment Level",
                commitment_levels,
                index=3,
                key="commitment"
            )
            st.session_state.shared_inputs[feature] = commitment_values[commitment_levels.index(selected_commitment)]

        elif feature == "The Sense of Having Children":
            children_sense_levels = [
                "Strongly opposed (one or both partners strongly against having children)",
                "Not interested (little to no desire for children together)",
                "Neutral/Uncertain (mixed feelings or undecided about having children)",
                "Interested (both partners want children and are planning for them)",
                "Strongly committed (both partners deeply desire children and actively planning)"
            ]
            children_sense_values = [20, 40, 60, 80, 95]

            selected_children_sense = st.selectbox(
                "The Sense of Having Children Together",
                children_sense_levels,
                index=3,
                key="sense_having_children"
            )
            st.session_state.shared_inputs[feature] = children_sense_values[children_sense_levels.index(selected_children_sense)]

        elif feature == "The Proportion of Common Genes":
            common_genes_levels = [
                "No genetic relation (completely different ethnic/family backgrounds)",
                "Distant relation (different ethnicities but some regional overlap)",
                "Moderate relation (same broad ethnic group or regional background)",
                "Close relation (same ethnic subgroup or similar family lineages)",
                "Very close relation (same community/tribal background or distant cousins)"
            ]
            common_genes_values = [50, 60, 70, 80, 75]

            selected_common_genes = st.selectbox(
                "The Proportion of Common Genes/Family Background",
                common_genes_levels,
                index=2,
                key="proportion_common_genes"
            )
            st.session_state.shared_inputs[feature] = common_genes_values[common_genes_levels.index(selected_common_genes)]

        elif feature == "Divorce in the Family of Grade 1":
            family_divorce_levels = [
                "Multiple divorces (both families have extensive divorce history)",
                "Frequent divorces (one or both families have several divorces)",
                "Some divorces (moderate divorce history in immediate families)",
                "Few divorces (minimal divorce history in immediate families)",
                "No divorces (both families have strong marriage traditions)"
            ]
            family_divorce_values = [20, 35, 50, 75, 90]

            selected_family_divorce = st.selectbox(
                "Divorce History in Immediate Families (parents, siblings)",
                family_divorce_levels,
                index=3,
                key="divorce_family_grade1"
            )
            st.session_state.shared_inputs[feature] = family_divorce_values[family_divorce_levels.index(selected_family_divorce)]

        elif feature == "Social Similarities":
            social_similarity_levels = [
                "Very different social backgrounds (completely different social circles)",
                "Somewhat different (different social groups but some overlap)",
                "Moderately similar (some shared social connections and backgrounds)",
                "Very similar (same social circles and similar backgrounds)",
                "Identical social backgrounds (grew up in same community/social group)"
            ]
            social_similarity_values = [20, 40, 60, 80, 95]

            selected_social_similarity = st.selectbox(
                "Social Similarities",
                social_similarity_levels,
                index=2,
                key="social_similarities"
            )
            st.session_state.shared_inputs[feature] = social_similarity_values[social_similarity_levels.index(selected_social_similarity)]

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
