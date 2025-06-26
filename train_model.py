#!/usr/bin/env python3
"""
Train the divorce prediction model from the root directory
This script can be run from the project root and will work when deployed
"""

# Dual-Perspective Divorce Prediction System
# This script creates a model that can predict divorce from both man's and woman's perspectives
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import the dual perspective predictor from utils
from utils.dual_perspective_predictor import DualPerspectivePredictor

print("=== Dual-Perspective Divorce Prediction Model Training ===")

# Load the dataset
df = pd.read_csv("data/divorce_data.csv")
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Convert Divorce Probability to binary label using median
median_threshold = df["Divorce Probability"].median()
df["Divorce Label"] = (df["Divorce Probability"] > median_threshold).astype(int)
print(f"Divorce threshold (median): {median_threshold:.3f}")
print(f"High risk cases: {df['Divorce Label'].sum()}, Low risk cases: {(df['Divorce Label'] == 0).sum()}")
df.drop(columns=["Divorce Probability"], inplace=True)

# Categorize features by perspective type for dual-perspective modeling
# Individual features: Can be different for man vs woman
individual_features = [
    "Education", "Mental Health", "Self Confidence", "Good Income",
    "Addiction", "Independency", "Start Socializing with the Opposite Sex Age "
]

# Shared features: Same for both partners (relationship-level)
shared_features = [
    "Age Gap", "Economic Similarity", "Social Similarities", "Cultural Similarities",
    "Common Interests", "Religion Compatibility", "No of Children from Previous Marriage",
    "Engagement Time", "Commitment", "The Sense of Having Children",
    "The Proportion of Common Genes", "Divorce in the Family of Grade 1"
]

# Perspective-dependent features: Interpreted differently by each partner
perspective_dependent_features = [
    "Social Gap", "Desire to Marry", "Relationship with the Spouse Family",
    "Loyalty", "Relation with Non-spouse Before Marriage",
    "Spouse Confirmed by Family", "Love"
]

# Get all available features (some might not exist in the dataset)
all_columns = df.columns.tolist()
all_columns.remove("Divorce Label")

# Filter features that actually exist in the dataset
individual_features = [f for f in individual_features if f in all_columns]
shared_features = [f for f in shared_features if f in all_columns]
perspective_dependent_features = [f for f in perspective_dependent_features if f in all_columns]

print("\n=== Feature Categorization ===")
print("Individual features (can differ between partners):", individual_features)
print("Shared features (same for both partners):", shared_features)
print("Perspective-dependent features (interpreted differently):", perspective_dependent_features)

# Create feature sets for different modeling approaches
all_features = individual_features + shared_features + perspective_dependent_features

# Select top features based on correlation with Divorce Label
correlations = df.corr()["Divorce Label"].abs().sort_values(ascending=False)
top_features = correlations.drop("Divorce Label").head(15).index.tolist()

print(f"\n=== Top {len(top_features)} Features by Correlation ===")
for i, feature in enumerate(top_features, 1):
    print(f"{i:2d}. {feature}: {correlations[feature]:.3f}")

# Define features and labels for main model
X = df[top_features]
y = df["Divorce Label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n=== Model Training ===")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train the main Random Forest Classifier
main_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
main_model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = main_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nMain Model Accuracy: {accuracy:.3f}")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': top_features,
    'importance': main_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Feature Importance ===")
for idx, row in feature_importance.head(10).iterrows():
    print(f"{row['feature']}: {row['importance']:.3f}")

# Create the dual-perspective predictor using the imported class
dual_predictor = DualPerspectivePredictor(
    main_model,
    top_features,
    {
        'individual': individual_features,
        'shared': shared_features,
        'perspective_dependent': perspective_dependent_features
    }
)

print("\n=== Saving Models and Configuration ===")

# Save the dual-perspective predictor
with open("model/divorce_model.pkl", "wb") as f:
    pickle.dump(dual_predictor, f)
print("SUCCESS: Dual-perspective model saved")

with open("model/divorce_features.pkl", "wb") as f:
    pickle.dump(top_features, f)
print("SUCCESS: Feature list saved")

# Create dual-perspective feature mapping for the app
dual_perspective_mapping = {
    'man_individual': individual_features,
    'woman_individual': individual_features,  # Same features but for woman
    'shared': shared_features,
    'man_perspective': perspective_dependent_features,
    'woman_perspective': perspective_dependent_features,  # Same features but from woman's view
    'all_features': top_features
}

# Save feature categorization for the dual-perspective app
feature_categories = {
    'individual': individual_features,
    'shared': shared_features,
    'perspective_dependent': perspective_dependent_features,
    'top_features': top_features,
    'dual_perspective_mapping': dual_perspective_mapping
}

with open("model/feature_categories.pkl", "wb") as f:
    pickle.dump(feature_categories, f)
print("SUCCESS: Feature categories saved")

# Save a configuration for the dual-perspective system
dual_config = {
    'model_accuracy': accuracy,
    'feature_importance': feature_importance.to_dict('records'),
    'threshold': median_threshold,
    'total_features': len(top_features),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open("model/dual_perspective_config.pkl", "wb") as f:
    pickle.dump(dual_config, f)
print("SUCCESS: Dual-perspective configuration saved")

print("\n=== Training Complete ===")
print("Files created:")
print("- model/divorce_model.pkl (Main prediction model)")
print("- model/divorce_features.pkl (Selected features)")
print("- model/feature_categories.pkl (Feature categorization)")
print("- model/dual_perspective_config.pkl (System configuration)")
print(f"\nModel ready for dual-perspective predictions with {accuracy:.1%} accuracy!")
