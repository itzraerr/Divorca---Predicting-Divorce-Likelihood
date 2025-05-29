# This script trains a model to predict divorce probability based on various features.
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("data/divorce_data.csv")

# Convert Divorce Probability to binary label using median
median_threshold = df["Divorce Probability"].median()
df["Divorce Label"] = (df["Divorce Probability"] > median_threshold).astype(int)
df.drop(columns=["Divorce Probability"], inplace=True)

# Select top 15 features based on correlation with Divorce Label
correlations = df.corr()["Divorce Label"].abs().sort_values(ascending=False)
top_features = correlations.drop("Divorce Label").head(15).index.tolist()

# Define features and labels
X = df[top_features]
y = df["Divorce Label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save the trained model and features
with open("model/divorce_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/divorce_features.pkl", "wb") as f:
    pickle.dump(top_features, f)
