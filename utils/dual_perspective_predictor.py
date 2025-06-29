# Enhanced Dual-Perspective Prediction Class
import pandas as pd
import numpy as np

class DualPerspectivePredictor:
    def __init__(self, model_info, feature_list, feature_categories):
        # Handle both simple models and models with scalers
        if isinstance(model_info, tuple):
            self.base_model, self.scaler = model_info
            self.has_scaler = True
        else:
            self.base_model = model_info
            self.scaler = None
            self.has_scaler = False

        self.feature_list = feature_list
        self.individual_features = feature_categories['individual']
        self.shared_features = feature_categories['shared']
        self.perspective_dependent_features = feature_categories['perspective_dependent']
    
    def predict_from_perspective(self, man_inputs, woman_inputs, shared_inputs):
        """
        Make predictions from both male and female perspectives

        Args:
            man_inputs: dict with man's individual and perspective-dependent feature values
            woman_inputs: dict with woman's individual and perspective-dependent feature values
            shared_inputs: dict with shared relationship feature values

        Returns:
            dict with predictions from both perspectives
        """
        # Create feature vectors for both perspectives
        man_vector = self._create_feature_vector(man_inputs, shared_inputs, 'man')
        woman_vector = self._create_feature_vector(woman_inputs, shared_inputs, 'woman')

        # Convert to DataFrames with proper feature names to avoid warnings
        man_df = pd.DataFrame([man_vector], columns=self.feature_list)
        woman_df = pd.DataFrame([woman_vector], columns=self.feature_list)

        # Apply scaling if needed
        if self.has_scaler:
            man_scaled = self.scaler.transform(man_df)
            woman_scaled = self.scaler.transform(woman_df)

            # Get predictions with scaled data
            man_prediction = self.base_model.predict(man_scaled)[0]
            woman_prediction = self.base_model.predict(woman_scaled)[0]

            # Get prediction probabilities
            try:
                man_prob = self.base_model.predict_proba(man_scaled)[0][1]
                woman_prob = self.base_model.predict_proba(woman_scaled)[0][1]
            except:
                man_prob = float(man_prediction)
                woman_prob = float(woman_prediction)
        else:
            # Get predictions without scaling
            man_prediction = self.base_model.predict(man_df)[0]
            woman_prediction = self.base_model.predict(woman_df)[0]

            # Get prediction probabilities
            try:
                man_prob = self.base_model.predict_proba(man_df)[0][1]
                woman_prob = self.base_model.predict_proba(woman_df)[0][1]
            except:
                man_prob = float(man_prediction)
                woman_prob = float(woman_prediction)

        return {
            'man_prediction': int(man_prediction),
            'woman_prediction': int(woman_prediction),
            'man_probability': float(man_prob),
            'woman_probability': float(woman_prob),
            'combined_risk': float((man_prob + woman_prob) / 2),
            'confidence_score': float(abs(man_prob - 0.5) + abs(woman_prob - 0.5)) / 2  # Higher when both are confident
        }
    
    def _create_feature_vector(self, individual_inputs, shared_inputs, perspective):
        """Create feature vector for a specific perspective"""
        vector = []

        # Combine all inputs for easier access
        all_inputs = {**individual_inputs, **shared_inputs}

        for feature in self.feature_list:
            if feature in all_inputs:
                vector.append(all_inputs[feature])
            elif feature in self.individual_features:
                # Use individual inputs for this perspective
                vector.append(individual_inputs.get(feature, 50.0))
            elif feature in self.shared_features:
                # Use shared inputs
                vector.append(shared_inputs.get(feature, 50.0))
            elif feature in self.perspective_dependent_features:
                # Use perspective-specific inputs
                vector.append(individual_inputs.get(feature, 50.0))
            else:
                # Calculate enhanced features if they're missing
                if feature == 'Psych_Compatibility':
                    love = all_inputs.get('Love', 50)
                    commitment = all_inputs.get('Commitment', 50)
                    loyalty = all_inputs.get('Loyalty', 50)
                    vector.append((love + commitment + loyalty) / 3)
                elif feature == 'Economic_Stability':
                    education = all_inputs.get('Education', 50)
                    income = all_inputs.get('Good Income', 50)
                    vector.append((education + income) / 2)
                elif feature == 'Family_Harmony':
                    family_rel = all_inputs.get('Relationship with the Spouse Family', 50)
                    vector.append(family_rel)
                elif feature == 'Social_Compatibility':
                    social = all_inputs.get('Social Similarities', 50)
                    cultural = all_inputs.get('Cultural Similarities', 50)
                    religion = all_inputs.get('Religion Compatibility', 50)
                    vector.append((social + cultural + religion) / 3)
                elif feature == 'Risk_Factors':
                    addiction = all_inputs.get('Addiction', 0)
                    age_gap = all_inputs.get('Age Gap', 0)
                    vector.append((addiction + age_gap * 2) / 3)
                elif feature == 'Perfect_Match':
                    # Calculate perfect match indicator
                    psych = (all_inputs.get('Love', 50) + all_inputs.get('Commitment', 50) + all_inputs.get('Loyalty', 50)) / 3
                    econ = (all_inputs.get('Education', 50) + all_inputs.get('Good Income', 50)) / 2
                    family = all_inputs.get('Relationship with the Spouse Family', 50)
                    risk = (all_inputs.get('Addiction', 0) + all_inputs.get('Age Gap', 0) * 2) / 3
                    perfect = 100 if (psych > 80 and econ > 70 and family > 75 and risk < 20) else 0
                    vector.append(perfect)
                elif feature == 'High_Risk':
                    # Calculate high risk indicator
                    psych = (all_inputs.get('Love', 50) + all_inputs.get('Commitment', 50) + all_inputs.get('Loyalty', 50)) / 3
                    family = all_inputs.get('Relationship with the Spouse Family', 50)
                    risk = (all_inputs.get('Addiction', 0) + all_inputs.get('Age Gap', 0) * 2) / 3
                    high_risk = 100 if (psych < 40 or risk > 60 or family < 30) else 0
                    vector.append(high_risk)
                else:
                    # Default value if feature not categorized
                    vector.append(50.0)

        return vector
