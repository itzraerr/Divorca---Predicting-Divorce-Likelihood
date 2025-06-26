# Dual-Perspective Prediction Class
class DualPerspectivePredictor:
    def __init__(self, base_model, feature_list, feature_categories):
        self.base_model = base_model
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
        
        # Get predictions
        man_prediction = self.base_model.predict([man_vector])[0]
        woman_prediction = self.base_model.predict([woman_vector])[0]
        
        # Get prediction probabilities if available
        try:
            man_prob = self.base_model.predict_proba([man_vector])[0][1]
            woman_prob = self.base_model.predict_proba([woman_vector])[0][1]
        except:
            man_prob = man_prediction
            woman_prob = woman_prediction
        
        return {
            'man_prediction': man_prediction,
            'woman_prediction': woman_prediction,
            'man_probability': man_prob,
            'woman_probability': woman_prob,
            'combined_risk': (man_prob + woman_prob) / 2
        }
    
    def _create_feature_vector(self, individual_inputs, shared_inputs, perspective):
        """Create feature vector for a specific perspective"""
        vector = []
        
        for feature in self.feature_list:
            if feature in self.individual_features:
                # Use individual inputs for this perspective
                vector.append(individual_inputs.get(feature, 50.0))
            elif feature in self.shared_features:
                # Use shared inputs
                vector.append(shared_inputs.get(feature, 50.0))
            elif feature in self.perspective_dependent_features:
                # Use perspective-specific inputs
                vector.append(individual_inputs.get(feature, 50.0))
            else:
                # Default value if feature not categorized
                vector.append(50.0)
        
        return vector
