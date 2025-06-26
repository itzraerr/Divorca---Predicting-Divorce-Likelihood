# 💔 Divorca - Dual-Perspective Divorce Prediction System

A comprehensive machine learning system that predicts divorce likelihood from both the man's and woman's perspectives using relationship and personal characteristics.

## 🌟 Features

- **Dual-Perspective Analysis**: Separate predictions for both partners
- **Multi-Screen Interface**: Organized, user-friendly navigation with 6 screens
- **Comprehensive Assessment**: 25+ relationship and personal factors
- **Research-Based**: Categories based on relationship psychology research
- **Interactive Web App**: Built with Streamlit for easy use

## 🏗️ Project Structure

```
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── run_app.py               # Main entry point to run the app
├── train_model.py           # Entry point to train the model
├── app/                     # Main application
│   ├── __init__.py
│   ├── divorce_app.py       # Streamlit web application
│   └── utils/               # Utility modules
│       ├── __init__.py
│       └── dual_perspective_predictor.py
├── src/                     # Source code for training
│   ├── __init__.py
│   └── train_divorce_model.py
├── data/                    # Dataset
│   └── divorce_data.csv
└── model/                   # Trained model files
    ├── divorce_features.pkl
    ├── divorce_model.pkl
    └── feature_categories.pkl
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd divorca-divorce-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

**Option 1: Using the entry point script (Recommended)**
```bash
python run_app.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run app/divorce_app.py
```

The application will open in your web browser at `http://localhost:8501`

### Training the Model

To retrain the model with new data:

**Option 1: Using the entry point script (Recommended)**
```bash
python train_model.py
```

**Option 2: Direct training script**
```bash
python src/train_divorce_model.py
```

## 📱 Application Screens

The application is organized into 6 intuitive screens:

1. **👨 Man's Personal Characteristics** - Individual traits and characteristics
2. **👨 Man's Relationship Perspective** - His view on relationship factors
3. **👩 Woman's Personal Characteristics** - Individual traits and characteristics
4. **👩 Woman's Relationship Perspective** - Her view on relationship factors
5. **💑 Shared Relationship Factors** - Common relationship characteristics
6. **📊 Prediction Results** - Dual-perspective analysis and results

## 🔍 Assessment Categories

### Individual Features (7 categories)
- Education Level (7 levels)
- Mental Health (3 categories)
- Self Confidence (0-5 star rating)
- Income Level (4 categories)
- Addiction Level (4 categories)
- Independence Level (5 categories)
- Start Socializing Age (6 categories)

### Perspective-Dependent Features (7 categories)
- Social Gap (5 levels)
- Desire to Marry (5 levels)
- Relationship with Spouse Family (5 levels)
- Loyalty (5 levels)
- Relation with Non-spouse Before Marriage (5 levels)
- Spouse Confirmed by Family (5 levels)
- Love (5 levels) - **Gender-specific assessment**

### Shared Relationship Features (12 categories)
- Age Gap (0-50 years)
- Economic Similarity (4 categories)
- Social Similarities (5 categories)
- Cultural Similarities (5 categories)
- Common Interests (5 categories)
- Religion Compatibility (5 categories)
- Number of Children from Previous Marriage (1-5)
- Engagement Time (5 categories)
- Commitment (5 categories)
- The Sense of Having Children (5 categories)
- The Proportion of Common Genes (5 categories)
- Divorce in Family of Grade 1 (5 categories)

## 🧠 Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 25+ carefully selected relationship and personal factors
- **Approach**: Dual-perspective prediction system
- **Accuracy**: Trained on relationship psychology research data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole basis for making important life decisions. Professional counseling is recommended for relationship issues.
