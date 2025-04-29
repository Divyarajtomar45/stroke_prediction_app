import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data for stroke prediction
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    
    # Define features and their distributions
    age = np.random.normal(loc=55, scale=15, size=n_samples).clip(18, 100)
    gender = np.random.choice(['Male', 'Female'], size=n_samples)
    hypertension = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    heart_disease = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # BMI tends to be higher for those with hypertension
    bmi_base = np.random.normal(loc=25, scale=5, size=n_samples)
    bmi = bmi_base + hypertension * 3 + np.random.normal(scale=1, size=n_samples)
    bmi = bmi.clip(15, 50)
    
    # Smoking status
    smoking_status = np.random.choice(['never smoked', 'formerly smoked', 'smokes'], 
                                     size=n_samples, p=[0.5, 0.3, 0.2])
    
    # Glucose level, higher for those with heart disease
    avg_glucose_level = np.random.normal(loc=90, scale=20, size=n_samples) + heart_disease * 30
    avg_glucose_level = avg_glucose_level.clip(50, 300)
    
    # Calculate stroke risk based on features
    stroke_prob = (
        0.01 +  # base probability
        0.001 * (age - 50) +  # age factor
        0.1 * hypertension +  # hypertension factor
        0.15 * heart_disease +  # heart disease factor
        0.001 * (avg_glucose_level - 90) +  # glucose factor
        0.01 * (bmi - 25)  # BMI factor
    )
    
    # Add more weight for combinations of risk factors
    stroke_prob += 0.1 * (hypertension * heart_disease)
    stroke_prob += 0.005 * (age > 70) * heart_disease
    stroke_prob += 0.005 * (smoking_status == 'smokes') * (age > 60)
    
    # Clip probabilities between 0 and 0.7 (maximum stroke risk)
    stroke_prob = np.clip(stroke_prob, 0, 0.7)
    
    # Generate stroke labels based on probabilities
    stroke = np.random.binomial(n=1, p=stroke_prob)
    
    # Create a DataFrame
    data = pd.DataFrame({
        'age': age,
        'gender': gender,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'avg_glucose_level': avg_glucose_level,
        'smoking_status': smoking_status,
        'stroke': stroke
    })
    
    return data

# Generate dataset
stroke_data = generate_synthetic_data(2000)

# Basic data exploration
print("Dataset shape:", stroke_data.shape)
print("\nStroke distribution:")
print(stroke_data['stroke'].value_counts(normalize=True) * 100)

# Visual exploration of key factors
plt.figure(figsize=(15, 10))

# Age distribution by stroke
plt.subplot(2, 3, 1)
sns.histplot(data=stroke_data, x='age', hue='stroke', bins=20, kde=True)
plt.title('Age Distribution by Stroke')

# BMI distribution by stroke
plt.subplot(2, 3, 2)
sns.histplot(data=stroke_data, x='bmi', hue='stroke', bins=20, kde=True)
plt.title('BMI Distribution by Stroke')

# Glucose level distribution by stroke
plt.subplot(2, 3, 3)
sns.histplot(data=stroke_data, x='avg_glucose_level', hue='stroke', bins=20, kde=True)
plt.title('Glucose Level Distribution by Stroke')

# Hypertension counts by stroke
plt.subplot(2, 3, 4)
sns.countplot(data=stroke_data, x='hypertension', hue='stroke')
plt.title('Hypertension Counts by Stroke')

# Heart disease counts by stroke
plt.subplot(2, 3, 5)
sns.countplot(data=stroke_data, x='heart_disease', hue='stroke')
plt.title('Heart Disease Counts by Stroke')

# Smoking status counts by stroke
plt.subplot(2, 3, 6)
sns.countplot(data=stroke_data, x='smoking_status', hue='stroke')
plt.xticks(rotation=45)
plt.title('Smoking Status Counts by Stroke')

plt.tight_layout()
plt.savefig('stroke_exploratory_analysis.png')

# Prepare features and target
X = stroke_data.drop('stroke', axis=1)
y = stroke_data['stroke']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define categorical and numerical features
categorical_features = ['gender', 'smoking_status']
numerical_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'avg_glucose_level']

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ])

# Create a pipeline with preprocessor and classifier
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')

# Get feature importances
def get_feature_importance(model, categorical_features, numerical_features):
    # Get feature names after preprocessing
    ohe = model.named_steps['preprocessor'].named_transformers_['cat']
    cat_columns = ohe.get_feature_names_out(categorical_features)
    
    # Combine all feature names
    feature_names = list(numerical_features) + list(cat_columns)
    
    # Get feature importances
    importances = model.named_steps['classifier'].feature_importances_
    
    # Create dataframe of feature importances
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    return feature_importance.sort_values('Importance', ascending=False)

# Display feature importances
feature_imp = get_feature_importance(model, categorical_features, numerical_features)
print("\nFeature Importances:")
print(feature_imp)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_imp)
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')

# Save the model
with open('stroke_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel saved as 'stroke_prediction_model.pkl'")