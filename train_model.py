import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import cv2
import os

# Original training data
X_train = np.array([
    [240, 200, 180],  # Light skin tone (fair)
    [220, 180, 150],  # Light skin tone (fair)
    [150, 120, 100],  # Medium skin tone
    [180, 130, 100],  # Medium skin tone
    [100, 70, 60],    # Dark skin tone
    [70, 50, 40],     # Dark skin tone
    [250, 230, 220],  # Very fair skin tone
    [180, 150, 130],  # Olive skin tone
    [120, 80, 60],    # Tan skin tone
    [60, 40, 30],     # Very dark skin tone
])

y_train = np.array([
    "Pastels",         # Suitable for light/fair skin
    "Pastels",         # Suitable for light/fair skin
    "Earthy Tones",    # Suitable for medium skin
    "Earthy Tones",    # Suitable for medium skin
    "Warm Tones",      # Suitable for dark skin
    "Warm Tones",      # Suitable for dark skin
    "Neutrals",        # Suitable for very fair skin
    "Earthy Tones",    # Suitable for olive skin
    "Pastels",         # Suitable for tan skin
    "Warm Tones",      # Suitable for very dark skin
])

# Function to add noise and generate more unique data
def generate_more_data(X, y, num_new_samples=1000):
    new_X = []
    new_y = []
    for i in range(num_new_samples):
        idx = np.random.randint(len(X))
        base_X = X[idx]
        base_y = y[idx]
        noise = np.random.randint(-10, 10, size=(3,))
        new_X.append(base_X + noise)
        new_y.append(base_y)
    return np.array(new_X), np.array(new_y)

# Generate 1000 new samples
X_train_expanded, y_train_expanded = generate_more_data(X_train, y_train, num_new_samples=1000)

# Combine the original and new data
X_train_combined = np.vstack((X_train, X_train_expanded))
y_train_combined = np.hstack((y_train, y_train_expanded))

# Label Encoding for categorical labels
label_encoder = LabelEncoder()
y_train_combined_encoded = label_encoder.fit_transform(y_train_combined)

# Function to convert RGB to LAB color space
def rgb_to_lab(rgb):
    rgb = np.uint8([[rgb]])  # Convert to 2D array
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    return lab[0][0]  # Return Lab values

# Apply RGB to Lab conversion to the dataset
X_train_lab = np.array([rgb_to_lab(rgb) for rgb in X_train_combined])

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_train_lab, y_train_combined_encoded, test_size=0.2, random_state=42)

# Convert to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Set parameters for XGBoost
params = {
    'objective': 'multi:softmax',
    'num_class': len(np.unique(y_train)),  # Number of classes
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'merror'
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)

# Save the trained model
os.makedirs('ml', exist_ok=True)
joblib.dump(model, 'ml/recommendation_model.joblib')

print("Model trained and saved successfully!")