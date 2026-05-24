import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("data/housing.csv")

# -------------------------------
# Handle Missing Values
# -------------------------------

df = df.fillna(df.median(numeric_only=True))

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Feature Encoding
# -------------------------------

le = LabelEncoder()

for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# Features and Target
# -------------------------------

X = df.drop('SalePrice', axis=1)

y = df['SalePrice']

# -------------------------------
# Split Data
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------

predictions = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)

print("R2 Score:", r2)

# -------------------------------
# Show Predictions
# -------------------------------

result = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': predictions
})

print(result.head())
# Save model

pickle.dump(model, open("models/house_model.pkl", "wb"))

print("Model Saved Successfully!")