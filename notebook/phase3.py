# ============================
# PHASE 3: PRICE PREDICTION MODEL
# ============================

# 1️⃣ Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer  # 👈 important

import joblib

# 2️⃣ Load data (use your actual file name here if different)
print("📂 Loading data...")
df = pd.read_csv(
    r"C:\Users\Radha\OneDrive\Pictures\Documents\DATA SCIENCE Airbnb project\data\AB_NYC_2019.csv"
)
print("Shape:", df.shape)
print(df.head())

# 3️⃣ Define target and features
print("\n🎯 Defining features (X) and target (y)...")

target_col = "price"
y = df[target_col]
X = df.drop(columns=[target_col])

# Drop useless ID/text columns if they exist
cols_to_drop = ["id", "name", "host_id", "host_name", "last_review"]
X = X.drop(columns=[c for c in cols_to_drop if c in X.columns])

print("Features shape:", X.shape)
print("Columns:", X.columns.tolist())

# 4️⃣ Train-test split
print("\n✂️ Train-test split...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train:", X_train.shape, " Test:", X_test.shape)

# 5️⃣ Separate numeric & categorical columns
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

print("\n🔢 Numeric features:", numeric_features)
print("🔤 Categorical features:", categorical_features)

# 6️⃣ Preprocessing pipelines WITH IMPUTATION
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 7️⃣ Define models to compare
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1,
    ),
}

# 8️⃣ Train, evaluate, and pick best model
print("\n🚀 Training models...")
results = {}
best_model_name = None
best_mae = np.inf
best_pipeline = None

for name, model in models.items():
    print("\n==============================")
    print(f"🔹 Training {name}...")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    # Older sklearn: no 'squared' argument → compute RMSE manually
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)

    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    print(f"{name} MAE:  {mae:.2f}")
    print(f"{name} RMSE: {rmse:.2f}")
    print(f"{name} R²:   {r2:.3f}")

    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_pipeline = pipe

print("\n✅ BEST MODEL:", best_model_name)
print("✅ BEST MAE:", best_mae)
print("\nAll results:", results)

# 9️⃣ Save best model as price_model.pkl
print("\n💾 Saving best model as price_model.pkl ...")
joblib.dump(best_pipeline, "price_model.pkl")
print("✅ Saved!")

# 🔟 Quick sanity check – reload and predict on one sample
print("\n🧪 Sanity check on one test sample...")

loaded_model = joblib.load("price_model.pkl")
sample = X_test.iloc[[0]]
true_price = y_test.iloc[0]
pred_price = loaded_model.predict(sample)[0]

print("True price:", true_price)
print("Predicted price:", round(pred_price, 2))
print("\nSample features used for prediction:")
print(sample)








































































































