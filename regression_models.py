import numpy as np
import pandas as pd
import json
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

df = pd.read_csv("dataset_trees.csv")


required_columns = [
    "bioma", 
    "precipitacao", 
    "clima", 
    "temperatura", 
    "tempo_abandono", 
    "altura_arvore"
]


X = df.drop(columns=["altura_arvore"])
y = df["altura_arvore"]

categorical_features = ["bioma", "clima"]
numeric_features = ["precipitacao", "temperatura", "tempo_abandono"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ]
)

models = {
    "SVM": SVR(kernel="rbf", C=10, epsilon=0.2),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=50000, random_state=42)
}


kf = KFold(
    n_splits=5, 
    shuffle=True, 
    random_state=21
)


results = {}


for name, model in models.items():
    mae_scores, rmse_scores = [], []

    print(f"\n Training: {name}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = root_mean_squared_error(y_val, y_pred)

        mae_scores.append(mae)
        rmse_scores.append(rmse)

        print(f"  Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}")


    results[name] = {
        "MAE_mean": np.mean(mae_scores),
        "MAE_std": np.std(mae_scores),
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores)
    }


with open("metrics.json", "w") as f:
    json.dump(results, f, indent=4)

print("End of treraining. Metrics saved.")
print(json.dumps(results, indent=4))
