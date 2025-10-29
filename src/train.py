import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


from src.data_prep import load_raw, basic_clean


CATEGORICAL = [
    "gender","Partner","Dependents","PhoneService","MultipleLines",
    "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies","Contract",
    "PaperlessBilling","PaymentMethod"
]

NUMERICAL = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]

def train_and_export(raw_csv, out_model, out_preds):
    # 1) Veri yükle & temel temizlik
    df = load_raw(raw_csv)
    df = basic_clean(df)


    df["TotalCharges"] = pd.to_numeric(df.get("TotalCharges"), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 2) X, y
    X = df[CATEGORICAL + NUMERICAL].copy()
    y = df["churn_true"]   # basic_clean bunu 0/1’e çeviriyor

    # 3) Ön işleme + model
    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERICAL),
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("preproc", preproc),
        ("clf", model)
    ])

    # 4) Eğitim
    pipe.fit(X, y)

    # 5) Kaydet
    joblib.dump(pipe, out_model)

    # 6) Tahmin & dışa aktarım
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    export = df.copy()
    export["churn_pred"] = pred
    export["churn_proba"] = proba
    export.to_csv(out_preds, index=False)

    # 7) Rapor
    print("ROC-AUC:", roc_auc_score(y, proba))
    print(classification_report(y, pred))
