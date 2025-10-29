import pandas as pd
from sklearn.model_selection import train_test_split

CATEGORICAL = [
    "gender","SeniorCitizen","Partner","Dependents","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod"
]

NUMERICAL = ["tenure","MonthlyCharges","TotalCharges"]

TARGET = "Churn"  # expects "Yes"/"No"

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # standardize column names
    df.columns = [c.strip().replace(" ", "").replace("-", "") for c in df.columns]
    # handle TotalCharges numeric conversion if string
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # drop duplicates
    df = df.drop_duplicates()
    # strip spaces
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    # map target to 0/1
    if TARGET in df.columns:
        df["churn_true"] = (df[TARGET].str.lower().isin(["yes","1","true"])).astype(int)
    return df

def split(df: pd.DataFrame, test_size: float=0.2, random_state: int=42):
    X = df.drop(columns=["churn_true", TARGET], errors="ignore")
    y = df["churn_true"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
