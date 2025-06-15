import pandas as pd
from sklearn.model_selection import train_test_split
 
def load_data(red_path: str, white_path: str):
    print("Loading data...")

    red   = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')
    red['type'], white['type'] = 1, 0

    df = pd.concat([red, white], ignore_index=True)

    # upewniamy się, że dane są liczbowe
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna().reset_index(drop=True)

    return df


def X_y_split(df, y_col: str, drop_col=None):
    print("Finding target values...")

    if drop_col is None:
        drop_col = []
    X = df.drop(columns=[y_col] + drop_col)
    y = df[y_col]
    return X, y


def train_val_test_split(X, y,
                         test_size=0.2,
                         val_size=0.125, random=42):
    print("Dividing into train-val-test...")
    

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=random
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
