import joblib
from prepare_data import load_data, X_y_split, train_val_test_split
from finalize_model import evaluate_model


# Ponowne wczytanie danych i podział
df = load_data('data\winequality-red.csv', 'data\winequality-white.csv')
X, y = X_y_split(df, y_col='type', drop_col=['quality'])
_, _, X_test, _, _, y_test = train_val_test_split(
    X, y, test_size=0.2, val_size=0.125, random=123
)

# Załadowanie trenującego modelu
best_model = joblib.load('wine_model.pkl')


# Ewaluacja
metrics, report = evaluate_model(best_model, X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
if 'roc_auc' in metrics:
    print(f"Test ROC AUC:   {metrics['roc_auc']:.4f}")
print("Confusion matrix:\n", metrics['confusion_matrix'])
print(report)