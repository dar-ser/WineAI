import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

 
def finalize_model(grid_search, X_train, X_val, y_train, y_val,
                   output_filepath='wine_model.pkl'):
    """
    - robimy refit na X\X_test
    - zapisujemy i zwracamy model
    """

    print("Fitting best model...")
    best_model = grid_search.best_estimator_
    best_model.fit(
        pd.concat([X_train, X_val]),
        pd.concat([y_train, y_val])
    )
    joblib.dump(best_model, output_filepath)
    return best_model


def evaluate_model(model, X_test, y_test):
    """
    - przewidujemy wyniki dla X_test
    - zwracamy dokładność i raport klasyfikacji
    """
    print("Evaluating test data...")

    y_pred = model.predict(X_test)
    metrics = {'accuracy': accuracy_score(y_test, y_pred)}
    
    try:    # jeżeli model obsługuje predict_proba, dodaj AUC
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except Exception:
        pass

    # macierz pomyłek i raport
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return metrics, report

