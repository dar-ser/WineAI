from prepare_data import load_data, X_y_split, train_val_test_split
from find_model import search_best_model
from finalize_model import finalize_model, evaluate_model
from plots import visualize_data
 
# --- przygotowanie danych ---
df = load_data('data\winequality-red.csv', 'data\winequality-white.csv')
X, y = X_y_split(df, y_col='type', drop_col=['quality'])

X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
    X, y, test_size=0.2, val_size=0.125, random=42
)

# --- wizualizacja ---
visualize_data(X_train) 

# --- wyszukiwanie najlepszego modelu ---
grid = search_best_model(X_train, y_train)

print(f"\nBest params: {grid.best_params_}")
print(f"\nValidation accuracy: {grid.score(X_val, y_val):.4f}\n")

best_model = finalize_model(grid, X_train, X_val, y_train, y_val, output_filepath='wine_model.pkl')

# --- ewaluacja na zbiorze testowym ---
metrics, report = evaluate_model(best_model, X_test, y_test)
print(f"Test accuracy: {metrics['accuracy']:.4f}")
if 'roc_auc' in metrics:
    print(f"Test ROC AUC:   {metrics['roc_auc']:.4f}")
print("Confusion matrix:\n", metrics['confusion_matrix'])
print(report)