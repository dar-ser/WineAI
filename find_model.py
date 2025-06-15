import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from prepare_pipeline import make_pipeline
from parameteres import get_param_grid
 
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def search_best_model(X_train, y_train,  scoring='accuracy'):
    print("Finding best model...")

    scaler = MinMaxScaler()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe       = make_pipeline(scaler)
    param_grid = get_param_grid()
    grid = GridSearchCV(
        estimator = pipe,
        param_grid = param_grid,
        cv = cv,
        scoring = scoring,
        n_jobs = 1,
        verbose = 0,
    )
    
    print("\nConsidering:")
    grid.fit(X_train, y_train)
    print()

    return grid


def save_model(grid_search, filename='best_wine_quality_model.pkl'):
    print("Saving best model...")
    joblib.dump(grid_search.best_estimator_, filename)
