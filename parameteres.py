from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans

 
def get_param_grid():
    print("Setting parameteres...")

    common = {
        'pca__n_components': [3,4,5],
    }

    param_grid = [
        # LogisticRegression: l1 penalties
        {
            **common,
            'clf': [LogisticRegression(max_iter=5000)],
            'clf__penalty': ['l1'],
            'clf__solver': ['liblinear', 'saga'],
            'clf__C': [0.001, 0.01, 0.1, 1, 10]
        },
        # LogisticRegression: l2 penalties
        {
            **common,
            'clf': [LogisticRegression(max_iter=5000)],
            'clf__penalty': ['l2'],
            'clf__solver': ['lbfgs', 'sag', 'saga', 'newton-cg'],
            'clf__C': [0.001, 0.01, 0.1, 1, 10]
        },
        # LogisticRegression: elasticnet
        {
            **common,
            'clf': [LogisticRegression(max_iter=5000)],
            'clf__penalty': ['elasticnet'],
            'clf__solver': ['saga'],
            'clf__C': [0.001, 0.01, 0.1, 1, 10],
            'clf__l1_ratio': [0.5, 0.75]
        },

        # K-Nearest Neighbors
        {
            **common,
            'clf': [KNeighborsClassifier()],
            'clf__n_neighbors': [3, 5, 7, 9],
            'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan', 'minkowski']
        },

        # Support Vector Machine:
        #  Linear kernel
        {
            **common,
            'clf': [SVC(kernel='linear')],
            'clf__C': [0.1, 1, 10, 100]
        },
        #  RBF kernel
        {
            **common,
            'clf': [SVC(kernel='rbf')],
            'clf__C': [0.1, 1, 10, 100],
            'clf__gamma': ['scale', 'auto']
        },
        #  Polynomial kernel
        {
            **common,
            'clf': [SVC(kernel='poly')],
            'clf__C': [0.1, 1, 10, 100],
            'clf__gamma': ['scale', 'auto'],
            'clf__degree': [2, 3]
        },

        # Decision Tree
        {
            **common,
            'clf': [DecisionTreeClassifier(random_state=42)],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [None, 5, 10, 20],
            'clf__min_samples_split': [2, 5, 10],
            'clf__min_samples_leaf': [1, 2, 4]
        },

        # Random Forest
        {
            **common,
            'clf': [RandomForestClassifier(random_state=42)],
            'clf__n_estimators': [100, 300],
            'clf__criterion': ['gini', 'entropy'],
            'clf__max_depth': [None, 10, 20],
            'clf__max_features': ['sqrt', 'log2', None]
        },

        # Gradient Boosting
        {
            **common,
            'clf': [GradientBoostingClassifier(random_state=42)],
            'clf__n_estimators': [100, 300],
            'clf__learning_rate': [0.01, 0.1, 0.2],
            'clf__max_depth': [3, 5, 7],
            'clf__subsample': [0.8],
            'clf__max_features': ['sqrt', None]
        },  
        # K-Means Clustering
        {
            **common,
            'clf': [KMeans(random_state=42)],
            'clf__n_clusters': [2, 3, 4, 5, 6],
            'clf__init': ['k-means++', 'random'],
            'clf__n_init': [10, 20]
        }
    ]

    return param_grid
