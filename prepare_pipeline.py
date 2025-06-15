from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression

# nadpisanie Pipeline'u tak, by wypisywał zmiany klasyfikatorów
class PrintablePipeline(Pipeline):
 
    _global_last_clf = None

    def __init__(self, steps, **params):
        super().__init__(steps, **params)

    def set_params(self, **params):
        if 'clf' in params:
            new_cls = params['clf'].__class__
            if new_cls is not PrintablePipeline._global_last_clf:
                print(f"clf: {new_cls.__name__}")
                PrintablePipeline._global_last_clf = new_cls
        return super().set_params(**params)


def make_pipeline(scaler, k_features = 10):
    print("Preparing pipeline...")

    pipe = PrintablePipeline([
        ('scaler', scaler),
        ('pca',    PCA()),
        ('select', SelectKBest(k=k_features)),
        ('clf',    LogisticRegression(max_iter=5000))
    ])

    return pipe