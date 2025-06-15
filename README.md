# Projekt - Klasyfikacja typu wina

Ten projekt służy do klasyfikacji typu wina (czerwone vs białe) na podstawie jego chemicznych składników. 
Działanie projektu polega na przygotowaniu danych, strojeniu hiperparametrów, dopasowaniu najlepszego modelu i jego ewaluacji.



## Dane źródłowe

Zestawy danych pochodzą z repozytorium UCI Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)



## Struktura katalogów

```
root/
├── data/
│   ├── winequality-red.csv       # Dane dla wina czerwonego
│   └── winequality-white.csv     # Dane dla wina białego
├── pictures/
│   ├── correlation.png           # Heatmapa korelacji
│   └── pca.png                   # Wykres krzywej PCA
│   └── best_classifier.png       # Najlepszy znaleziony model
│   └── separate_test_output.png  # Wynik dokładności najlepszego modelu dla innego zbioru testowego
├── prepare_data.py               # Ładowanie i podział danych
├── prepare_pipeline.py           # Definicja pipeline’ów (skalowanie, PCA, selekcja, klasyfikator)
├── find_model.py                 # Przeszukiwanie hyperparametrów (GridSearchCV)
├── plots.py                      # Funkcje do wizualizacji danych (korelacje, PCA)
├── finalize_model.py             # Finałowe dopasowanie modelu i ewaluacja
├── train_model.py                # Główny skrypt treningowy
├── test_model.py                 # Skrypt testujący model na innym zbiorze testowym
├── wine_model.pkl                # Zapisany gotowy model
└── README.md                     # Dokumentacja projektu  
```


## Użyte wersje

* Python 3.11.9 
* Inne pakiety:
  * joblib                       1.5.1
  * matplotlib                   3.10.3
  * numpy                        2.1.3
  * pandas                       2.2.3
  * scikit-learn                 1.5.2



## Użycie

### 1. Trenowanie modelu

Uruchom skrypt `train_model.py`, który:

1. Ładuje dane (obydwa pliki CSV),
2. Dzieli na zbiory treningowy, walidacyjny i testowy, (70:10:20)
3. Wykonuje wizualizację (korelacje, PCA),
4. Przeszukuje hiperparametry przy pomocy `GridSearchCV`,
5. Finalnie trenuje model na zbiorze treningowym+walidacyjnym i zapisuje go do `wine_model.pkl`.

### 2. Testowanie modelu

Po zakończonym trenowaniu uruchomienie pliku `test_model.py` sprawdzi dokładność modelu na nowo wylosowanym zbiorze testowym. 

## Opis modułów

* **prepare\_data.py**

  * `load_data(red_path, white_path)` – ładuje i łączy dane, dodaje kolumnę `type` (1=czerwone, 0=białe) i usuwa kolumnę `quality`.
  * `X_y_split(df, y_col, drop_col)` – rozdziela cechy i etykietę.
  * `train_val_test_split(...)` – dzieli dane na train/val/test.

* **plots.py**

  * `visualize_data(X_train, X_val, X_test)` – rysuje heatmapę korelacji i wykres PCA na zbiorze treningowym.

* **prepare\_pipeline.py**

  * `PrintablePipeline` – Pipeline wypisujący nazwę klasyfikatora przy każdej zmianie, umożliwiający śledzenie procesu GridSearch'a.
  * `make_pipeline(scaler, k_features, n_pca)` – buduje pipeline.

* **find\_model.py**

  * `search_best_model(X_train, y_train, ...)` – GridSearchCV nad pipeline’em i zadanym grid’em hyperparametrów.

* **finalize\_model.py**

  * `finalize_model(...)` – refit na train+val i zapis modelu.
  * `evaluate_model(model, X_test, y_test)` – zwraca słownik metryk (`accuracy`, `roc_auc`, `confusion_matrix`) oraz raport.

* **train\_model.py** – spaja wszystkie etapy: data prep → wizualizacje → search → finalize.

* **test\_model.py** – ocenia finalny model na trzymanym zbiorze testowym.

