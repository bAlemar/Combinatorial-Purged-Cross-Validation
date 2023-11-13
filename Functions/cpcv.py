import numpy as np
from sklearn.model_selection import BaseCrossValidator
from itertools import combinations

class cpcv(BaseCrossValidator):
    def __init__(self, n_splits, n_test_splits,intervalo=15):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.intervalo = intervalo
    def _generate_selected_fold_bounds(self, fold_bounds):
        lista_folds_test = list(combinations(fold_bounds, self.n_test_splits))
        # selected_fold_bounds = selected_fold_bounds[:self.n_splits]  # Keep only the required number of splits
        return lista_folds_test
    def split(self, X, y=None, groups=None):
        n_samples = list(range(len(X)))
        # Geração dos folds
        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(n_samples, self.n_splits)]
        # Análise Combinatória dos Folds
        lista_folds_test = self._generate_selected_fold_bounds(fold_bounds)
        lista_folds_test.reverse() # Selected bounds
        self.n_splits = lista_folds_test #Para puxar numero splits certo
        for folds_teste in lista_folds_test:
            teste_indices = []
            train_indices = []
            folds_train = [fold_train for fold_train in fold_bounds if fold_train not in folds_teste ]
            for start_test,end_test in folds_teste:
                intervalo_fold_test = list(range(start_test,end_test))
                teste_indices.extend(intervalo_fold_test)
                for pos,(start_train,end_train) in enumerate(folds_train):        
                    # Caso 1:
                    if start_test - self.intervalo in range(start_train,end_train):
                        folds_train[pos] = (start_train, end_train - self.intervalo)
                    # Caso 2: 
                    if end_test + self.intervalo in range(start_train,end_train):
                        folds_train[pos] = (start_train + self.intervalo, end_train)
            for start_train, end_train in folds_train:
                intervalo_fold_train = list(range(start_train,end_train))
                train_indices.extend(intervalo_fold_train)
            
            yield train_indices, teste_indices
    
    def get_n_splits(self, X=None, y=None, groups=None):
        n_samples = len(X)
        fold_bounds = [(fold[0], fold[-1] + 1) for fold in np.array_split(list(range(n_samples)), self.n_splits)]
        selected_fold_bounds = self._generate_selected_fold_bounds(fold_bounds)
        return len(selected_fold_bounds)