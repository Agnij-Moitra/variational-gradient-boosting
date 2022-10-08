
import numpy as np
import pandas as pd
from numba import prange
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, SGDRegressor, LassoLars, Lasso, Ridge, ARDRegression, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from concurrent.futures import ThreadPoolExecutor
import time
import math


class VGBRegressor(object):
    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        custom_loss=None,
        early_stopping: bool = False,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        complexity: bool = False,
        custom_models: list = None
    ):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.custom_loss = custom_loss
        self.early_stopping = early_stopping
        self._X = None
        self._y = None
        if custom_models:
            self._models = custom_models
        else:
            if complexity:
                self._models = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor, ExtraTreesRegressor,
                                ElasticNet, LassoLars, Lasso, GradientBoostingRegressor, HistGradientBoostingRegressor,
                                BaggingRegressor, SVR, NuSVR, XGBRegressor, XGBRFRegressor, SGDRegressor, KernelRidge, MLPRegressor,
                                Ridge, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor)
            else:
                self._models = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor,
                                ElasticNet, LassoLars, Lasso, SGDRegressor, BaggingRegressor,
                                Ridge, ARDRegression, RANSACRegressor, ExtraTreesRegressor)
        self._ensemble = []

    def _metrics(self, vt, vp, model, time=None):
        if self.custom_loss:
            return {'model': model, 'time': time, 'loss': self.custom_loss(vt, vp)}
        return {"model": model, "time": time, "loss": mean_absolute_error(vt, vp)}

    def _create_model(self, X, y, model_name, time_it: bool = False):
        model = Pipeline([
            ('scaler1', RobustScaler()),
            ('scaler2', MinMaxScaler()),
            ('model', model_name())
        ])
        if time_it:
            begin = time.perf_counter()
            model.fit(X, y)
            end = time.perf_counter()
            return (model, end - begin)
        return (model.fit(X, y), None)

    def _get_metrics(self, model_name):
        Xt, Xv, yt, yv = train_test_split(self._X, self._y)
        results = self._create_model(Xt, yt, model_name, time_it=False)
        model, time = results[0], results[1]
        return self._metrics(yv,
                             model.predict(Xv), model, time)

    def _get_results(self, X, y) -> list:
        results = []
        self._X = X
        self._y = y
        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            for i in executor.map(self._get_metrics, self._models):
                results.append(i)
        return results
        for i in self._models:
            try:
                results.append(self._get_metrics(X, y, i))
            except Exception:
                pass
        return results

    def fit(self, X_train, y_train):
        # base model: mean
        # computer residuals: y - y hat
        # for n_estimators: a) y = prev residuals && residuals * learning rate
        # add early stopping
        # restore best weights
        # ada boost and adaptive scaling for learning rates

        preds = pd.DataFrame(
            data={'p0': np.full((len(y_train)), y_train.mean(skipna=True))})
        residuals = pd.DataFrame(
            data={'r0': y_train - y_train.mean(skipna=True)})

        for i in prange(1, self.n_estimators + 1):
            y = residuals[f'r{i - 1}']
            results = self._get_results(X_train, y)
            min_loss = min(results, key=lambda x: x.get(
                "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
            min_model = [i['model']
                         for i in results if min_loss >= i['loss']][0]
            residuals[f'r{i}'] = y - min_model.predict(
                X_train) * self.learning_rate
            X_train[f'r{i}'] = y
            self._ensemble.append(min_model)
        return self._ensemble, residuals.mean()


"""


import numpy as np
import pandas as pd
from numba import prange
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, NuSVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge, ElasticNet, SGDRegressor, LassoLars, Lasso, Ridge, ARDRegression, RANSACRegressor, HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from xgboost import XGBRegressor, XGBRFRegressor
from time import perf_counter
from copy import deepcopy


class VGBRegressor(object):
    def __init__(
        self,
    ):
        self._ensemble = []

    def _metrics(self, vt, vp, model, time=None):
        if self.custom_loss:
            return {'model': model, 'time': time, 'loss': self.custom_loss(vt, vp)}
        return {"model": model, "time": time, "loss": mean_absolute_error(vt, vp)}

    def _create_model(self, X, y, model_name, time_it: bool = False):
        model = model_name()
        if time_it:
            begin = perf_counter()
            model.fit(X, y)
            end = perf_counter()
            return (model, end - begin)
        return (model.fit(X, y), None)

    def _get_metrics(self, model_name):
        try:
            Xt, Xv, yt, yv = train_test_split(self._X, self._y)
            results = self._create_model(Xt, yt, model_name, time_it=False)
            model, time = results[0], results[1]
            return self._metrics(yv,
                                 model.predict(Xv), model, time)
        except Exception:
            return

    def _get_results(self, X, y) -> list:
        results = []
        self._X = X
        self._y = y
        with ThreadPoolExecutor(max_workers=len(self._models)) as executor:
            for i in executor.map(self._get_metrics, self._models):
                print(i)
                if i:
                    results.append(i)
                    
        return results
        for i in self._models:
            try:
                results.append(self._get_metrics(X, y, i))
            except Exception:
                pass
        return results

    def fit(
        self, X_train, y_train,
        early_stopping: bool = False,
        early_stopping_min_delta: float = 0.001,
        early_stopping_patience: int = 10,
        custom_models: list = None,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        warm_start: bool = False,
        complexity: bool = False,
    ):
        if custom_models:
            self._models = custom_models
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        if custom_models:
            self._models = custom_models
        else:
            if complexity:
                self._models = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor, ExtraTreesRegressor,
                                ElasticNet, LassoLars, Lasso, GradientBoostingRegressor, HistGradientBoostingRegressor,
                                BaggingRegressor, NuSVR, XGBRegressor, XGBRFRegressor, SGDRegressor, KernelRidge, MLPRegressor,
                                Ridge, ARDRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor)
            else:
                self._models = (DecisionTreeRegressor, LinearRegression, BayesianRidge, KNeighborsRegressor,
                                ElasticNet, LassoLars, Lasso, SGDRegressor, BaggingRegressor,
                                Ridge, ARDRegression, RANSACRegressor, ExtraTreesRegressor)

        # base model: mean
        # computer residuals: y - y hat
        # for n_estimators: a) y = prev residuals && residuals * learning rate
        # add early stopping
        # restore best weights
        # ada boost and adaptive scaling for learning rates
        X_train = deepcopy(X_train)
        X_train = RobustScaler().fit_transform(X_train)
        X_train = MinMaxScaler().fit_transform(X_train)
        preds = pd.DataFrame(
            data={'yt': y_train, 'p0': np.full((len(y_train)), y_train.mean(skipna=True))})
        residuals = pd.DataFrame(
            data={'r0': y_train - y_train.mean(skipna=True)})
        #
        if not self.early_stopping:
            for i in prange(1, self.n_estimators + 1):
                if i == 1:
                    y = residuals['r0']
                    # return y
                # else:
                #     y = preds[f'p{i - 1}']
                results = self._get_results(X_train, y)
                return results
                min_loss = min(results, key=lambda x: x.get(
                    "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                min_model = [i['model']
                             for i in results if min_loss >= i['loss']][0]

                preds[f'p{i}'] = residuals.sum(axis=1) + min_model.predict(
                    X_train) * self.learning_rate
                residuals[f"r{i}"] = preds[f'yt'] - preds[f'p{i}']
                residuals_mean = residuals[f'r{i}'].mean()
                if warm_start:
                    X_train[f'r{i}'] = y
                self._ensemble.append(min_model)
        else:
            improve = []
            for i in prange(1, self.n_estimators + 1):
                y = residuals[f'r{i - 1}']
                results = self._get_results(X_train, y)
                min_loss = min(results, key=lambda x: x.get(
                    "loss", float('inf')))["loss"]  # https://stackoverflow.com/a/19619294
                min_model = [i['model']
                             for i in results if min_loss >= i['loss']][0]
                residuals[f'r{i}'] = y - min_model.predict(
                    X_train) * self.learning_rate
                res_mean = residuals[f'r{i}'].mean()
                improve.append((res_mean - y.mean()) / res_mean * 100)
                X_train[f'r{i}'] = y
                if self.early_stopping and i > len(improve):
                    # if improve[-1]
                    pass
                self._ensemble.append(min_model)
        residuals = residuals.mean()
        res_min = residuals.min()
        self._ensemble = self._ensemble[:[i for i in prange(
            len(residuals)) if residuals[i] == res_min][0]]
        return self._ensemble, residuals

"""