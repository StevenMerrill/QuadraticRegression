"""data object with fitting and plotting as well as a variable object"""
import numpy as np
from EstimatorLinearModel import Estimator
import matplotlib.pyplot as plt

class variableName:
    """stores variable data for use in data object"""
    name: str = None
    symbol: str = None
    unit: str = None

    def __init__(self, name, symbol, unit):
        self.name = name
        self.symbol = symbol
        self.unit = unit

    def axisLabel(self):
        """generates a axis label for the particular variable"""
        return self.name + " (" + self.unit + ")"

class data:
    """object to store data and do the fit and plot operations"""
    X_name: variableName = None
    y_name: variableName = None
    X: np.ndarray = None
    y: np.ndarray = None
    X_true: np.ndarray = None
    y_true: np.ndarray = None
    is_fitted: bool = False
    sample: int = None
    low: float = None
    high: float = None
    X_est: np.ndarray = None
    y_est: np.ndarray = None
    est: Estimator = None

    def __init__(self, X_name, y_name, X, y, X_true=None, y_true=None):
        self.X_name = X_name
        self.y_name = y_name
        self.X = X
        self.y = y
        self.X_true = X_true
        self.y_true = y_true
        self.sample = self.X.size
        self.low = self.X.min()
        self.high = self.X.max()

    def fit_and_plot(self):
        """runs a fit and a plot command with an estimator object"""
        self._fit()
        self._plot()

    def fit_for_tk(self):
        self._fit()
        return self._for_tk()
    
    def get_detail_string(self):
        return f"The data fits the equation: {self.get_eqn_string()} with a R2 Score of {self.get_score().round(2)}"
    
    def get_eqn_string(self):
        coefficients = self.est.get_weight() 
        return f"{self.y_name.symbol} = {np.round(coefficients["a"],2)} {self.X_name.symbol}^2 + {np.round(coefficients["b"],2)} {self.X_name.symbol} + {np.round(coefficients["c"],2)}"
    
    def get_score(self):
        return self.est.score(self.X,self.y)

    def _fit(self):
        """fits data to a scikit estimator object"""
        self.est = Estimator()
        self.est.fit(self.X,self.y)
        self.X_est = np.linspace(self.low,self.high,self.sample).reshape(self.sample,1)
        self.y_est = self.est.predict(self.X_est)
        self.is_fitted = True

    def _plot(self):
        """runs plot functions depending on config"""
        if (self.X_true is None or self.y_true is None):
            self._plot_no_true()
        else:
            self._plot_true()

    def _plot_true(self):
        """plots data if true value is available"""
        plt.scatter(self.X,self.y,color="black")
        plt.plot(self.X_est,self.y_est,color = "red")
        plt.plot(self.X_true,self.y_true,color = "blue")
        title = self.y_name.name + " versus" + self.X_name.name
        plt.title(title)
        plt.xlabel(self.X_name.axisLabel())
        plt.ylabel(self.y_name.axisLabel())
        plt.show()

    def _plot_no_true(self):
        """plots data if true value is not available"""
        plt.scatter(self.X,self.y,color="black")
        plt.plot(self.X_est,self.y_est,color = "red")
        title = self.y_name.name + " versus " + self.X_name.name
        plt.title(title)
        plt.xlabel(self.X_name.axisLabel())
        plt.ylabel(self.y_name.axisLabel())
        plt.show()

    def _for_tk(self):
        """runs tk functions depending on config"""
        if (self.X_true is None or self.y_true is None):
            return self._for_tk_no_true()
        else:
            return self._for_tk_true()

    def _for_tk_true(self):
        """returns a tk figure with appropriate data"""
        fig = plt.figure(figsize=(5,4),dpi=100)
        plot = fig.add_subplot(111)
        plot.scatter(self.X,self.y,color="black")
        plot.plot(self.X_est,self.y_est,color = "red")
        plot.plot(self.X_true,self.y_true,color = "blue")
        title = self.X_name.name + " versus" + self.y_name.name
        plot.set_title(title)
        plot.set_xlabel(self.X_name.axisLabel())
        plot.set_ylabel(self.y_name.axisLabel())
        return fig

    def _for_tk_no_true(self):
        """returns a tk figure with appropriate data"""
        fig = plt.figure(figsize=(5,4),dpi=100)
        plot = fig.add_subplot(111)
        plot.scatter(self.X,self.y,color="black")
        plot.plot(self.X_est,self.y_est,color = "red")
        title = self.X_name.name + " versus " + self.y_name.name
        plot.set_title(title)
        plot.set_xlabel(self.X_name.axisLabel())
        plot.set_ylabel(self.y_name.axisLabel())
        return fig


