# ***** Calculations *****
import numpy as np
from sklearn.metrics import mean_squared_error

class Vld_Mtc:
    formulas = {
        "NSE"               : lambda obs, sim: 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)),
        "RMSE"              : lambda obs, sim: np.sqrt(mean_squared_error(obs, sim)),
        "MAE"               : lambda obs, sim: np.mean(np.abs(obs - sim)),
        "Correlation"       : lambda obs, sim: np.corrcoef(obs, sim)[0, 1],  # Pearson correlation coefficient
        "Bias Ratio"        : lambda obs, sim: np.mean(sim) / np.mean(obs),  # β = mean(sim) / mean(obs)
        "Variability Ratio" : lambda obs, sim: (np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs)),  # γ'
        "KGE"               : lambda obs, sim: 1 - np.sqrt( (np.corrcoef(obs, sim)[0, 1] - 1) ** 2 + 
                                                            (np.mean(sim) / np.mean(obs) - 1) ** 2 + 
                                                            ((np.std(sim) / np.mean(sim)) 
                                                             / (np.std(obs) / np.mean(obs)) - 1) ** 2)  # Kling-Gupta Efficiency (KGE')
    }

    def __init__(self, name, unit):
        self.name       = name
        self.unit       = unit
        self.formula    = self.formulas.get(name)

    def compute(self, obs, sim):
        if self.formula:
            return self.formula(obs, sim)
        else:
            raise ValueError(f"Formula for {self.name} not found!")
