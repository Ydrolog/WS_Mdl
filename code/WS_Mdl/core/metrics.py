import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


class Vld_Mtc:
    registry = {
        'NSE': {
            'unit': '-',
            'formula': lambda obs, sim: 1 - (np.sum((obs - sim) ** 2) / np.sum((obs - np.mean(obs)) ** 2)),
        },
        'RMSE': {
            'unit': 'm',
            'formula': lambda obs, sim: np.sqrt(mean_squared_error(obs, sim)),
        },
        'MAE': {
            'unit': 'm',
            'formula': lambda obs, sim: np.mean(np.abs(obs - sim)),
        },
        'Correlation': {
            'unit': '-',
            'formula': lambda obs, sim: np.corrcoef(obs, sim)[0, 1],
        },
        'Bias Ratio': {
            'unit': '-',
            'formula': lambda obs, sim: np.mean(sim) / np.mean(obs),
        },
        'Variability Ratio': {
            'unit': '-',
            'formula': lambda obs, sim: (np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs)),
        },
        'KGE': {
            'unit': '-',
            'formula': lambda obs, sim: (
                1
                - np.sqrt(
                    (np.corrcoef(obs, sim)[0, 1] - 1) ** 2
                    + (np.mean(sim) / np.mean(obs) - 1) ** 2
                    + ((np.std(sim) / np.mean(sim)) / (np.std(obs) / np.mean(obs)) - 1) ** 2
                )
            ),
        },
    }

    def __init__(self, name):
        if name not in self.registry:
            raise ValueError(f'Formula for {name} not found!')

        self.name = name
        self.unit = self.registry[name]['unit']
        self.formula = self.registry[name]['formula']

    @staticmethod
    def _safe_compute(formula, obs, sim):
        try:
            obs = np.asarray(obs, dtype=float)
            sim = np.asarray(sim, dtype=float)

            if obs.size == 0 or sim.size == 0:
                return np.nan

            if obs.shape != sim.shape:
                return np.nan

            if np.all(np.isnan(obs)) or np.all(np.isnan(sim)):
                return np.nan

            value = formula(obs, sim)

            return value if np.isfinite(value) else np.nan

        except Exception:
            return np.nan

    def compute(self, obs, sim):
        return self._safe_compute(self.formula, obs, sim)

    @classmethod
    def all(cls):
        return [cls(name) for name in cls.registry]

    @classmethod
    def names(cls):
        return list(cls.registry)

    @classmethod
    def units(cls):
        return {name: spec['unit'] for name, spec in cls.registry.items()}

    @classmethod
    def to_DF(cls):
        return (
            pd.DataFrame.from_dict(cls.registry, orient='index')
            .drop(columns='formula')
            .rename_axis('Metric')
            .reset_index()
        )

    @classmethod
    def compute_all(cls, obs, sim, sim2=None):
        DF = pd.DataFrame(
            [
                {
                    'Metric': name,
                    'Unit': spec['unit'],
                    'S': cls._safe_compute(spec['formula'], obs, sim),
                }
                for name, spec in cls.registry.items()
            ]
        )

        if sim2 is not None:  # Add a second column of metrics for the second simulation
            DF['B'] = [cls._safe_compute(spec['formula'], obs, sim2) for name, spec in cls.registry.items()]

        DF.set_index('Metric', inplace=True)
        DF = DF[['S'] + (['B'] if sim2 is not None else []) + ['Unit']]
        return DF

    @classmethod
    def compute_selected(cls, obs, sim, names):
        return pd.DataFrame(
            [
                {
                    'Metric': name,
                    'Unit': cls.registry[name]['unit'],
                    'Value': cls._safe_compute(cls.registry[name]['formula'], obs, sim),
                }
                for name in names
            ]
        )
