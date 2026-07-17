from pathlib import Path

__all__ = ['CRS', 'Pa_WS']

CRS = 'EPSG:28992'

Pa_WS = Path(__file__).absolute().parents[3]

quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
