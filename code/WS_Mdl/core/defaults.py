from pathlib import Path

__all__ = ['CRS', 'Pa_WS']

CRS = 'EPSG:28992'

Pa_WS = Path(__file__).absolute().parents[3]
