import numpy as np

def _robust_median(xs):
    if not xs: return 0.5
    s = np.sort(xs)
    gaps = np.diff(s)
    gap_indices = np.where(gaps > 0.078 * 1280)[0] # Assuming coords are in pixels, but what are they?
    # Wait, in video_processing.py, are they normalized?
    pass
