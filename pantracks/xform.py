import numpy as np

def dfcrop(df, xmin=-np.inf, xmax=np.inf, ymin=-np.inf, ymax=np.inf, xcol='x', ycol='y'):
    """Crop a particle tracks DataFrame"""
    return df[(df[xcol] > xmin) & (df[xcol] < xmax) & \
            (df[ycol] > ymin) & (df[ycol] < ymax)]

