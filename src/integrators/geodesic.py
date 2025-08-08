import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from astropy import units as u

from einsteinpy.geodesic import Geodesic, Timelike, Nulllike
from einsteinpy.coordinates import Cartesian, Spherical
from scipy.spatial.transform import Rotation as R

def convert_ray_to_einstein_input(ray_pos_cart: tuple, ray_mom_cart: tuple):
    """Convert ray to normalized input (spherical) for EinsteinPy nulllike geodesic using map.

    The returned vector is normalized.
    Assumed ray position is on the y-axis.
    This requires csv file "coor_map.csv" generated from output_coordinate_map.
    If the ray position changes, the map is likely to change.
    """

    # Read the convert map
    df = pd.read_csv("coor_map.csv")
    theta = df["theta"].to_numpy()
    slope_yz = df["slope_yz"].to_numpy()

    # Find the slope of dz/dy
    # Rotate the ray to yz-plane
    rotation = R.from_euler("Y", -np.arctan2(ray_mom_cart[0], ray_mom_cart[2]), degrees=False)
    ray_mom_on_yz_cart = rotation.apply(np.asarray(ray_mom_cart)).flatten()
    slope = ray_mom_on_yz_cart[2] / ray_mom_on_yz_cart[1]

    # Look up the slope and get the theta
    if slope >= slope_yz.max():
        mom_theta = theta[0]
    elif slope <= slope_yz.min():
        mom_theta = theta[-1]
    else:
        i = np.argmax(slope > slope_yz)
        m = slope - slope_yz[i-1]
        n = slope_yz[i] - slope
        mom_theta = (n * theta[i-1] + m * theta[i]) / (m + n)

    # Get the vector spherical coordinate, and then do rotation along er
    unit_length = np.sqrt(1.0**2 + mom_theta**2)
    mom_sphe = np.array([-1.0 / unit_length, mom_theta / unit_length, 0.0])
    rotation = R.from_euler("X", np.arctan2(ray_mom_cart[0], ray_mom_cart[2]), degrees=False)
    mom_sphe = rotation.apply(mom_sphe).flatten()

    return mom_sphe

def inline(pos_cart: tuple, mom_cart: tuple):
    pass
def test_inline(args):

    import sys
    sys.output = "get " + str(args)
