import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from einsteinpy.geodesic import Nulllike

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
    try:
        slope = ray_mom_on_yz_cart[2] / ray_mom_on_yz_cart[1]
    except:
        if ray_mom_on_yz_cart[2] < 0:
            slope = -5000
        else:
            slope = 5000

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
    rotation = R.from_euler("X", -np.arctan2(ray_mom_cart[0], ray_mom_cart[2]), degrees=False)
    mom_sphe = rotation.apply(mom_sphe).flatten()

    return mom_sphe

def position_cartesian_to_spherical(x, y, z):
    """Convert cartesian x, y, z to spherical r, theta, phi"""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)

    return [r, theta, phi]

def keep_tracking(p1, p2, v1, v2):
    """Keep tracking the ray or not
    p1, p2, v1, v2 are all in cartesian

    If a ray is leaving the blackhole and not affected by it, meaning:
        1. dr increases and
        2. changes in angle smaller than e-3
        return False
    else:
        return True
    """

    # compute changes in angle, assume cos_theta > 0
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    theta = np.arccos(cos_theta)

    # compute dr
    r1, _, _ = position_cartesian_to_spherical(p1[0], p1[1], p1[2])
    r2, _, _ = position_cartesian_to_spherical(p2[0], p2[1], p2[2])
    dr = r2 - r1

    if theta < 0.001 and dr > 0:
        return False
    else:
        return True

def traj_cross_bh(traj):
    """Check if the trajectory cross the blackhole

    `traj` is a ndarray with shape (N, 3)
    Check if the geod solution already cross the event horizon and will brake the integrator:
        1. The changing angle is > pi/2
    """

    cos_theta = np.diag(np.dot(traj[:-1, :], np.transpose(traj[1:, :]))) / (np.linalg.norm(traj[:-1], axis=1) * np.linalg.norm(traj[1:], axis=1))
    theta = np.arccos(cos_theta)

    abnormal = np.argwhere(theta > np.pi/2.0)

    if len(abnormal) != 0:
        return True
    else:
        return False


def get_outgoing_ray(init_pos: tuple, init_mom: tuple):
    """Compute the outgoing ray from the incoming ray

    If the ray falls into the blackhole, position and momentum will set to (0, 0, 0).
    This is because we are not dealing with the emission from the blackhole itself.
    """

    pos = init_pos
    mom = init_mom
    init = True

    while init or keep_tracking(pos, pos, mom, mom):
        init = False

        # Convert cartesian to spherical
        pos_sphe = position_cartesian_to_spherical(pos[0], pos[1], pos[2])
        mom_sphe = convert_ray_to_einstein_input(pos, mom)

        # Compute geodesic step
        try:
            geod = Nulllike(
                metric = "Schwarzschild",
                metric_params = (0,),
                position=pos_sphe, #position,
                momentum=mom_sphe, #momentum (r, theta, phi), momentum length affects the trajectory length
                steps=10,
                delta=0.01,
                return_cartesian=True,  # if True, the position will returned in cartesian and momentum in spherical, otherwise both are spherical
                omega=0.01,             # Small omega values lead to more stable integration
                suppress_warnings=True, # Uncomment to view the tolerance warning
            )
        except:
            pos = (0, 0, 0)
            mom = (0, 0, 0)
            break

        # Check if the geod solution already cross the event horizon and will brake the integrator
        traj = geod.trajectory[1][:, 1:4]

        if traj_cross_bh(traj):
            pos = (0, 0, 0)
            mom = (0, 0, 0)
            break

        # Compute the position and momentum in cartesian
        p1 = traj[-3]
        p2 = traj[-2]
        p3 = traj[-1]
        v1 = p2 - p1
        v2 = p3 - p2

        pos = p2
        mom = v2

        # Check if we should keep tracking
        if keep_tracking(p1, p2, v1, v2) is not True:
            break

    # Bind pos and mom to sys.output
    import sys
    sys.output_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
    sys.output_mom = (float(mom[0]), float(mom[1]), float(mom[2]))