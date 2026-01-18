import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from examples.seismic import Model, AcquisitionGeometry
from devito import *
from examples.seismic import Receiver


def get_model(vp, shape, spacing, origin, nbl=70, save=False, title="model"):
    """
    Create a Devito seismic Model object.

    This utility function initializes a 2D seismic velocity model with
    absorbing boundary layers and standard numerical settings for wave
    propagation experiments.

    Args:
        vp (ndarray): Velocity model array 
        shape (tuple): Number of grid points in each spatial dimension.
        spacing (tuple): Grid spacing in each spatial dimension.
        origin (tuple): Physical origin of the model domain.
        nbl (int, optional): Number of boundary layers for damping.
            Defaults to 70.
        save (bool, optional): Placeholder flag for saving the model
            (currently unused). Defaults to False.
        title (str, optional): Placeholder title for the model
            (currently unused). Defaults to "model".

    Returns:
        Model: Devito Model object containing the velocity field and
        numerical discretization parameters.
    """
    model = Model(
        vp=vp,
        origin=origin,
        spacing=spacing,
        shape=shape,
        space_order=4,
        nbl=nbl,
        bcs="damp",
    )

    return model


def get_geometry(
    model,
    src_x,
    src_z,
    rec_x,
    rec_z,
    t0,
    tn,
    nof,
    src_type,
    f0=20.0,
    dt=None,
):
    """
    Generate acquisition geometries for multiple seismic shots.

    This function creates a list of `AcquisitionGeometry` objects,
    one for each source position. Receiver coordinates are generated
    relative to each source, allowing flexible source–receiver offsets.

    Args:
        model (Model): Devito Model object defining the computational domain.
        src_x (array_like): X-coordinates of seismic sources.
        src_z (array_like): Z-coordinates of seismic sources.
        rec_x (array_like): X-offsets of receivers relative to the source.
        rec_z (array_like): Z-coordinates of receivers.
        t0 (float): Start time of the simulation.
        tn (float): End time of the simulation.
        nof (float): Near-offset distance applied to receiver positions.
        src_type (str): Type of seismic source (e.g., "Ricker").
        f0 (float, optional): Dominant frequency of the source wavelet (Hz).
            Defaults to 20.0.
        dt (float, optional): Time sampling interval. Defaults to None
            (Devito determines it automatically).

    Returns:
        list: List of `AcquisitionGeometry` objects, one per source.
    """
    nsrc, nrec = len(src_x), len(rec_x)

    geometries = []

    src_coordinates = np.empty((nsrc, 2))
    src_coordinates[:, 0] = src_x
    src_coordinates[:, 1] = src_z

    for i in range(nsrc):
        rec_coordinates = np.empty((nrec, 2))
        rec_coordinates[:, 0] = rec_x
        rec_coordinates[:, 1] = rec_z

        # Shift receivers relative to current source
        rec_coordinates[:, 0] = src_coordinates[i, 0] - rec_x - nof

        geometry = AcquisitionGeometry(
            model,
            rec_positions=rec_coordinates[:, :],
            src_positions=src_coordinates[i, :],
            t0=t0,
            tn=tn,
            f0=f0,
            src_type=src_type,
        )
        geometries.append(geometry)

    return geometries


def grad_utils(model_true, geom):
    """
    Create gradient and data placeholders for seismic inversion.

    This helper function initializes Devito symbolic objects commonly
    required for gradient-based inversion workflows, including residuals
    and observed/synthetic data receivers.

    Args:
        model_true (Model): True or background Devito Model used for
            gradient computation.
        geom (AcquisitionGeometry): Acquisition geometry defining
            time axis and receiver positions.

    Returns:
        tuple:
            - grad (Function): Gradient field defined on the model grid.
            - residual (Receiver): Receiver object storing data residuals.
            - d_obs (Receiver): Receiver object for observed data.
            - d_syn (Receiver): Receiver object for synthetic data.
    """
    grad = Function(name="grad", grid=model_true.grid)

    residual = Receiver(
        name="residual",
        grid=model_true.grid,
        time_range=geom.time_axis,
        coordinates=geom.rec_positions,
    )

    d_obs = Receiver(
        name="d_obs",
        grid=model_true.grid,
        time_range=geom.time_axis,
        coordinates=geom.rec_positions,
    )

    d_syn = Receiver(
        name="d_syn",
        grid=model_true.grid,
        time_range=geom.time_axis,
        coordinates=geom.rec_positions,
    )

    return grad, residual, d_obs, d_syn
