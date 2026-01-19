import numpy as np
import tqdm
from scipy.signal import butter, sosfilt
from devito import *
from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from scipy.ndimage import gaussian_filter
from distributed import wait

from devito_ta.model_geom import grad_utils
from devito_ta.prepro import Filter, filter_data, CustomSource


def fm_single_shot(model, geom, save=False, dt=4.0):
    """Run acoustic forward modeling for a single shot.

    This function uses Devito's `AcousticWaveSolver` to simulate receiver data
    for one acquisition geometry (one source location). The receiver data is
    resampled to the desired output time sampling.

    Args:
        model (Model): Devito model containing `vp` and grid information.
        geom (AcquisitionGeometry): Acquisition geometry defining source/receiver
            positions and time axis.
        save (bool, optional): Whether to save the forward wavefield during modeling.
            Defaults to False.
        dt (float, optional): Time sampling used when resampling output receiver data
            via `d_obs.resample(dt)`. Defaults to 4.0.

    Returns:
        tuple:
            - d_obs (Receiver): Simulated receiver data, resampled to `dt`.
            - u0 (TimeFunction): Forward wavefield returned by Devito.
    """
    solver = AcousticWaveSolver(model, geom, space_order=4)
    d_obs, u0, _ = solver.forward(vp=model.vp, save=save, src=geom.src)  
    return d_obs.resample(dt), u0


def fm_multi_shots(model, geometry, n_workers, client, save=False, dt=4.0):
    """Run forward modeling for multiple shots in parallel using a Dask client.

    This function submits `fm_single_shot` tasks to a Dask cluster in batches of
    at most `n_workers` shots at a time, waits for completion, and gathers the
    results into a list.

    Args:
        model (Model): Devito model used for simulation.
        geometry (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel jobs submitted per batch.
        client: Dask client object used for `submit` and `gather`.
        save (bool, optional): Whether to save wavefields per shot in the forward solver.
            Defaults to False.
        dt (float, optional): Output sampling passed to `fm_single_shot` for resampling.
            Defaults to 4.0.

    Returns:
        list: List of per-shot results. Each element matches the return of
        `fm_single_shot`, i.e., `(d_obs_resampled, u0)`.
    """
    futures = []
    shot_data = []

    i = 0
    while i < len(geometry):

        # Submit up to n_workers jobs at a time
        batch = 0
        while batch < n_workers and i < len(geometry):
            geometry_i = AcquisitionGeometry(
                model,
                rec_positions=geometry[i].rec_positions,
                src_positions=geometry[i].src_positions,
                t0=geometry[i].t0, tn=geometry[i].tn, f0=geometry[i].f0,
                src_type=geometry[i].src_type
            )
            futures.append(client.submit(fm_single_shot, model, geometry_i, save=save, dt=dt))
            i += 1
            batch += 1

        # Wait for the current batch
        wait(futures)

        # Gather results
        results = client.gather(futures)
        shot_data.extend(results)

        # Clear futures list for next batch
        futures = []

    return shot_data


def compute_residual(residual, d_obs, d_syn, geom):
    """Compute and store the residual (synthetic - observed) for a shot.

    The residual is written in-place into `residual.data[:]`. Observed data is
    resampled to `geom.dt` to match synthetic data sampling and then cropped to
    synthetic length.

    Args:
        residual (Receiver): Output buffer to store residual values.
        d_obs (Receiver): Observed receiver data.
        d_syn (Receiver): Synthetic receiver data.
        geom (AcquisitionGeometry): Geometry providing target sampling `geom.dt`.

    Returns:
        Receiver: The same `residual` object with updated `.data`.
    """
    # Compute residual
    residual.data[:] = d_syn.data[:] - d_obs.resample(geom.dt).data[:][0:d_syn.data.shape[0], :]
    # residual.data[:] = d_syn.data[:] - d_obs.data[:][0:d_syn.data.shape[0], :]

    return residual


def grad_x_shot(model_true, model_init, geom, d_obs):
    """Compute objective value and gradient contribution for one shot.

    This function runs forward modeling with the current model (`model_init`),
    computes the residual against observed data, and then computes the gradient
    using Devito's adjoint-state implementation (`solver.gradient`).

    Args:
        model_true (Model): Reference model providing grid/time configuration.
        model_init (Model): Current/inversion model used to generate synthetic data.
        geom (AcquisitionGeometry): Shot geometry.
        d_obs (Receiver): Observed data for the shot.

    Returns:
        tuple:
            - obj (float): Shot misfit value defined as `0.5 * ||residual||^2`.
            - grad_i_crop (numpy.ndarray): Cropped gradient array with absorbing
              boundaries (nbl) removed.
    """
    # Get grad function and receivers
    grad_i, residual_i, _, _ = grad_utils(model_true, geom)

    # Initiate wave solver
    solver = AcousticWaveSolver(model_true, geom, space_order=4)
    d_syn, u_syn = solver.forward(vp=model_init.vp, save=True)[0:2]

    # Resample d_syn and residual_i
    residual_i = residual_i.resample(geom.dt)
    d_syn = d_syn.resample(geom.dt)

    # Get residual score
    residual = compute_residual(residual_i, d_obs, d_syn, geom)

    # Calculate objective value and gradient
    obj = 0.5 * norm(residual) ** 2
    solver.gradient(rec=residual, u=u_syn, vp=model_init.vp, grad=grad_i)

    # Remove nbl
    grad_i_crop = np.array(grad_i.data[:])[model_init.nbl:-model_init.nbl, model_init.nbl:-model_init.nbl]

    return obj, grad_i_crop


def normalize(data, vp):
    """Normalize an array by velocity cubed.

    This helper divides `data` by `vp**3`. The operation is performed in-place
    due to `/=`.

    Args:
        data (numpy.ndarray): Array to normalize (modified in-place).
        vp (numpy.ndarray | float): Velocity model (or scalar) used for scaling.
            Must be broadcastable to the shape of `data`.

    Returns:
        numpy.ndarray: Normalized array.
    """
    data /= (vp ** 3)

    return data


def grad_multi_shots(model_true, model_init, geoms, n_workers, client, d_true):
    """Compute total objective and gradient over multiple shots in parallel.

    This function distributes per-shot gradient computations (`grad_x_shot`) using
    a Dask client, accumulates objective values and gradients across all shots,
    and returns the summed objective and gradient.

    Notes:
        - The returned gradient is negated at the end (`grad_total = -grad_total`),
          which may be required depending on the optimizer convention.
        - Absorbing boundary layers (`nbl`) are removed from the accumulated gradient.

    Args:
        model_true (Model): Reference model providing grid/time configuration.
        model_init (Model): Current/inversion model used to generate synthetic data.
        geoms (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel tasks per batch.
        client: Dask client object used for `submit`, `wait`, and `gather`.
        d_true (list[Receiver]): List of observed shot records aligned with `geoms`.

    Returns:
        tuple:
            - obj_total (float): Sum of per-shot objective values.
            - grad_total (numpy.ndarray): Accumulated cropped gradient array.
    """
    nbl = model_init.nbl
    futures = []

    # Initial objective value and gradient
    obj_total = 0.0
    grad_total = np.zeros(model_init.vp.data[nbl:-nbl, nbl:-nbl].shape, dtype=np.float64)

    s = 0
    while s < len(geoms):
        # Submit up to n_workers jobs at a time
        batch = 0

        while batch < n_workers and s < len(geoms):
            geom_s = AcquisitionGeometry(
                model_true,
                rec_positions=geoms[s].rec_positions,
                src_positions=geoms[s].src_positions,
                t0=geoms[s].t0, tn=geoms[s].tn, f0=geoms[s].f0,
                src_type=geoms[s].src_type
            )

            futures.append(client.submit(grad_x_shot,
                                         model_true, model_init,
                                         geom_s, d_true[s]))

            s += 1
            batch += 1

        # Wait for all futures in the current batch to complete
        wait(futures)

        # Gather results
        results = client.gather(futures)

        # Sum objective and gradient results
        for res in results:
            obj_total += res[0]
            grad_total += res[1]

        # Reset futures for next batch
        futures = []

    # Cut water layer
    # grad_total[:, :50] = 0
    grad_total = -grad_total  # deavtivate if using scipy optimizer

    # Normalize and blur gradient
    # grad_total = normalize(grad_total, model_init.vp.data[nbl:-nbl, nbl:-nbl])

    return obj_total, grad_total


# Born approximation
def born_x_shot(model_true, model_background, model_scat, geom):
    """Compute Born (linearized) migrated image for a single shot.

    This function computes a Born-modeled dataset via Devito's Jacobian operator
    and then migrates it using the adjoint Jacobian to obtain a gradient-like
    image (Born image) for one shot.

    Args:
        model_true (Model): Model used to configure solver (grid, operators).
        model_background (Model): Background model used for linearization.
        model_scat: Scattering perturbation passed as `dmin` to Devito Jacobian.
        geom (AcquisitionGeometry): Shot geometry.

    Returns:
        numpy.ndarray: Cropped Born image with boundary layers removed.
    """
    # Get grad function and receivers
    born_i = grad_utils(model_background, geom)[0]
    nbl = model_background.nbl

    # Demigrated operator
    solver = AcousticWaveSolver(model_true, geom, space_order=4)
    _, u_fm = solver.forward(vp=model_background.vp, save=True)[:2]

    u = TimeFunction(name='u', grid=model_background.grid, time_order=2, space_order=4)

    d_born, u, U = solver.jacobian(dmin=model_scat, u=u, vp=model_background.vp)[:3]
    d_born = d_born.resample(geom.dt)

    # Migrated operator
    solver.jacobian_adjoint(rec=d_born, u=u_fm, vp=model_background.vp, grad=born_i)

    # Remove nbl
    born_i_crop = np.array(born_i.data[:])[nbl:-nbl, nbl:-nbl]

    return born_i_crop


def born_multi_shots(model_true, model_background, model_scat, geoms, n_workers, client):
    """Compute accumulated Born image over multiple shots in parallel.

    This function distributes per-shot Born imaging (`born_x_shot`) via a Dask
    client and sums the resulting Born images. It also returns a cropped and
    normalized version of the scattering model.

    Args:
        model_true (Model): Model used to configure solver.
        model_background (Model): Background model for linearization.
        model_scat: Scattering perturbation used in the Jacobian operator.
        geoms (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel tasks per batch.
        client: Dask client object used for `submit`, `wait`, and `gather`.

    Returns:
        tuple:
            - model_scat (numpy.ndarray): Cropped (nbl removed) and normalized
              scattering model.
            - born_total (numpy.ndarray): Accumulated Born image (sum over shots).
    """
    nbl = model_background.nbl
    futures = []

    # Initial born, forward wavefield, and linear wavefield
    born_total = np.zeros(model_true.vp.data[nbl:-nbl, nbl:-nbl].shape, dtype=np.float64)

    s = 0
    while s < len(geoms):
        # Submit up to n_workers jobs at a time
        batch = 0

        while batch < n_workers and s < len(geoms):
            geom_s = AcquisitionGeometry(
                model_true,
                rec_positions=geoms[s].rec_positions,
                src_positions=geoms[s].src_positions,
                t0=geoms[s].t0, tn=geoms[s].tn, f0=geoms[s].f0,
                src_type=geoms[s].src_type
            )

            futures.append(client.submit(born_x_shot, model_true,
                                         model_background,
                                         model_scat, geom_s))

            s += 1
            batch += 1

        # Wait for all futures in the current batch to complete
        wait(futures)

        # Gather results
        results = client.gather(futures)

        # Sum born results
        for res in results:
            born_total += res

        # Reset futures for next batch
        futures = []

    model_scat = model_scat.data[nbl:-nbl, nbl:-nbl]
    model_scat = normalize(model_scat, model_background.vp.data[nbl:-nbl, nbl:-nbl])

    return model_scat, born_total
