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
    """Run forward modeling for a single shot and return shot record + wavefield.

    This function performs acoustic forward modeling using Devito's
    `AcousticWaveSolver`. It returns the simulated receiver data (optionally
    resampled) and the saved wavefield `u0` (if `save=True` in Devito context).

    Args:
        model (Model): Devito seismic model containing velocity (`model.vp`) and grid.
        geom (AcquisitionGeometry): Acquisition geometry for a single shot.
        save (bool, optional): Whether to save the forward wavefield during modeling.
            Defaults to False.
        dt (float, optional): Output receiver data time sampling used in `resample(dt)`.
            Defaults to 4.0.

    Returns:
        tuple:
            - d_obs (Receiver): Simulated receiver data resampled to `dt`.
            - u0 (TimeFunction): Forward wavefield returned by Devito solver.
    """
    solver = AcousticWaveSolver(model, geom, space_order=4)
    d_obs, u0, _ = solver.forward(vp=model.vp, save=save, src=geom.src)  # [0:2]
    return d_obs.resample(dt), u0


def fm_multi_shots(model, geometry, n_workers, client, save=False, dt=4.0):
    """Run forward modeling for multiple shots in parallel using a Dask client.

    This function submits batches of `fm_single_shot` jobs to a Dask cluster
    and gathers results. Jobs are submitted in batches up to `n_workers` to
    control parallelism.

    Args:
        model (Model): Devito seismic model.
        geometry (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel tasks submitted per batch.
        client: Dask client used to submit and gather tasks.
        save (bool, optional): Whether to save wavefields in each forward simulation.
            Defaults to False.
        dt (float, optional): Output receiver data time sampling for resampling.
            Defaults to 4.0.

    Returns:
        list: List of results for each shot, where each element is
        `(d_obs_resampled, u0)` as returned by `fm_single_shot`.
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
    """Compute data residual (synthetic - observed) into a Receiver buffer.

    This function writes the residual directly into `residual.data[:]`.

    Notes:
        - Current implementation assumes `d_obs` and `d_syn` are already aligned
          in time sampling and shape.
        - Some alternative resampling/cropping logic is commented out.

    Args:
        residual (Receiver): Receiver object used as output buffer.
        d_obs (Receiver): Observed data.
        d_syn (Receiver): Synthetic data.
        geom (AcquisitionGeometry): Acquisition geometry (currently unused).

    Returns:
        Receiver: The same `residual` object with updated `.data`.
    """
    # Compute residual
    # residual.data[:] = d_syn.data[:] - d_obs.resample(geom.dt).data[:][0:d_syn.data.shape[0], :]
    residual.data[:] = d_syn.data[:] - d_obs.data[:]  # [0:d_syn.data.shape[0], :]

    return residual


def grad_x_shot(model_true, model_init, geom, d_obs, freq, nfilt):
    """Compute objective value and gradient contribution for one shot.

    This function:
      1) Runs forward modeling to generate synthetic data using `model_init.vp`
      2) Applies a lowpass filter at cutoff `freq`
      3) Computes residual (synthetic - observed)
      4) Computes the L2 objective and gradient via Devito's adjoint-state method
      5) Crops absorbing boundary layers (nbl)

    Args:
        model_true (Model): Reference model providing the computational grid and nbl.
        model_init (Model): Current/inversion model used to generate synthetic data.
        geom (AcquisitionGeometry): Geometry for the shot.
        d_obs (Receiver): Observed data for this shot (must match geometry receiver layout).
        freq (float): Cutoff frequency for lowpass filtering (Hz).
        nfilt (int): Butterworth filter order.

    Returns:
        tuple:
            - obj (float): Shot objective value, 0.5 * ||residual||^2.
            - grad_i_crop (numpy.ndarray): Cropped gradient array (without nbl).
    """
    nbl = model_true.nbl

    # Get grad function and receivers
    grad_i, residual_i, _, _ = grad_utils(model_true, geom)

    # Initiate wave solver
    solver = AcousticWaveSolver(model_true, geom, space_order=4)
    d_syn, u_syn = solver.forward(vp=model_init.vp, save=True)[0:2]

    # Filter data
    Filt = Filter([freq], [nfilt], geom.dt, plotflag=False)
    d_syn_data = Filt.apply_filter(d_syn.data.T, ifilt=0).T[None, :]

    _, _, d_syn, _ = grad_utils(model_true, geom)
    d_syn = d_syn.resample(geom.dt)
    d_syn.data[:] = d_syn_data[0]

    # Resample d_syn and residual_i
    residual_i = residual_i.resample(geom.dt)

    # Get residual score
    residual = compute_residual(residual_i, d_obs, d_syn, geom)

    # Calculate objective value and gradient
    obj = 0.5 * norm(residual) ** 2
    solver.gradient(rec=residual, u=u_syn, vp=model_init.vp, grad=grad_i)

    # Remove nbl
    grad_i_crop = np.array(grad_i.data[:])[nbl:-nbl, nbl:-nbl]

    return obj, grad_i_crop


def normalize(data, vp):
    """Normalize (scale and clip) a gradient-like array using velocity `vp`.

    This utility applies a scaling factor of `2 / vp^3` and then clips the
    result to [-1, 1]. Often used to stabilize updates or normalize training data.

    Args:
        data (numpy.ndarray): Input array to normalize. Modified in-place due to `*=`.
        vp (numpy.ndarray | float): Velocity model used for scaling. Must be broadcastable
            to `data`.

    Returns:
        numpy.ndarray: Normalized array (same object as `data` if in-place applies).
    """
    data *= (2 / (vp ** 3))
    data = np.clip(data, -1, 1)

    return data


def grad_multi_shots(model_true, model_init, geoms, n_workers, client, d_true, freq, nfilt):
    """Compute total objective and gradient across multiple shots in parallel.

    This function parallelizes per-shot gradient computation using Dask. It submits
    `grad_x_shot` tasks in batches, gathers results, and accumulates the total
    objective value and gradient.

    Notes:
        - The returned gradient is negated at the end (`grad_total = -grad_total`).
          This may be required depending on the optimizer convention.

    Args:
        model_true (Model): Reference model for grid and boundary settings.
        model_init (Model): Current/inversion model used for synthetic modeling.
        geoms (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel tasks per batch.
        client: Dask client used to submit and gather tasks.
        d_true (list[Receiver]): Observed data list aligned with `geoms` (same length/order).
        freq (float): Cutoff frequency for filtering (Hz).
        nfilt (int): Filter order.

    Returns:
        tuple:
            - obj_total (float): Sum of shot objectives.
            - grad_total (numpy.ndarray): Cropped accumulated gradient array.
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
                                         geom_s, d_true[s], freq, nfilt))

            s += 1
            batch += 1

        # Tunggu semua future batch selesai
        wait(futures)

        # Ambil hasil
        results = client.gather(futures)

        # Jumlahkan hasil objective dan grad
        for res in results:
            obj_total += res[0]
            grad_total += res[1]

        # Reset futures untuk batch berikutnya
        futures = []

    grad_total = -grad_total  # deactivate if using scipy optimizer

    return obj_total, grad_total


# Born approximation
def born_x_shot(model_true, model_background, model_scat, geom, freq, nfilt):
    """Compute Born (linearized) migrated image for a single shot.

    This function:
      1) Runs forward modeling in the background model to obtain wavefield `u_fm`
      2) Applies the Jacobian (Born modeling) using scattering perturbation `model_scat`
      3) Filters the Born data
      4) Applies the adjoint Jacobian (migration) to obtain a Born image / gradient-like field
      5) Crops absorbing boundaries (nbl)

    Args:
        model_true (Model): Model providing solver configuration (grid, etc.).
        model_background (Model): Background model used for linearization.
        model_scat: Scattering perturbation used as `dmin` in Devito Jacobian.
        geom (AcquisitionGeometry): Shot geometry.
        freq (float): Cutoff frequency for filtering (Hz).
        nfilt (int): Filter order.

    Returns:
        numpy.ndarray: Cropped Born image (without boundary layers).
    """
    # Get grad function and receivers
    born_i = grad_utils(model_background, geom)[0]
    nbl = model_background.nbl

    # Demigrated operator
    solver = AcousticWaveSolver(model_true, geom, space_order=4)
    _, u_fm = solver.forward(vp=model_background.vp, save=True)[:2]

    u = TimeFunction(name='u', grid=model_background.grid,
                     time_order=2, space_order=4)

    d_born, u, U = solver.jacobian(dmin=model_scat, u=u, vp=model_background.vp)[:3]
    d_born = d_born.resample(geom.dt)

    # Filter data
    Filt = Filter([freq], [nfilt], geom.dt, plotflag=False)
    d_born_data = Filt.apply_filter(d_born.data.T, ifilt=0).T[None, :]

    _, _, d_born, _ = grad_utils(model_true, geom)
    d_born = d_born.resample(geom.dt)
    d_born.data[:] = d_born_data[0]

    # Migrated operator
    solver.jacobian_adjoint(rec=d_born, u=u_fm, vp=model_background.vp, grad=born_i)

    # Remove nbl
    born_i_crop = np.array(born_i.data[:])[nbl:-nbl, nbl:-nbl]

    return born_i_crop


def born_multi_shots(model_true, model_background, model_scat, geoms, n_workers, client, freq, nfilt):
    """Compute the accumulated Born image across multiple shots in parallel.

    This function parallelizes `born_x_shot` per-shot computations using Dask,
    accumulates the resulting Born images, and also returns the cropped scattering
    model.

    Args:
        model_true (Model): Model providing solver configuration.
        model_background (Model): Background model used for Born linearization.
        model_scat: Scattering perturbation passed to the Jacobian.
        geoms (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel tasks per batch.
        client: Dask client used to submit and gather tasks.
        freq (float): Cutoff frequency for filtering (Hz).
        nfilt (int): Filter order.

    Returns:
        tuple:
            - model_scat (numpy.ndarray): Cropped scattering model (without nbl).
            - born_total (numpy.ndarray): Accumulated Born image (sum over shots).
    """
    nbl = model_background.nbl
    futures = []

    # Initial born, forward wavefield, ad linear wavefield
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
                                         model_background, model_scat,
                                         geom_s, freq, nfilt))

            s += 1
            batch += 1

        # Tunggu semua future batch selesai
        wait(futures)

        # Ambil hasil
        results = client.gather(futures)

        # Jumlahkan hasil objective dan grad
        for res in results:
            born_total += res

        # Reset futures untuk batch berikutnya
        futures = []

    model_scat = model_scat.data[nbl:-nbl, nbl:-nbl]

    return model_scat, born_total
