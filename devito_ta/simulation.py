import numpy as np
from scipy.signal import butter, sosfilt
from devito import *
from examples.seismic import Model, AcquisitionGeometry
from examples.seismic.acoustic import AcousticWaveSolver
from distributed import wait

from devito_ta.model_geom import grad_utils


def fm_single_shot(model, geom, save=False, dt=4.0):
    """Run acoustic forward modeling for a single shot.

    This function uses Devito's `AcousticWaveSolver` to simulate one shot record
    (receiver data) for a given acquisition geometry. The returned receiver data
    is resampled to the specified output sampling `dt`.

    Args:
        model (Model): Devito seismic model containing the velocity field (`model.vp`)
            and grid information.
        geom (AcquisitionGeometry): Acquisition geometry defining source/receiver
            positions and the time axis.
        save (bool, optional): Whether to save the wavefield during forward modeling.
            Defaults to False.
        dt (float, optional): Output time sampling used when resampling the receiver
            data via `d_obs.resample(dt)`. Defaults to 4.0.

    Returns:
        tuple:
            - d_obs (Receiver): Simulated receiver data resampled to `dt`.
            - u0 (TimeFunction): Forward wavefield returned by Devito.
    """
    solver = AcousticWaveSolver(model, geom, space_order=4)
    d_obs, u0, _ = solver.forward(vp=model.vp, save=save, src=geom.src) #[0:2]
    return d_obs.resample(dt), u0


def fm_multi_shots(model, geometry, n_workers, client, save=False, dt=4.0):
    """Run forward modeling for multiple shots in parallel using a Dask client.

    This function submits `fm_single_shot` tasks to a Dask cluster in batches
    of at most `n_workers` jobs, waits for the batch completion, and gathers
    results into a list.

    Args:
        model (Model): Devito seismic model used for simulation.
        geometry (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel jobs submitted per batch.
        client: Dask client used to submit and gather tasks.
        save (bool, optional): Whether to save wavefields during forward modeling.
            Defaults to False.
        dt (float, optional): Output time sampling used when resampling the receiver
            data. Defaults to 4.0.

    Returns:
        list: List of per-shot results, where each element is `(d_obs_resampled, u0)`
        as returned by `fm_single_shot`.
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
    """Compute and store the residual (synthetic - observed) for one shot.

    The residual is written in-place into `residual.data[:]`. Observed data is
    resampled to match the geometry time step (`geom.dt`) and cropped to match
    the synthetic data length.

    Args:
        residual (Receiver): Output buffer to store residual values.
        d_obs (Receiver): Observed receiver data.
        d_syn (Receiver): Synthetic receiver data.
        geom (AcquisitionGeometry): Geometry providing `geom.dt` for resampling.

    Returns:
        Receiver: The same `residual` object with updated `.data`.
    """
    # Compute residual
    residual.data[:] = d_syn.data[:] - d_obs.resample(geom.dt).data[:][0:d_syn.data.shape[0], :]
    #residual.data[:] = d_syn.data[:] - d_obs.data[:][0:d_syn.data.shape[0], :]

    return residual


def grad_x_shot(model_true, model_init, geom, d_obs):
    """Compute objective value and gradient contribution for one shot.

    This function performs:
      1) Forward modeling to generate synthetic data using `model_init.vp`
      2) Residual computation against observed data
      3) L2 objective computation: `0.5 * ||residual||^2`
      4) Gradient computation via Devito's adjoint-state method
      5) Cropping out absorbing boundary layers (`nbl`)

    Args:
        model_true (Model): Reference model providing solver/grid configuration.
        model_init (Model): Current/inversion model used to generate synthetic data.
        geom (AcquisitionGeometry): Shot geometry.
        d_obs (Receiver): Observed data for this shot.

    Returns:
        tuple:
            - obj (float): Shot objective value.
            - grad_i_crop (numpy.ndarray): Cropped gradient array with boundary layers removed.
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
    obj = 0.5 * norm(residual)**2
    solver.gradient(rec=residual, u=u_syn, vp=model_init.vp, grad=grad_i)

    # Remove nbl
    grad_i_crop = np.array(grad_i.data[:])[model_init.nbl:-model_init.nbl, model_init.nbl:-model_init.nbl]

    return obj, grad_i_crop


def normalize(data, vp):
    """Normalize an array by velocity cubed.

    This function divides `data` by `vp**3`. The operation is performed in-place
    due to the use of `/=`.

    Args:
        data (numpy.ndarray): Array to normalize (modified in-place).
        vp (numpy.ndarray | float): Velocity model (or scalar) used for scaling.
            Must be broadcastable to `data`.

    Returns:
        numpy.ndarray: Normalized array.
    """
    data /= (vp**3)

    return data


def grad_multi_shots(model_true, model_init, geoms, n_workers, client, d_true, blur):
    """Compute total objective and gradient across multiple shots in parallel.

    This function distributes per-shot gradient computations (`grad_x_shot`)
    through a Dask client and accumulates objective and gradient across all shots.
    It then applies a simple mute on the first 50 columns (water-layer cut),
    negates the gradient (optimizer convention), and normalizes by `vp**3`.

    Notes:
        The `blur` argument is accepted but currently unused in this function.

    Args:
        model_true (Model): Reference model providing solver/grid configuration.
        model_init (Model): Current/inversion model used to generate synthetic data.
        geoms (list[AcquisitionGeometry]): List of shot geometries.
        n_workers (int): Maximum number of parallel tasks submitted per batch.
        client: Dask client used for `submit`, `wait`, and `gather`.
        d_true (list[Receiver]): Observed data list aligned with `geoms`.
        blur: Placeholder parameter (currently unused).

    Returns:
        tuple:
            - obj_total (float): Sum of per-shot objective values.
            - grad_total (numpy.ndarray): Accumulated cropped gradient array after
              mute, sign convention, and normalization.
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

    # Cut water layer
    grad_total[:, :50] = 0
    grad_total = -grad_total     # deavtivate if using scipy optimizer

    # Normalize and blur gradient
    grad_total = normalize(grad_total, model_init.vp.data[nbl:-nbl, nbl:-nbl])

    return obj_total, grad_total
