import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_shotrecord_cust(
    rec1,
    title1,
    rec2,
    title2,
    rec3,
    title3,
    model,
    t0,
    tn,
    save=False,
    colorbar=True,
):
    """
    Plot and compare three seismic shot records side by side.

    Each shot record is visualized as an image of receiver amplitudes
    over time. The function is typically used to compare different
    modeling results or processing stages.

    Args:
        rec1 (ndarray): First shot record with shape (nt, nreceivers).
        title1 (str): Title for the first shot record.
        rec2 (ndarray): Second shot record with shape (nt, nreceivers).
        title2 (str): Title for the second shot record.
        rec3 (ndarray): Third shot record with shape (nt, nreceivers).
        title3 (str): Title for the third shot record.
        model (Model): Seismic model object containing domain origin.
        t0 (int): Start time index for plotting.
        tn (int): End time index for plotting.
        save (bool, optional): If True, saves the figure to disk.
            Defaults to False.
        colorbar (bool, optional): If True, displays a shared colorbar.
            Defaults to True.

    Returns:
        None
    """
    extent = [model.origin[0], rec1.shape[1], tn / 1000, t0]

    titles = [title1, title2, title3]
    data = [rec1, rec2, rec3]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))

    for ax_i, title, dat in zip(ax, titles, data):
        cax = ax_i.imshow(
            dat,
            cmap="seismic",
            vmin=-2,
            vmax=2,
            aspect="auto",
            extent=extent,
        )
        ax_i.set_title(title, fontsize=14, pad=10)
        ax_i.set_xlabel("Trace Number")
        ax_i.set_ylabel("Time (s)")
        ax_i.invert_xaxis()

    fig.subplots_adjust(right=0.88, wspace=0.3)

    if colorbar:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
        fig.colorbar(cax, cax=cbar_ax)

    if save:
        plt.savefig("pics/shotrecord.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_velocity_cust(
    model,
    source=None,
    receiver=None,
    colorbar=True,
    cmap="jet",
    save=False,
    name="velocity",
    title="model",
):
    """
    Plot a 2D seismic velocity model with optional source and receiver locations.

    The velocity field is extracted from a Devito `Model` object and
    displayed as an image. Source and receiver positions can be
    overlaid for visualization of acquisition geometry.

    Args:
        model (Model): Seismic model object containing velocity fields
            and domain information.
        source (ndarray, optional): Array of source coordinates with
            shape (nsrc, 2). Defaults to None.
        receiver (ndarray, optional): Array of receiver coordinates with
            shape (nrec, 2). Defaults to None.
        colorbar (bool, optional): If True, displays a colorbar.
            Defaults to True.
        cmap (str, optional): Colormap used for velocity visualization.
            Defaults to "jet".
        save (bool, optional): If True, saves the figure to disk.
            Defaults to False.
        name (str, optional): Filename used when saving the figure.
            Defaults to "velocity".
        title (str, optional): Title of the plot. Defaults to "model".

    Returns:
        None
    """
    domain_size = 1e-3 * np.array(model.domain_size)
    extent = [
        model.origin[0],
        model.origin[0] + domain_size[0],
        model.origin[1] + domain_size[1],
        model.origin[1],
    ]

    slices = tuple(slice(model.nbl, -model.nbl) for _ in range(2))

    if getattr(model, "vp", None) is not None:
        field = model.vp.data[slices]
    else:
        field = model.lam.data[slices]

    plot = plt.imshow(
        np.transpose(field),
        animated=True,
        cmap=cmap,
        vmin=1.5,
        vmax=4.7,
        extent=extent,
    )

    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("X position (km)")
    plt.ylabel("Depth (km)")

    if receiver is not None:
        plt.scatter(
            1e-3 * receiver[:, 0],
            1e-3 * receiver[:, 1],
            s=25,
            c="w",
            marker="D",
        )

    if source is not None:
        plt.scatter(
            1e-3 * source[:, 0],
            1e-3 * source[:, 1],
            s=25,
            c="red",
            marker="o",
        )

    plt.xlim(model.origin[0], model.origin[0] + domain_size[0])
    plt.ylim(model.origin[1] + domain_size[1], model.origin[1])

    if colorbar:
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(plot, cax=cax)
        cbar.set_label("Velocity (km/s)")

    if save:
        plt.savefig(f"pics/{name}.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_acquisition_geometry(geometries, title):
    """
    Plot acquisition geometry for multiple shots.

    Each acquisition geometry is plotted on a separate horizontal
    level, showing source and receiver positions along the x-axis.

    Args:
        geometries (list): List of AcquisitionGeometry objects.
        title (str): Title of the plot.

    Returns:
        None
    """
    nshots = len(geometries)

    plt.figure(figsize=(20, 15))
    plt.title(title)

    for i, geom in enumerate(geometries):
        rec = geom.rec_positions[:, 0]
        src = geom.src_positions[0, 0]

        plt.scatter(
            rec,
            np.full_like(rec, i + 1),
            marker="o",
            c="r",
            label="Receiver [200]" if i == 0 else "",
        )
        plt.scatter(
            src,
            i + 1,
            s=500,
            marker="*",
            c="b",
            label="Source [1]" if i == 0 else "",
        )

    plt.xlabel("X Position (m)")
    plt.ylabel("Shot Number")
    plt.ylim(0, nshots + 10)
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_freq(wavpad, wavelet, old_wave, geom, freq_lim, title_right=None, title_left=None):
    """
    Plot time-domain wavelets and their frequency spectra.

    This function compares the original and filtered wavelets in both
    time and frequency domains, typically for quality control of
    source wavelet processing.

    Args:
        wavpad (int): Number of samples used for FFT padding.
        wavelet (ndarray): Filtered wavelet signal.
        old_wave (ndarray): Original (unfiltered) wavelet signal.
        geom (AcquisitionGeometry): Geometry object containing time step.
        freq_lim (float): Maximum frequency (kHz) shown in the spectrum.
        title_right (str, optional): Title for the frequency plot.
            Defaults to None.
        title_left (str, optional): Title for the time-domain plot.
            Defaults to None.

    Returns:
        None
    """
    f = np.fft.rfftfreq(wavpad, geom.dt)
    fwave = np.fft.rfft(wavelet[:wavpad], wavpad)
    old_fwave = np.fft.rfft(old_wave[:wavpad], wavpad)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

    ax[0].plot(wavelet[:], "r")
    ax[0].plot(old_wave[:], "k")
    ax[0].set_title(title_left)
    ax[0].set_xlabel("Time (ms)")
    ax[0].set_ylabel("Amplitude")

    ax[1].plot(f, np.abs(fwave), "r", label="Filtered Freq")
    ax[1].plot(f, np.abs(old_fwave), "k", label="Old Freq")
    ax[1].set_xlim(0, freq_lim)
    ax[1].set_title(title_right)
    ax[1].set_xlabel("Frequency (kHz)")
    ax[1].set_ylabel("Amplitude")

    plt.legend()
    plt.show()
