"""
Script for generating GIFs from opm data
09/02/2025
"""

import numpy as np
import imageio

def scale_to_uint8(
    arr: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    gamma: float = 1.0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Scale an array to uint8 with adjustable contrast.

    Parameters
    ----------
    arr : ndarray
        Input image data (any dtype).
    vmin, vmax : float, optional
        Intensity window to map -> 0..255. If None, min/max of data are used.
    gamma : float
        Gamma correction (1.0 = none, <1 brighter shadows, >1 darker shadows).
    contrast : float
        Contrast factor (>1 increases, <1 decreases).
    """
    a = arr.astype(np.float32, copy=False)

    # pick window
    if vmin is None:
        vmin = float(np.nanmin(a))
    if vmax is None:
        vmax = float(np.nanmax(a))

    # apply window
    a = (a - vmin) / max(vmax - vmin, 1e-12)
    a = np.clip(a, 0, 1)

    # gamma correction
    if gamma != 1.0:
        a = np.power(a, gamma)

    # contrast stretch around 0.5
    if contrast != 1.0:
        a = 0.5 + contrast * (a - 0.5)
        a = np.clip(a, 0, 1)

    return (a * 255).astype(np.uint8)

def save_gif(arr: np.ndarray,
             path: str,
             fps: int = 10,
             duration_ms: int = 1000,
             loop: int = 0,
             palettesize: int = 256) -> None:
    """
    Save a NumPy array as an animated GIF.

    Parameters
    ----------
    arr : np.ndarray
        Shape (T,H,W) for grayscale or (T,H,W,C) with C in {1,3,4}.
        Channels-first (T,C,H,W) is also accepted.
        Dtype can be uint8, int, or float. Floats are assumed to be 0..1
        (if max<=1.0) and are scaled to 0..255; otherwise they are clipped.
    path : str
        Output filename, e.g. 'out.gif'.
    fps : int
        Frames per second (controls speed).
    loop : int
        Number of loops for the GIF. 0 means loop forever.
    palettesize : int
        Color palette size (2..256). 256 usually looks best.

    Returns
    -------
    None
    """
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D array, got shape {arr.shape}")
        
    a = scale_to_uint8(arr=arr, gamma=0.8, contrast=1.05)

    # Arrange dimensions
    if a.ndim == 3:
        # (T,H,W) grayscale — fine as-is
        frames = a
    else:
        # 4D: either (T,H,W,C) or (T,C,H,W)
        if a.shape[-1] in (1, 3, 4):          # channels last
            frames = a
        elif a.shape[1] in (1, 3, 4):         # channels first -> move to last
            frames = np.transpose(a, (0, 2, 3, 1))
        else:
            raise ValueError(f"Can't infer channel dim for shape {a.shape}")

        # If single-channel, squeeze to grayscale
        if frames.shape[-1] == 1:
            frames = frames[..., 0]

    # imageio expects a sequence of frames shaped (H,W) or (H,W,3/4)
    duration = 1.0 / float(fps)
    imageio.mimsave(
        uri=path,
        ims=list(frames),
        format="GIF",
        duration=duration_ms,
        loop=loop,
        palettesize=palettesize,
    )

# ---- Example usage ----
if __name__ == "__main__":
    
    """
    Put our own code to load data here and call function
    """
    from pathlib import Path
    import zarr
    import tensorstore as ts
    from opm_v2.utils import sensorless_ao as ao

    """Make a gif from OPM data"""
    zarr_path = Path('/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/in_IB_noAO_max_z_decon_deskewed.zarr')

    # open raw datastore
    spec = {
        "driver" : "zarr3",
        "kvstore" : {
            "driver" : "file",
            "path" : str(zarr_path)
        }
    }
    # datastore = ts.open(spec).result()[0, :-1, 0, 0, 500:600, 1400:1500]
    datastore = ts.open(spec).result()[0, :, 0, 0, :, :]
    data = datastore.read().result()
    data = data[:-1, 950:1025,1000:1075]
    save_gif(data, "paramecium_wo_AO_zoom.gif", duration_ms=1500)
    
    """Make a gif from A.O. results"""
    # data_path = Path(
    #     r'/home/steven/Documents/qi2lab/projects/local_working_files/OPM/opm_ao/2025 828_161621_ao_optimizeNOW/ao_results.zarr'
    # )

    # results = ao.load_optimization_results(data_path)

    # all_images = results['all_images']
    # all_metrics = results['all_metrics']
    # metrics_per_iteration = results['metrics_per_iteration']
    # images_per_iteration = results['images_per_iteration']

    # save_gif(images_per_iteration, "images_.gif", duration_ms=500)
    