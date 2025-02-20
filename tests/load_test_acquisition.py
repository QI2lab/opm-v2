import zarr
from pathlib import Path
import numpy as np
import json
import matplotlib.pyplot as plt

zarr_folder_path = Path(r"G:\20250220_test\opm_ao_mda_test.ome.zarr\data.zarr")
positions_path = Path(r"G:\20250220_test\opm_ao_mda_test.ome.zarr\exp_ao_positions.json")

with open(positions_path, "r") as f:
    positions_list = json.load(f)
positions = np.asarray(positions_list)
print(positions.shape)

images = zarr.open(zarr_folder_path, mode="r")

for img in images:
    plt.figure()
    plt.imshow(img)

plt.show()