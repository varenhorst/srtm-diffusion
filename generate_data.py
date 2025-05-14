import os
import numpy as np
from PIL import Image
import srtm

OUTPUT_FOLDER = "srtm_tiles"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load SRTM data
elevation_data = srtm.get_data()

def get_elevation_tile(lat_start, lon_start, resolution=128):
    """
    Generates a grid of elevation for a 1x1 degree tile.
    `resolution`: number of samples per degree (e.g., 120 → 120x120 pixels)
    """
    tile = np.zeros((resolution, resolution), dtype=np.float32)
    lat_step = 1.0 / resolution
    lon_step = 1.0 / resolution

    for i in range(resolution):
        for j in range(resolution):
            lat = lat_start + i * lat_step
            lon = lon_start + j * lon_step
            elevation = elevation_data.get_elevation(lat, lon)
            tile[i, j] = elevation if elevation is not None else np.nan
    return tile

def save_tile_as_image(elevation_tile, filename):
    # Normalize ignoring NaNs
    mask = ~np.isnan(elevation_tile)
    min_elev = np.nanmin(elevation_tile)
    max_elev = np.nanmax(elevation_tile)
    norm = (elevation_tile - min_elev) / (max_elev - min_elev + 1e-6)
    norm[~mask] = 0  # Make NaNs black

    # Check if the image is mostly black
    non_black_pixels = np.sum(norm > 0)
    total_pixels = norm.size
    if non_black_pixels / total_pixels < 0.1:  # Adjust threshold as needed
        print(f"Tile {filename} is mostly black. Skipping save.")
        return  # Skip saving if mostly black

    img = Image.fromarray((norm * 255).astype(np.uint8))
    img.save(os.path.join(OUTPUT_FOLDER, filename))

def main():
    LAT_START = 24
    LAT_END = 50
    LON_START = -125
    LON_END = -67
    RESOLUTION = 128  # pixels per tile

    for lat in range(LAT_START, LAT_END):
        for lon in range(LON_START, LON_END):
            filename = f"tile_{lat}_{lon}.png"
            if os.path.exists(os.path.join(OUTPUT_FOLDER, filename)):
                print(f"Tile {lat}N, {lon}E already exists. Skipping...")
                continue  # Skip to the next iteration if the file exists
            print(f"Processing tile {lat}N, {lon}E...")
            tile = get_elevation_tile(lat, lon, resolution=RESOLUTION)
            save_tile_as_image(tile, filename)

if __name__ == "__main__":
    main()
