from typing import List
from osgeo import gdal
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


FDEO_DIR = os.path.dirname(os.path.dirname(__file__))


def stack_raster_months(sorted_input_files: List[str]):
    month_data = []
    for file in sorted_input_files:
        raster = gdal.Open(file)
        print('opened raster')
        data = raster.GetRasterBand(1).ReadAsArray()
        print('got data')
        month_data.append(data)

    return stack_arrays(month_data)


def stack_arrays(input_arrays: List[np.array]):
    if len(input_arrays) > 1:
        return np.stack(input_arrays, axis=2)
    return input_arrays[0].reshape((*input_arrays[0].shape, 1))


def set_tiff_resolution(input_res_array: str, input_res_geotransform: List[float], input_res_x_size: int,
                        input_res_y_size: int, target_resolution_geotransform, target_x_size: int, target_y_size: int):

    input_res_lons, input_res_lats = get_geo_locations_from_geotransform(input_res_geotransform, input_res_x_size,
                                                                         input_res_y_size)

    # Lats and lons must be ascending. For most data, the GeoTransform defines the top left corner of the data and thus
    # the lats will be descending. If this is the case, sort the lats and flip the data about the "x" axis
    if input_res_lats[0] > input_res_lats[1]:
        print('Flipping')
        input_res_lats = sorted(input_res_lats)
        input_res_array = np.flip(input_res_array, axis=0)  # Flip about the 'lon' axis for testing

    print(input_res_array.shape, 'ira')

    print(input_res_lats[0], input_res_lats[-1], input_res_lons[0], input_res_lons[-1])

    input_res_interp = RegularGridInterpolator((input_res_lats, input_res_lons),
                                               input_res_array, method='linear', bounds_error=False)

    target_res_lons, target_res_lats = get_geo_locations_from_geotransform(target_resolution_geotransform,
                                                                           target_x_size, target_y_size)

    print(target_res_lats[0], target_res_lats[-1], target_res_lons[0], target_res_lons[-1])

    target_lat_mesh, target_lon_res = np.meshgrid(target_res_lats, target_res_lons, indexing='ij')

    points = np.array([target_lat_mesh.flatten(), target_lon_res.flatten()]).T

    target_res_data = input_res_interp(points).reshape((len(target_res_lats), len(target_res_lons)))

    return target_res_data


def get_geo_locations_from_tif(raster):
    # Get geolocation information
    geo_transform = raster.GetGeoTransform()
    x_size = geo_transform[1]
    y_size = geo_transform[5]
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]

    # Get geolocation of each data point
    lats = []
    for row in range(raster.RasterYSize):
        lats.append(y_origin + (row * y_size))

    lons = []
    for col in range(raster.RasterXSize):
        lons.append(x_origin + (col * x_size))

    return lons, lats


def get_geo_locations_from_geotransform(geo_transform, num_cols: int, num_rows: int):
    # Get geolocation information
    x_res = geo_transform[1]
    y_res = geo_transform[5]
    x_origin = geo_transform[0]
    y_origin = geo_transform[3]

    # Get geolocation of each data point
    lats = []
    for row in range(num_rows):
        lats.append(y_origin + (row * y_res))

    lons = []
    for col in range(num_cols):
        lons.append(x_origin + (col * x_res))

    return lons, lats


def plot_file(input_file, months: int, year_to_plot: int, month_to_plot: int):
    month_titles = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    lc1 = np.loadtxt(os.path.join(FDEO_DIR, 'data', 'lc1.csv'), delimiter=",")

    val = np.loadtxt(input_file)

    # reshape the loaded array back to its original shape
    val = val.reshape((112, 244, months))

    for i in range(112):
        for j in range(244):
            if (lc1[i][j] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
                val[i][j] = float("NaN")

    # month to graph histograms
    mo_index = (year_to_plot - 1) * 12 + month_to_plot - 1

    # plot probabilities of observations
    val_split = np.dsplit(val, months)
    val = val_split[mo_index]
    val = val.reshape((112, 244))
    # exclude LC types out of the scope of the study
    for i in range(112):
        for j in range(244):
            if (lc1[i][j] != 4) & (lc1[i][j] != 5) & (lc1[i][j] != 7) & (lc1[i][j] != 8) & (lc1[i][j] != 10):
                val[i][j] = float("NaN")

    cmap = ListedColormap(['green', 'white', 'red'])
    plt.imshow(val, cmap=cmap)
    plt.xlabel(month_titles[month_to_plot - 1])
    plt.show()

    # val = np.rot90(val.T)
    # fig, (fig1) = plt.subplots(1, 1)
    # fig1.pcolor(val)
    cmap = ListedColormap(['green', 'blue', 'red'])
    plt.imshow(np.squeeze(val[:, :, 0]), cmap=cmap)
    plt.show()

def pretty_plot(input_tiff: str, title: str, xlabel: str, ylabel: str, save_filename: str):
    tiff_file = gdal.Open(input_tiff)
    data = tiff_file.GetRasterBand(1).ReadAsArray()
    data = np.clip(data, 0, np.inf)
    min = np.min(data.flatten())
    max = np.max(data.flatten())

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Set the color map
    cmap = plt.cm.get_cmap('viridis')  # Choose the desired colormap

    # Plot the array
    im = ax.imshow(np.flipud(data), cmap=cmap, vmin=min, vmax=max, origin='lower')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Add a colorbar
    cbar = fig.colorbar(im)

    fig.savefig(save_filename, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()