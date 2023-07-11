import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os

FDEO_DIR = os.path.dirname(os.path.dirname(__file__))

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
    print(mo_index)
    # plot probabilities of observations
    val = val[:, :, mo_index]
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