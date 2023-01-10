import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def visualize_proj(proj, title=None, mode: str='F'):
    if type(proj) is torch.Tensor:
        proj_np = proj.numpy()
    else:
        proj_np = proj
    proj_range_vis = np.copy(proj_np)
    proj_range_vis = proj_range_vis.astype(np.float32)
    proj_range_vis /= np.amax(proj_range_vis) / 255.0
    img = Image.fromarray(proj_range_vis, mode=mode)
    img.show(title)

def visualize_semantics(labels, title=None):
    if type(labels) is torch.Tensor:
        labels_np = labels.numpy()
    else:
        labels_np = labels
    labels_np *= 255.0
    labels_np = labels_np.astype(np.uint8)
    img = Image.fromarray(labels_np, mode='RGB')
    img.show(title)

def visualize_histogram(data, title='Remissions', bins: int=30, plot_dist=False, discard_values=[0]):
    if type(data) is torch.Tensor:
        data_np = np.copy(data.numpy())
    else:
        data_np = np.copy(data)

    # Discard 0's and 1's for statistics
    if discard_values:
        for i, val in enumerate(discard_values):
            data_np = data_np[data_np != val]

    # always discard max value
    data_np = data_np[data_np != np.amax(data_np)]

    mu, std = norm.fit(data_np)
    print(f"{title}. mu: {mu}, std: {std}")
    plt.hist(data_np, density=True, bins=bins)
    plt.ylabel(f'{title} distribution')
    plt.xlabel('Data')
    if plot_dist:
        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title("{} histogram. Fit Values: {:.2f} and {:.2f}".format(title, mu, std))
    else:
        plt.title("{} histogram".format(title))
    plt.show()

    return mu, std