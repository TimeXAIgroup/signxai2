from math import ceil

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator

from utils.ecg_data import perform_shape_switch


def plot_ecg(
        ecg,
        explanation=None,
        sampling_rate=500,
        title='ECG',
        lead_index=None,
        columns=2,
        row_height=6,
        show_lead_name=True,
        show_grid=True,
        show_separate_line=True,
        show_colorbar=False,
        bubble_size=40,
        line_width=1,
        style='fancy',
        save_to=None,
        cmap='seismic',
        clim_min=-1,
        clim_max=1,
        colorbar_label=None,
        colorbar_tickvalues=None,
        shape_switch=True,
        dpi=300
):
    """ Code based on https://github.com/dy1901/ecg_plot """

    try:
        mpl.colormaps.register(LinearSegmentedColormap.from_list('FairReds', [(1, 1, 1), (1, 0, 0)], N=256), name='FairReds')
        mpl.colormaps.register(LinearSegmentedColormap.from_list('FairBlues', [(0, 0, 1), (1, 1, 1)], N=256), name='FairBlues')
        mpl.colormaps.register(LinearSegmentedColormap.from_list('BlackToRed', [(0, 0, 0), (1, 0, 0)], N=256), name='BlackToRed')
        mpl.colormaps.register(LinearSegmentedColormap.from_list('BlueToBlack', [(0, 0, 1), (0, 0, 0)], N=256), name='BlueToBlack')
        mpl.colormaps.register(LinearSegmentedColormap.from_list('BlueBlackRed', [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0)], N=256), name='BlueBlackRed')
    except ValueError:
        pass

    if shape_switch:
        ecg = perform_shape_switch(ecg)

        if explanation is not None:
            explanation = perform_shape_switch(explanation)

    x_stride = 0.2
    y_stride = 0.5
    display_factor = 1.25

    lead_order = list(range(0, len(ecg)))
    secs = len(ecg[0]) / sampling_rate
    leads = len(lead_order)
    rows = ceil(leads / columns)

    if bubble_size is None:
        bubble_size = 20 * line_width,

    if lead_index is None:
        lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    fig, ax = plt.subplots(figsize=(int(secs * columns * display_factor), int(rows * row_height / 5 * display_factor)))
    fig.subplots_adjust(
        hspace=0,
        wspace=0,
        left=0,
        right=1,
        bottom=0,
        top=1
    )

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    if style == 'red':
        color_major = (1, 0, 0)
        color_minor = (1, 0.7, 0.7)
        color_line = (0, 0, 0)
        color_sep_line = (0, 0, 0)
        color_text = (0, 0, 0)
    elif style == 'fancy':
        color_major = (0.8, 0.8, 0.8)
        color_minor = (0.9, 0.9, 0.9)
        color_line = (0, 0, 0)
        color_sep_line = (0, 0, 0)
        color_text = (0, 0, 0)
    else:
        color_major = (0.4, 0.4, 0.4)
        color_minor = (0.75, 0.75, 0.75)
        color_line = (0, 0, 0)
        color_sep_line = (0, 0, 0)
        color_text = (0, 0, 0)

    fig.suptitle(title, y=0.995, color=color_text)

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, x_stride))
        ax.set_yticks(np.arange(y_min, y_max, y_stride))

        ax.minorticks_on()

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))

        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    for c in range(0, columns):
        for i in range(0, rows):
            if c * rows + i < leads:
                y_offset = -(row_height / 2) * ceil(i % rows)
                x_offset = 0

                if c > 0:
                    x_offset = secs * c

                    if show_separate_line:
                        ax.plot([x_offset, x_offset], [ecg[t_lead][0] + y_offset - 0.5, ecg[t_lead][0] + y_offset + 0.5], linewidth=line_width * display_factor, color=color_sep_line)

                t_lead = lead_order[c * rows + i]
                step = 1.0 / sampling_rate

                if show_lead_name:
                    ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor, color=color_text)

                if explanation is None:
                    ax.plot(np.arange(0, round(len(ecg[t_lead]) * step, 3), step) + x_offset,
                            ecg[t_lead] + y_offset,
                            linewidth=line_width,
                            color=color_line)
                else:
                    colors_lead = explanation[t_lead]
                    plot_lead_bubbles(
                        x=np.arange(0, round(len(ecg[t_lead]) * step, 3), step) + x_offset,
                        y=ecg[t_lead] + y_offset,
                        z=colors_lead,
                        ax=ax,
                        linewidth=line_width,
                        bubble_size=bubble_size,
                        color_line=color_line,
                        clim_min=clim_min,
                        clim_max=clim_max
                    )

    # Add bottom label
    ax.text(0.07, y_offset - 1.2, '25 mm/s, 10 mm/mV, recording length: {:.1f} s'.format(secs), fontsize=9 * display_factor, color=color_text)

    # Add colorbar at bottom
    if show_colorbar:
        if colorbar_tickvalues is not None:
            colorbar_ticklabels = {x: '{:.1f} s'.format(x / 500) for x in colorbar_tickvalues}
            ticks = list(colorbar_ticklabels.keys())
        else:
            ticks = None

        cax = fig.add_axes([0.875, 0.027, 0.1, 0.0075])
        cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin=clim_min, vmax=clim_max), cmap=cmap), orientation='horizontal', cax=cax, ticks=ticks)
        cb.ax.tick_params(labelsize=6 * display_factor)
        cb.set_label(colorbar_label, labelpad=-25, fontsize=7 * display_factor)

        if colorbar_tickvalues is not None:
            cb.ax.set_xticklabels([colorbar_ticklabels[k] for k in colorbar_ticklabels])

    if save_to is not None:
        plt.savefig(save_to, dpi=dpi)
        plt.close()
    else:
        plt.show()
        plt.close()

    return save_to


def plot_lead_bubbles(x, y, z, ax, linewidth, bubble_size=1.0, color_line=(0, 0, 0), clim_min=None, clim_max=None):
    ax.plot(x, y, linewidth=linewidth, color=color_line, zorder=3)

    x_dots_pos = []
    y_dots_pos = []
    z_dots_pos = []
    x_dots_neg = []
    y_dots_neg = []
    z_dots_neg = []

    for vz, vx, vy in sorted(zip(z, x, y)):
        # if vz > 0.01:
        if vz > 0:
            x_dots_pos.append(vx)
            y_dots_pos.append(vy)
            z_dots_pos.append(vz)

    for vz, vx, vy in sorted(zip(z, x, y), reverse=True):
        # if vz < -0.01:
        if vz < 0:
            x_dots_neg.append(vx)
            y_dots_neg.append(vy)
            z_dots_neg.append(vz)

    if len(z_dots_neg) > 0:
        if clim_min is None:
            bubble_sizes_neg = (np.abs(z_dots_neg) / np.max(np.abs(z_dots_neg))) * bubble_size
        else:
            bubble_sizes_neg = (np.abs(z_dots_neg) / np.abs(clim_min)) * bubble_size

        ax.scatter(x_dots_neg, y_dots_neg, marker='o', c=z_dots_neg, cmap='Blues_r', s=bubble_sizes_neg, zorder=2, vmin=clim_min, vmax=0)

    if len(z_dots_pos) > 0:
        if clim_max is None and len(z_dots_pos) > 0:
            bubble_sizes_pos = (np.abs(z_dots_pos) / np.max(np.abs(z_dots_pos))) * bubble_size
        else:
            bubble_sizes_pos = (np.abs(z_dots_pos) / np.abs(clim_max)) * bubble_size

        ax.scatter(x_dots_pos, y_dots_pos, marker='o', c=z_dots_pos, cmap='FairReds', s=bubble_sizes_pos, zorder=2, vmin=0, vmax=clim_max)
