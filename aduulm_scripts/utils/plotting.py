import matplotlib.pyplot as plt
# import tikzplotlib
import numpy as np
from fontTools.unicodedata import block


def plot_selfassessment_with_nis(
    selfassessment_measures,
    nis_measures,
    alpha_nis,
    save_plots=False,
    save_folder=None,
    sensor_name='',
    opinion_type='',
    assessor_type='',
    ylabels=None,
    titles=None,
):
    """Plot Self-Assessor measures and Normalized Innovation Squared (NIS).

    Parameters
    ----------
    selfassessment_measures : list of tuples
        Contains (delta, u_delta, theta) values over time.
    nis_measures : list of tuples
        Contains (nis, (nis_lower_border, nis_upper_border)) values over time.
    alpha_nis : float
        Desired confidence level for NIS borders.
    save_plots : bool, optional
        Whether to save the plots, by default False.
    save_folder : str, optional
        Folder where plots should be saved, required if save_plots=True.
    sensor_name : str, optional
        Sensor name for saving files, default is '' empty.
    opinion_type : str, optional
        Type of opinion (e.g., "detection", "association"), default ''.
    assessor_type : str, optional
        SelfAssessor type (e.g., "NNSelfAssessor"), default ''.
    ylabels : dict, optional
        Custom y-axis labels. Example: {"sa": "Kalman SA", "nis": "NIS"}.
    titles : dict, optional
        Custom titles. Example: {"sa": "Detection SA", "nis": "NIS for detection"}.
    """

    # Default axis labels
    if ylabels is None:
        ylabels = {"sa": "SA measures", "nis": "NIS"}
    # Default titles if not provided
    if titles is None:
        name_parts = []
        if assessor_type:
            name_parts.append(assessor_type)
        if opinion_type:
            name_parts.append(opinion_type)
        if sensor_name:
            name_parts.append(sensor_name.capitalize())
        suffix = ": ".join(name_parts) if name_parts else ""

        titles = {
            "sa": f"Self-Assessment (SA){' - ' + suffix if suffix else ''}",
            "nis": f"Normalized Innovation Squared (NIS)",
        }

    # Define colors
    colors = {
        'sa_measure': (31 / 255, 119 / 255, 180 / 255),
        'threshold': (255 / 255, 127 / 255, 14 / 255),
        'uncertainty': (44 / 255, 160 / 255, 44 / 255),
        'conf_int': (255 / 255, 0 / 255, 0 / 255)
    }

    # Extract values
    delta, u_delta, theta = zip(*selfassessment_measures)
    nis, nis_lower_border, nis_upper_border = zip(
        *[(n, b[0], b[1]) for n, b, _ in nis_measures]
    )

    # Create figure and axes
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    # ---- SA plot ----
    ax0.set_title(titles["sa"])
    ax0.plot(delta, label='SA measure', color=colors['sa_measure'])
    ax0.plot(theta, label='Threshold', color=colors['threshold'])
    ax0.plot(u_delta, label='Uncertainty', color=colors['uncertainty'])
    ax0.set_ylabel(ylabels["sa"])
    ax0.set_xlabel('Time step k')
    ax0.grid(True)
    ax0.legend(loc="upper right")
    ax0.tick_params(labelbottom=True)

    # ---- NIS plot ----
    ax1.set_title(titles["nis"])
    ax1.plot(nis, label='NIS', marker='.', color=colors['sa_measure'])
    ax1.plot(nis_lower_border,
             label=f'{100 * (1 - alpha_nis):.0f}% conf. int.',
             linestyle='--', color='tab:red')
    ax1.plot(nis_upper_border, linestyle='--', color=colors['conf_int'])
    ax1.set_ylabel(ylabels["nis"])
    ax1.set_xlabel('Time step k')
    ax1.grid(True)
    ax1.legend(loc="upper right")

    # Save if requested
    if save_plots:
        if save_folder is None:
            raise ValueError("save_folder must be provided if save_plots is True.")

        base_name = "_".join(filter(None, [assessor_type, opinion_type, sensor_name]))
        if not base_name:
            base_name = "selfassessor"

        fig.savefig(f'{save_folder}/{base_name}.png', bbox_inches='tight')
        # tikzplotlib.save(f'{save_folder}/{base_name}.tex', figure=fig, encoding='utf-8')


    return fig


def plot_multisensor_selfassessment(selfassessment_measures,
                                    assessor_type='',
                                    save_plots=False,
                                    save_folder=None):
    """Plot Multi-Sensor Kalman Self-Assessor measures over time (without NIS).

    Parameters
    ----------
    selfassessment_measures : list of tuples
        Contains (delta, u_delta, theta) values over time.
    assessor_type : str, optional
        SelfAssessor type (e.g., "NNSelfAssessor"), default ''.
    save_plots : bool, optional
        Whether to save the plots, by default False.
    save_folder : str, optional
        Folder where plots should be saved, required if save_plots=True.
    """

    # Define colors
    colors = {
        'sa_measure': (31 / 255, 119 / 255, 180 / 255),
        'threshold': (255 / 255, 127 / 255, 14 / 255),
        'uncertainty': (44 / 255, 160 / 255, 44 / 255),
    }

    # Extract values
    delta, u_delta, theta = zip(*selfassessment_measures)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot Self-assessment measures
    title_prefix = f" - {assessor_type}" if assessor_type else ""
    ax.set_title(
        f"Multi-Sensor Overall Self-Assessment (SA){title_prefix if title_prefix else ''}"
    )
    ax.plot(delta, label='SA measure', color=colors['sa_measure'])
    ax.plot(theta, label='Threshold', color=colors['threshold'])
    ax.plot(u_delta, label='Uncertainty', color=colors['uncertainty'])

    ax.set_ylabel('SA measures')
    ax.set_xlabel('Time step k')
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 1))

    plt.show()

    # Save plots if needed
    if save_plots:
        if save_folder is None:
            raise ValueError("save_folder must be provided if save_plots is True.")
        fig.savefig(f'{save_folder}/multisensor_selfassessor.png', bbox_inches='tight')
        # tikzplotlib.save(f'{save_folder}/multisensor_selfassessor.tex', figure=fig, encoding='utf-8')

    plt.close(fig)


def plot_multiple_selfassessment_measures(measures_dict, assessor_type='', save_plots=False, save_folder=None):
    """Plot multiple Self-Assessor measures over time.

    Each measure is plotted in a separate figure and (optionally) saved.
    For 'threshold' and 'association_situation', plots the projected probability only.

    Parameters
    ----------
    measures_dict : dict
        Dictionary of SA measures. Keys are measure names (str),
        values are lists of tuples (delta, u_delta, theta).
    assessor_type : str, optional
        SelfAssessor type (e.g., "NNSelfAssessor"), default ''.
    save_plots : bool, optional
        Whether to save the plots. Default: False.
    save_folder : str, optional
        Folder where plots should be saved. Required if save_plots=True.
    """

    # Define consistent colors
    colors = {
        'sa_measure': (31 / 255, 119 / 255, 180 / 255),
        'threshold': (255 / 255, 127 / 255, 14 / 255),
        'uncertainty': (44 / 255, 160 / 255, 44 / 255),
        'proj_prob': (214 / 255, 39 / 255, 40 / 255)
    }

    # Default title prefix
    title_prefix = f"{assessor_type}: " if assessor_type else "SelfAssessor: "

    for measure_name, values in measures_dict.items():
        if not values:
            print(f"Skipping {measure_name}: no values provided.")
            continue

        # Create individual figure
        fig, ax = plt.subplots(figsize=(8, 4))

        # Check if this is a projected probability measure
        if measure_name.lower() in {"threshold", "association_situation"}:
            # Plot projected probability
            ax.plot(values, label='Projected probability', color=colors['proj_prob'])
            ax.set_ylabel('Projected probability')
            ax.set_title(f"Self-Assessment (SA) - {title_prefix}{measure_name.capitalize()} Opinion (Projected Prob.)")
        else:
            # Unpack tuples into arrays
            delta, u_delta, theta = zip(*values)
            # Plot SA measures
            ax.plot(delta, label='SA measure', color=colors['sa_measure'])
            ax.plot(theta, label='Threshold', color=colors['threshold'])
            ax.plot(u_delta, label='Uncertainty', color=colors['uncertainty'])
            ax.set_ylabel('SA measures')
            ax.set_title(f"Self-Assessment (SA) - {title_prefix}{measure_name.capitalize()} Opinion")

        # Common labels & grid
        ax.set_xlabel('Time step k')
        ax.grid(True)
        ax.legend(loc="upper right")

        # Show figure
        plt.show()

        # Save if requested
        if save_plots:
            if save_folder is None:
                raise ValueError("save_folder must be provided if save_plots is True.")
            fig.savefig(f"{save_folder}/selfassessor_{measure_name}.png", bbox_inches='tight')
            # tikzplotlib.save(f"{save_folder}/selfassessor_{measure_name}.tex",
            #                  figure=fig, encoding='utf-8')

        plt.close(fig)


def plot_gospa(gospa_metric, folder_name=None, save_name="gospa", average=False, save_plots=False):
    """
    Plot the GOSPA metric over time from StoneSoup TimeRangeMetric object(s).

    Parameters
    ----------
    gospa_metric : TimeRangeMetric or list of TimeRangeMetric
        GOSPA metric object(s) from StoneSoup. Each timestep should have attributes distance, localisation, missed, false.
    folder_name : str, optional
        Folder where plots are saved if save_plots=True.
    save_name : str, optional
        Name for the saved files, default "gospa".
    average : bool, optional
        If True, computes average over multiple TimeRangeMetric objects.
    save_plots : bool, optional
        Whether to save PNG and TikZ files.
    """

    # Ensure we have a list of metrics for averaging
    metrics_list = gospa_metric if isinstance(gospa_metric, list) else [gospa_metric]

    # Convert TimeRangeMetric to list of dicts per timestep
    all_runs_values = []
    for metric in metrics_list:
        run_values = []
        for step in metric.value:
            # Each step.value is expected to be a dict with keys: distance, localisation, missed, false
            run_values.append(step.value)
        all_runs_values.append(run_values)

    # Helper to convert list-of-dicts to arrays
    def extract_arrays(run_values):
        distance = np.array([step["distance"] for step in run_values])
        localisation = np.array([step["localisation"] for step in run_values])
        missed = np.array([step["missed"] for step in run_values])
        false = np.array([step["false"] for step in run_values])
        return distance, localisation, missed, false

    # Aggregate / average if needed
    if average:
        distance, localisation, missed, false = extract_arrays(all_runs_values[0])
        for run_idx in range(1, len(all_runs_values)):
            d_tmp, l_tmp, m_tmp, f_tmp = extract_arrays(all_runs_values[run_idx])
            distance += d_tmp
            localisation += l_tmp
            missed += m_tmp
            false += f_tmp
        distance /= len(all_runs_values)
        localisation /= len(all_runs_values)
        missed /= len(all_runs_values)
        false /= len(all_runs_values)
    else:
        distance, localisation, missed, false = extract_arrays(all_runs_values[0])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(distance)), distance, label="Distance")
    ax.plot(np.arange(len(localisation)), localisation, label="Localisation")
    ax.plot(np.arange(len(missed)), missed, label="Missed")
    ax.plot(np.arange(len(false)), false, label="False")

    ax.set_xlabel("Time step k")
    ax.set_ylabel("GOSPA metric")
    ax.set_title("Generalized Optimal Sub-pattern Assignment (GOSPA)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    if save_plots:
        if folder_name is None:
            raise ValueError("folder_name must be provided if save_plots is True.")
        fig.savefig(f"{folder_name}/{save_name}.png", bbox_inches='tight')
        # tikzplotlib.save(f"{folder_name}/{save_name}.tex", figure=fig, encoding='utf-8')

    plt.close(fig)
