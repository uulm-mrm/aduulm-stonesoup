import matplotlib.pyplot as plt
import tikzplotlib


def plot_selfassessment(selfassessment_measures, nis_measures, alpha_nis,
                        save_plots=False, save_folder=None, sens_name='sensor'):
    """Plot Kalman Self-Assessor measures and Normalized Innovation Squared (NIS) over time.

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
    sens_name : str, optional
        Sensor name for saving files, default is 'sensor'.
    """

    # Define colors
    colors = {
        'sa_measure': (31 / 255, 119 / 255, 180 / 255),
        'threshold': (255 / 255, 127 / 255, 14 / 255),
        'uncertainty': (44 / 255, 160 / 255, 44 / 255),
        'conf_int': (255 / 255, 0 / 255, 0 / 255)
    }

    # Extract values
    delta, u_delta, theta = zip(*selfassessment_measures)
    nis, nis_lower_border, nis_upper_border = zip(*[(n, b[0], b[1]) for n, b, _ in nis_measures])

    # Create figure and axes
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    plt.subplots_adjust(hspace=0.4)

    # Plot Self-assessment measures
    ax0.set_title('Self-Assessment (SA)')
    ax0.plot(delta, label='SA measure', color=colors['sa_measure'])
    ax0.plot(theta, label='Threshold', color=colors['threshold'])
    ax0.plot(u_delta, label='Uncertainty', color=colors['uncertainty'])

    ax0.set_ylabel('Kalman SA')
    ax0.set_xlabel('Time step k')
    ax0.grid(True)
    ax0.legend(loc="upper right", bbox_to_anchor=(1, 1))

    # Plot NIS
    ax1.set_title('Normalized Innovation Squared (NIS)')
    ax1.plot(nis, label='NIS', marker='.', color=colors['sa_measure'])
    ax1.plot(nis_lower_border, label=f'{100 * (1 - alpha_nis):.0f}% conf. int.', linestyle='--', color='tab:red')
    ax1.plot(nis_upper_border, linestyle='--', color=colors['conf_int'])

    ax1.set_ylabel('NIS')
    ax1.set_xlabel('Time step k')
    ax1.grid(True)
    ax1.legend(loc="upper right", bbox_to_anchor=(1, 1))

    plt.show()

    # Save plots if needed
    if save_plots:
        if save_folder is None:
            raise ValueError("save_folder must be provided if save_plots is True.")
        fig.savefig(f'{save_folder}/selfassessor_{sens_name}.png', bbox_inches='tight')
        tikzplotlib.save(f'{save_folder}/selfassessor_{sens_name}.tex', figure=fig, encoding='utf-8')

    plt.close(fig)
