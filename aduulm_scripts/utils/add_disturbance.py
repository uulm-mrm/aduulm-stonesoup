import warnings
import numpy as np


def disturbance_transition_model(transition_model, gt_configs, k):
    """
    Apply disturbance effects ("jump" or "drift") to a ground truth transition model.

    This function modifies the noise characteristics of the given transition model
    at runtime based on disturbance configurations. The disturbances can simulate
    sudden changes ("jump") or gradual changes ("drift") in process noise.

    Parameters
    ----------
    transition_model : CombinedLinearGaussianTransitionModel
        The current ground truth transition model, consisting of a list of sub-models
        with modifiable `noise_diff_coeff` attributes.

    gt_configs : dict
        Ground truth disturbance configuration with the following keys:
        - "disturbance_mode" (list[str]): List of disturbance types per model.
          Supported values are:
            * "none" : no disturbance applied
            * "jump" : sudden change in process noise
            * "drift" : gradual change in process noise
        - "parameters" (list[list[tuple]]): Parameters for each disturbance mode.
          Each entry corresponds to one mode in "disturbance_mode". Examples:
            * For "jump": [(time_step, multiplier), ...]
            * For "drift": [(start_step, end_step, drift_factor), ...]

    k : int
        Current time step.

    Returns
    -------
    CombinedLinearGaussianTransitionModel
        The updated transition model with applied disturbance effects.

    Notes
    -----
    - For "jump", at the specified time step, the `noise_diff_coeff` is multiplied
      by the given multiplier.
    - For "drift", between `start_step < k <= end_step`, the `noise_diff_coeff`
      is gradually scaled using the provided drift factor.
    - Disturbance modes not recognized will trigger a warning.
    """
    for idx_dist_mode, dist_mode in enumerate(gt_configs['disturbance_mode']):
        if dist_mode.lower() == 'none':
            continue  # No disturbance applied

        elif dist_mode.lower() == 'jump':
            for z in range(len(gt_configs['parameters'][idx_dist_mode])):
                if k == gt_configs['parameters'][idx_dist_mode][z][0]:
                    # Apply jump disturbance
                    for i, model in enumerate(transition_model.model_list):
                        model.noise_diff_coeff *= gt_configs['parameters'][idx_dist_mode][z][1] if gt_configs['disturb_noise_coeff'][i] else 1

        elif dist_mode.lower() == 'drift':
            for z in range(len(gt_configs['parameters'][idx_dist_mode])):
                if gt_configs['parameters'][idx_dist_mode][z][0] < k <= gt_configs['parameters'][idx_dist_mode][z][1]:
                    dq = gt_configs['parameters'][idx_dist_mode][z][2] - 1
                    nsteps_drift = (
                            gt_configs['parameters'][idx_dist_mode][z][1] -
                            gt_configs['parameters'][idx_dist_mode][z][0]
                    )
                    k_drift = k - gt_configs['parameters'][idx_dist_mode][z][0]

                    for model in transition_model.model_list:
                        model.noise_diff_coeff *= (
                                (nsteps_drift + k_drift * dq) /
                                (nsteps_drift + (k_drift - 1) * dq)
                        )

        else:
            warnings.warn(
                f"Ground truth disturbance mode '{dist_mode}' in configuration "
                f"does not match valid options ('none', 'jump', 'drift')."
            )

    return transition_model


def disturbance_measurement_noise(measurement_model, gt_configs, k):
    """
    Apply disturbances to a measurement model's noise covariance based on a disturbance configuration.

    This function modifies the measurement model covariance (`noise_covar`)
    dynamically according to specified disturbance modes such as sudden jumps,
    gradual drifts, periodic outliers, or randomly varying noise.

    Parameters
    ----------
    measurement_model : MeasurementModel
        The measurement model whose noise covariance will be modified.
        Must have a `noise_covar` attribute (numpy array).

    gt_configs : dict
        Ground truth disturbance configuration with the following keys:
        - "disturbance_mode" (list[str]): List of disturbance types applied to the model.
          Supported values are:
            * "none"        : no disturbance
            * "jump"        : sudden multiplicative change in covariance
            * "drift"       : gradual multiplicative change in covariance over time
            * "outliers"    : periodic spikes in noise covariance
            * "noisy_noise" : randomly varying covariance
        - "parameters" (list[list[tuple]]): Parameters for each disturbance mode.
          Each entry corresponds to one mode in "disturbance_mode". Examples:
            * For "jump": [(time_step, multiplier), ...]
            * For "drift": [(start_step, end_step, drift_factor), ...]
            * For "outliers": [(start_step, end_step, period, multiplier), ...]
            * For "noisy_noise": [(start_step, end_step, min_factor, max_factor), ...]

    k : int
        Current time step.

    Returns
    -------
    MeasurementModel
        Updated measurement model with disturbances applied.

    Notes
    -----
    - Covariance scaling is always multiplicative.
    - "jump" applies the multiplier squared (`multiplier^2`).
    - "drift" gradually modifies the covariance between the start and end step.
    - "outliers" applies periodic spikes (multiplying/dividing by factor).
    - "noisy_noise" samples a random factor from a uniform distribution.
    - Invalid disturbance modes trigger a warning.

    """
    measurement_model.noise_covar = measurement_model.noise_covar.astype(float)  # Ensure float type

    for idx_dist_mode, dist_mode in enumerate(gt_configs['disturbance_mode']):
        if dist_mode.lower() == 'none':
            pass

        elif dist_mode.lower() == 'jump':
            for z in gt_configs['parameters'][idx_dist_mode]:
                if k == z[0]:
                    for i, comp in enumerate(z[2]):
                        if comp:
                            if comp == 1:
                                measurement_model.noise_covar[i, i] *= pow(z[1], 2)
                            elif comp==2:
                                measurement_model.noise_covar[i, i] *= pow(1/z[1], 2)
                            # print(measurement_model.noise_covar[i, i])

        elif dist_mode.lower() == 'drift':
            for z in gt_configs['parameters'][idx_dist_mode]:
                if z[0] < k <= z[1]:
                    dq = z[2] - 1
                    nsteps_drift = z[1] - z[0]
                    k_drift = k - z[0]
                    factor = (nsteps_drift + k_drift * dq) / (nsteps_drift + (k_drift - 1) * dq)
                    measurement_model.noise_covar = pow(np.sqrt(measurement_model.noise_covar) * factor, 2)

        elif dist_mode.lower() == 'outliers':
            for z in gt_configs['parameters'][idx_dist_mode]:
                if z[0] <= k < z[1]:
                    period = z[2]
                    factor = pow(z[3], 2)
                    if (k - z[0]) % period == 0:
                        measurement_model.noise_covar *= factor
                    elif (k - z[0]) % period == 1:
                        measurement_model.noise_covar /= factor

        elif dist_mode.lower() == 'noisy_noise':
            for z in gt_configs['parameters'][idx_dist_mode]:
                if z[0] < k <= z[1]:
                    factor = np.random.uniform(pow(z[2], 2), pow(z[3], 2))
                    measurement_model.noise_covar *= factor

        else:
            warnings.warn(
                f"Invalid disturbance mode '{dist_mode}'. "
                "Valid options: 'none', 'jump', 'drift', 'outliers', 'noisy_noise'."
            )

    return measurement_model


def disturbance_prob_det_and_clutter(gt_configs, k, manipulation_value):
    """
    Apply disturbances (e.g., 'jump', 'drift') to scalar parameters such as
    expected clutter rate or probability of detection.

    Parameters
    ----------
    gt_configs : dict
        Ground truth disturbance configuration with the following structure:
            - 'disturbance_mode' : list of str
                Disturbance modes, e.g., ['jump', 'drift'].
            - 'parameters' : list of lists
                Parameters for each disturbance mode:
                    * 'jump'  : [(timestep, factor), ...]
                    * 'drift' : [(start, end, drift_factor), ...]
    k : int
        Current time step.
    manipulation_value : float
        Current scalar value (e.g., expected clutter rate or probability of detection).

    Returns
    -------
    float
        Updated scalar value after applying disturbances.
    """

    for idx, mode in enumerate(gt_configs['disturbance_mode']):
        mode = mode.lower()

        if mode == 'none':
            continue

        elif mode == 'jump':
            for timestep, factor in gt_configs['parameters'][idx]:
                if k == timestep:
                    manipulation_value *= factor

        elif mode == 'drift':
            for start, end, drift_factor in gt_configs['parameters'][idx]:
                if start < k <= end:
                    # Drift requires recursive scaling since the initial value evolves each step
                    dq = drift_factor - 1  # incremental factor per step
                    nsteps_drift = end - start
                    k_drift = k - start
                    manipulation_value = pow(
                        np.sqrt(manipulation_value) *
                        (nsteps_drift + k_drift * dq) /
                        (nsteps_drift + (k_drift - 1) * dq),
                        2
                    )

        else:
            warnings.warn(f"Invalid disturbance mode '{mode}' specified. Skipping.")

    return manipulation_value
