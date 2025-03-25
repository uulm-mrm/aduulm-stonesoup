import numpy as np

def disturbance_transition_model(transition_model, gt_configs, k):
    """
    This function modifies the transition model based on the disturbance modes ('jump', 'drift').

    Args:
        transition_model: The current ground truth transition model (CombinedLinearGaussianTransitionModel).
        gt_configs: The ground truth configuration.
        k: The current time step.

    Returns:
        Updated transition model with disturbance effects.
    """
    for idx_dist_mode, dist_mode in enumerate(gt_configs['disturbance_mode']):
        if dist_mode.lower() == 'none':
            continue  # No disturbance applied

        elif dist_mode.lower() == 'jump':
            for z in range(len(gt_configs['parameters'][idx_dist_mode])):
                if k == gt_configs['parameters'][idx_dist_mode][z][0]:
                    # Apply jump disturbance (modifying noise_diff_coeff and thus covariance matrix)
                    for model in transition_model.model_list:
                        model.noise_diff_coeff = model.noise_diff_coeff * gt_configs['parameters'][idx_dist_mode][z][1]

        elif dist_mode.lower() == 'drift':
            for z in range(len(gt_configs['parameters'][idx_dist_mode])):
                if gt_configs['parameters'][idx_dist_mode][z][0] < k <= gt_configs['parameters'][idx_dist_mode][z][1]:
                    dq = gt_configs['parameters'][idx_dist_mode][z][2] - 1
                    nsteps_drift = gt_configs['parameters'][idx_dist_mode][z][1] - gt_configs['parameters'][idx_dist_mode][z][0]
                    k_drift = k - gt_configs['parameters'][idx_dist_mode][z][0]

                    for model in transition_model.model_list:
                        model.noise_diff_coeff = model.noise_diff_coeff * ((nsteps_drift + k_drift * dq) /
                                                                    (nsteps_drift + (k_drift - 1) * dq))

        else:
            warnings.warn(f"Ground truth disturbance mode {dist_mode} specified in configuration "
                          f"does not match valid options.")

    return transition_model


def disturbance_measurement_model(measurement_model, gt_configs, k):
    """
    Apply disturbances to the measurement model based on the ground truth configuration.

    Args:
        measurement_model: The measurement model whose noise covariance needs modification.
        gt: The ground truth disturbance configuration.
        k: Current time step.

    Returns:
        Updated measurement model with disturbances applied.
    """
    measurement_model.noise_covar = measurement_model.noise_covar.astype(float)  # Ensure float type

    for idx_dist_mode, dist_mode in enumerate(gt_configs['disturbance_mode']):
        if dist_mode.lower() == 'none':
            pass

        elif dist_mode.lower() == 'jump':
            for z in gt_configs['parameters'][idx_dist_mode]:
                if k == z[0]:
                    measurement_model.noise_covar *= pow(z[1], 2)

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
            warnings.warn(f"Invalid disturbance mode: {dist_mode}")

    return measurement_model
