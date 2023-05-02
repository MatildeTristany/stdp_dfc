from networks import tpdi_networks
from networks import bp_network
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import numpy as np
from utils import utils

def generate_data_from_teacher(args, num_train=1000, num_test=100, n_in=5, n_out=5,
                               n_hidden=[1000,1000,1000,1000], activation='tanh',
                               device=None, num_val=None, random_seed=None,
                               random_seed_teacher=None):
    """
    Generate data for a regression task through a teacher model.
    This function generates random input patterns and creates a random MLP
    (fully-connected neural network), that is used as a teacher model. I.e., the
    generated input data is fed through the teacher model to produce target
    outputs. The so produced dataset can be used to train and assess a
    student model. Hence, a learning procedure can be verified by validating its
    capability of training a student network to mimic a given teacher network.
    Input samples will be uniformly drawn from a unit cube.
    .. warning::
        Since this is a synthetic dataset that uses random number generators,
        the generated dataset depends on externally configured random seeds
        (and in case of GPU computation, it also depends on whether CUDA
        operations are performed in a derterministic mode).
    Args:
        num_train (int): Number of training samples.
        num_test (int): Number of test samples.
        n_in (int): Passed as argument ``n_in`` to class
            :class:`networks.networks.DTPNetwork`
            when building the teacher model.
        n_out (int): Passed as argument ``n_out`` to class
            :class:`networks.networks.DTPNetwork`
            when building the teacher model.
        n_hidden (list): Passed as argument ``n_hidden`` to class
            :class:`networks.networks.DTPNetwork` when building the teacher model.
        activation (str): Passed as argument ``activation`` to
            class :class:`networks.networks.DTPNetwork` when building the
            teacher model
        random_seed (int): The random seed to be used for data generation.
        random_seed_teacher (int):  The random seed to be used for the teacher
            generation. This allows generating different data for a same teacher.
            If not provided, the same value as `random_seed` will be used.

    Returns:
        See return values of function :func:`regression_cubic_poly`.
    """

    if random_seed is None:
        random_seed = 420
    if random_seed_teacher is None:
        random_seed_teacher = random_seed

    fixed_random_seed = np.random.RandomState(random_seed)

    device = torch.device('cpu')
    if num_val is None:
        num_val = num_test
    # torch.manual_seed(args.random_seed)
    # torch.cuda.manual_seed_all(args.random_seed)
    # np.random.seed(args.random_seed)
    # random.seed(args.random_seed)

    rand = np.random

    train_x = fixed_random_seed.uniform(low=-1, high=1, size=(num_train, n_in))
    test_x = fixed_random_seed.uniform(low=-1, high=1, size=(num_test, n_in))
    val_x = fixed_random_seed.uniform(low=-1, high=1, size=(num_val, n_in))

    teacher = bp_network.BPNetwork(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                                      activation=activation, output_activation='linear',
                                      bias=True, initialization='teacher',
                                      random_seed=random_seed_teacher)

    if args.double_precision:
        train_y = teacher.forward(torch.from_numpy(train_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).to(torch.float64).to(device)) \
            .detach().cpu().numpy()
    else:
        train_y = teacher.forward(torch.from_numpy(train_x).float().to(device))\
            .detach().cpu().numpy()
        test_y = teacher.forward(torch.from_numpy(test_x).float().to(device))\
            .detach().cpu().numpy()
        val_y = teacher.forward(torch.from_numpy(val_x).float().to(device))\
            .detach().cpu().numpy()

    return train_x, test_x, val_x, train_y, test_y, val_y


def get_student_teacher_dataset(args, device, input_data_seed=None):
    """Get a dataset based on the student-teacher setting.

    This function can be used not only to create a dataset before training, but to
    ensure that a set of samples is used throughout different training epochs
    via the command-line option `dont_reuse_training_samples`.

    Args:
        args: The command line arguments.
        device: The cuda device.
        input_data_seed (int): The random seed to be used for generating the
            random input fed to the teacher. If None, the value of
            `args.data_random_seed` will be used.

    Returns:
        (....): Tuple containing:

        - **train_loader**: The train loader.
        - **test_loader**: The test loader.
        - **val_loader**: The validation loader.
    """
    if input_data_seed is None:
        input_data_seed = args.data_random_seed

    if not args.load_ST_dataset:
        if args.teacher_linear:
            activation = 'linear'
        else:
            activation = 'tanh'
        train_x, test_x, val_x, train_y, test_y, val_y = \
            generate_data_from_teacher(
                n_in=args.size_input, n_out=args.size_output,
                n_hidden=args.teacher_size_hidden, device=device,
                num_train=args.num_train, num_test=args.num_test,
                num_val=args.num_val,
                args=args, activation=activation,
                random_seed=input_data_seed,
                random_seed_teacher=args.data_random_seed)
        # Reset the torch seed (which has been overwritten for teacher).
        torch.manual_seed(args.random_seed)
    else:
        train_x = np.load('./data/train_x.npy')
        test_x = np.load('./data/test_x.npy')
        val_x = np.load('./data/val_x.npy')
        train_y = np.load('./data/train_y.npy')
        test_y = np.load('./data/test_y.npy')
        val_y = np.load('./data/val_y.npy')

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if args.double_precision:
        torch.set_default_dtype(torch.float64)

    train_loader = DataLoader(utils.RegressionDataset(train_x, train_y, args.double_precision),
                              batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(utils.RegressionDataset(test_x, test_y, args.double_precision),
                             batch_size=args.batch_size, shuffle=False)

    if args.no_val_set:
        val_loader = None
    else:
        val_loader = DataLoader(utils.RegressionDataset(val_x, val_y, args.double_precision),
                                batch_size=args.batch_size, shuffle=False)

    return train_loader, test_loader, val_loader


def build_network(args):
    """
    Create the network based on the provided command line arguments
    Args:
        args: command line arguments
    Returns: a network
    """

    forward_requires_grad = args.forward_requires_grad
    if args.classification:
        assert args.output_activation == 'softmax', "Output layer should " \
                    "represent probabilities => use softmax"
        output_activation = 'linear'
        
    else:
        output_activation = args.output_activation

    kwargs_bp = {
                'n_in': args.size_input,
                'n_hidden': args.size_hidden,
                'n_out': args.size_output,
                'activation': args.hidden_activation,
                'bias': not args.no_bias,
                'initialization': args.initialization,
                'output_activation': output_activation,
                }

    kwargs_dtp = {
                'sigma': args.sigma,
                'forward_requires_grad': forward_requires_grad,
                'save_df': args.save_df,
                'fb_activation': args.fb_activation,
                }

    kwargs_dfc = {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'ndi': args.ndi,
                'target_stepsize': args.target_stepsize,
                'alpha_di': args.alpha_di,
                'dt_di': args.dt_di,
                'dt_di_fb': args.dt_di_fb,
                'tmax_di': args.tmax_di,
                'tmax_di_fb': args.tmax_di_fb,
                'epsilon_di': args.epsilon_di,
                'reset_K': args.reset_K,
                'initialization_K': args.initialization_K,
                'noise_K': args.noise_K,
                'compare_with_ndi': args.compare_with_ndi,
                'out_dir': args.out_dir,
                'learning_rule': args.learning_rule,
                'use_initial_activations': args.use_initial_activations,
                'sigma': args.sigma,
                'sigma_fb': args.sigma_fb,
                'sigma_output': args.sigma_output,
                'sigma_output_fb': args.sigma_output_fb,
                'forward_requires_grad': forward_requires_grad,
                'save_df': args.save_df,
                'clip_grad_norm': args.clip_grad_norm,
                'k_p': args.k_p,
                'k_p_fb': args.k_p_fb,
                'inst_system_dynamics': args.inst_system_dynamics,
                'alpha_fb': args.alpha_fb,
                'noisy_dynamics': args.noisy_dynamics,
                'fb_learning_rule': args.fb_learning_rule,
                'inst_transmission': args.inst_transmission,
                'inst_transmission_fb': args.inst_transmission_fb,
                'time_constant_ratio': args.time_constant_ratio,
                'time_constant_ratio_fb': args.time_constant_ratio_fb,
                'apical_time_constant': args.apical_time_constant,
                'apical_time_constant_fb': args.apical_time_constant_fb,
                'grad_deltav_cont': args.grad_deltav_cont,
                'efficient_controller': args.efficient_controller,
                'proactive_controller': args.proactive_controller,
                'save_NDI_updates': args.save_NDI_angle,
                'save_eigenvalues': args.save_eigenvalues,
                'save_eigenvalues_bcn': args.save_eigenvalues_bcn,
                'save_norm_r': args.save_norm_r,
                'save_stdp_measures': args.save_stdp_measures,
                'save_correlations': args.save_correlations,
                'save_epoch': args.save_epoch,
                'simulate_layerwise': args.simulate_layerwise,
                'include_non_converged_samples': args.include_non_converged_samples,
                'low_pass_filter_u': args.low_pass_filter_u,
                'tau_f': args.tau_f,
                'tau_noise': args.tau_noise,
                'decay_rate': args.decay_rate,
                'stdp_samples': args.stdp_samples,
                'classification': args.classification,
                'use_jacobian_as_fb': args.use_jacobian_as_fb,
                'stability_tricks': args.stability_tricks,
                'freeze_fb_weights': args.freeze_fb_weights,
                'scaling_fb_updates': args.scaling_fb_updates,
                'at_steady_state': args.at_steady_state,
                'average_ss': args.average_ss,
                'not_low_pass_filter_r': args.not_low_pass_filter_r,
                'use_diff_hebbian_updates': args.use_diff_hebbian_updates,
                'use_stdp_updates': args.use_stdp_updates
                }

    kwargs_dfc = {**kwargs_bp, **kwargs_dfc}
    kwargs_dtp = {**kwargs_bp, **kwargs_dtp}

    if args.network_type == 'DTP':
        net = dtp_networks.DTPNetwork(**kwargs_dtp)
    elif args.network_type == 'BP':
        net = bp_network.BPNetwork(**kwargs_bp)
    elif args.network_type == 'DFC':
        net = tpdi_networks.DFCNetwork(**kwargs_dfc)
    elif args.network_type == 'DFC_sfb':
        net = tpdi_networks.DFC_sfb_Network(**kwargs_dfc)
    elif args.network_type == 'DFC_single_phase':
        kwargs_dfc_one_phase = {'pretrain_without_controller': \
                                           args.pretrain_without_controller,
                                'not_high_pass_filter_u_fb':
                                           args.not_high_pass_filter_u_fb}
        net = tpdi_networks.DFC_single_phase_Network(**kwargs_dfc,
                                                     **kwargs_dfc_one_phase)
    elif args.network_type == 'MN':
        net = tpdi_networks.MNNetwork(**kwargs_dfc)
    elif args.network_type == 'DFA':
        net = direct_feedback_networks.DDTPMLPNetwork(**kwargs_dtp,
                              size_hidden_fb=None,
                              fb_hidden_activation='linear',
                              recurrent_input=False,
                              fb_weight_initialization=args.initialization_K)
    elif args.network_type == 'GN':
        net = tpdi_networks.GNNetwork(**kwargs_dfc)
    else:
        raise ValueError('The provided network type {} is not supported'.format(
            args.network_type
        ))

    return net