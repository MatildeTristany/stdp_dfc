import argparse
import warnings
from datetime import datetime
import os
import json
from utils import utils
import pickle

def parse_cmd_arguments(default=False, argv=None):
    """
    Parse command-line arguments.
    Args:
        default (optional): If True, command-line arguments will be ignored and
            only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.
    Returns:
        The Namespace object containing argument names and values.
    """

    description = 'Training a Dynamical Feedback Control Network'

    parser = argparse.ArgumentParser(description=description)

    dataset_args(parser)
    training_args(parser)
    adam_args(parser)
    network_args(parser)
    miscellaneous_args(parser)
    logging_args(parser)
    dynamical_inversion_args(parser)
    dfc_args(parser)

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    post_process_args(config)
    check_invalid_args(config)

    return config


def dataset_args(parser, ddataset='student_teacher',
                 dnum_train=1000, dnum_val=1000, dnum_test=1000,
                 dtarget_class_value=1):
    """
    Args:
        parser (argparse.ArgumentParser): argument parser
        ddataset: default
        dnum_train: default
        dnum_val: default
        dnum_test: default
        dtarget_class_value: default
    Returns: The created argument group, in case more options should be added.\
    """

    dgroup = parser.add_argument_group('Dataset options')
    dgroup.add_argument('--dataset', type=str, default=ddataset,
                        choices=['mnist', 'student_teacher', 'fashion_mnist',
                                 'cifar10', 'mnist_autoencoder'],
                        help='Used dataset for classification/regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--num_train', type=int, default=dnum_train,
                        help='Number of training samples used for the '
                             'student teacher regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--num_test', type=int, default=dnum_test,
                        help='Number of test samples used for the '
                             'student teacher regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--num_val', type=int, default=dnum_val,
                        help='Number of validation samples used for the'
                             'student teacher regression. '
                             'Default: %(default)s.')
    dgroup.add_argument('--no_preprocessing_mnist', action='store_true',
                        help='take the mnist input values between 0 and 1, '
                             'instead of standardizing them.')
    dgroup.add_argument('--no_val_set', action='store_true',
                        help='Flag indicating that no validation set is used'
                             'during training.')
    dgroup.add_argument('--load_ST_dataset', action='store_true',
                        help='Load a synthetic student-teacher dataset from the'
                             'data folder, instead of randomly generating one.'
                             'Warning: only compatible with networks of '
                             'input size 10 and output size 2.')
    dgroup.add_argument('--target_class_value', type=float,
                        default=dtarget_class_value,
                        help='For classification tasks, the value that the '
                             'correct class should have. Values of 1 '
                             'correspond to one-hot-encoding, and values '
                             'smaller than one correspond to soft targets. '
                             'Default: %(default)s.')
    dgroup.add_argument('--dont_reuse_training_samples', action='store_true',
                        help='Flag indicating whether new training samples '
                             'should be used at every training epoch for ' 
                             'the student-teacher dataset.')

    return dgroup


def training_args(parser, depochs=2, dbatch_size=128, dlr='0.1',
                  dlr_fb='0.1', dtarget_stepsize=0.001,
                  doptimizer='SGD', doptimizer_fb='SGD', dmomentum=0.,
                  dsigma=0.08, dforward_wd=0., dfeedback_wd=.01,
                  depochs_fb=1, dextra_fb_epochs=0, dextra_fb_minibatches=0,
                  dclip_grad_norm=-1):
    """
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments.
    Returns:
        The created argument group, in case more options should be added.
    """

    tgroup = parser.add_argument_group('Training options')
    tgroup.add_argument('--epochs', type=int, metavar='N', default=depochs,
                        help='Number of training epochs. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--batch_size', type=int, metavar='N', default=dbatch_size,
                        help='Training batch size. '
                             'Choose divisor of "num_train". '
                             'Default: %(default)s.')
    tgroup.add_argument('--lr', type=str, default=dlr,
                        help='Learning rate of optimizer for the forward '
                             'parameters. You can either provide a single '
                             'float that will be used as lr for all the layers,'
                             'or a list of learning rates (e.g. [0.1,0.2,0.5]) '
                             'specifying a lr for each layer. The lenght of the'
                             ' list should be equal to num_hidden + 1. The list'
                             'may not contain spaces. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--lr_fb', type=str, default=dlr_fb,
                        help='Learning rate of optimizer for the feedback '
                             'parameters. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--lr_fb_init', type=str, default=None,
                        help='Learning rate of optimizer for the feedback '
                             'parameters during the initial pre-training of '
                             'the feedback weights for epoch_fb epochs. '
                             'Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--target_stepsize', type=float, default=dtarget_stepsize,
                        help='Step size for computing the output target based'
                             'on the output gradient. Default: ' +
                             '%(default)s.')
    tgroup.add_argument('--optimizer', type=str, default=doptimizer,
                        choices=['SGD', 'RMSprop', 'Adam'],
                        help='Optimizer used for training. Default: '
                             '%(default)s.')
    tgroup.add_argument('--optimizer_fb', type=str, default=doptimizer_fb,
                        choices=[None, 'SGD', 'RMSprop', 'Adam'],
                        help='Optimizer used for training the feedback '
                             'parameters.')
    tgroup.add_argument('--opt_reset_interval', type=int, default=-1,
                        help='After how many epochs to reset the '
                             'optimizer. When using Adam, this allows a ' +
                             'kind of scheduling. By default, no ' +
                             'resetting is done.')
    tgroup.add_argument('--momentum', type=float, default=dmomentum,
                        help='Momentum of the SGD or RMSprop optimizer. ' +
                             'Default: %(default)s.')
    tgroup.add_argument('--sigma', type=float, default=dsigma,
                        help='svd of gaussian noise used to corrupt the'
                             'layer activations during the controller '
                             'dynamics. Default: %(default)s.')
    tgroup.add_argument('--sigma_output', type=float, default=-1,
                        help='svd of gaussian noise used to corrupt the'
                             'output layer activations during the controller '
                             'dynamics in the forward weight training. '
                             'Default: %(default)s.')
    tgroup.add_argument('--sigma_fb', type=float, default=-1,
                        help='svd of gaussian noise used to corrupt the'
                             'layer activations during the controller '
                             'dynamics for the feedback learning phase.'
                             ' Default: %(default)s.')
    tgroup.add_argument('--sigma_output_fb', type=float, default=-1,
                        help='svd of gaussian noise used to corrupt the'
                             'output layer activations during the controller '
                             'dynamics in the feedback weight training. '
                             'Default: %(default)s.')
    tgroup.add_argument('--forward_wd', type=float, default=dforward_wd,
                        help='Weight decay for the forward weights. '
                             'Default: %(default)s.')
    tgroup.add_argument('--feedback_wd', type=float, default=dfeedback_wd,
                        help='Weight decay for the feedback weights. '
                             'Default: %(default)s.')
    tgroup.add_argument('--train_parallel', action='store_true',
                        help='Flag indicating that for each minibatch, '
                             'both the feedforward and feedback parameters '
                             'are updated. If not True, the default option '
                             'is taken, where we alternate between training the '
                             'forward parameters for 1 epoch, after which we '
                             'turn to a "sleep-phase" for 1 epoch, where only'
                             'the feedback weights are trained.')
    tgroup.add_argument('--normalize_lr', action='store_true',
                        help='Flag indicating that we should take the real '
                             'learning rate of the forward parameters to be:'
                             'lr=lr/target_stepsize. This makes the hpsearch'
                             'easier, as lr and target_stepsize have similar'
                             'behavior.')
    tgroup.add_argument('--epochs_fb', type=float, default=depochs_fb,
                        help='Number of epochs on which the feedback parameters'
                             'are trained before the actual training (of '
                             'both forward and feedback parameters) is started.'
                             'Default: %(default)s.')
    tgroup.add_argument('--pretrain_without_controller', action='store_true',
                        help='If active, the feedback weight pretraining '
                             'will use the forward activations without '
                             'controller active as a desired target. Only has '
                             'an effect when using "DFC_single_phase", '
                             'since by default in this setting the actual '
                             'supervised labels are used as targets.')
    tgroup.add_argument('--freeze_forward_weights', action='store_true',
                        help='Only train the feedback weights, the forward '
                             'weights stay fixed. We still compute the targets'
                             'to update the forward weights, such that they '
                             'can be logged and investigated.')
    tgroup.add_argument('--freeze_fb_weights',
                        action='store_true',
                        help='Only train the forward parameters, the '
                             'feedback parameters stay fixed.')
    tgroup.add_argument('--freeze_fb_weights_output', action='store_true',
                        help='Freeze the feedback weights of the output layer,'
                             'such that they remain equal to the identity '
                             'matrix (their initialization).')
    tgroup.add_argument('--shallow_training', action='store_true',
                        help='Train only the parameters of the last layer and'
                             'let the others stay fixed.')
    tgroup.add_argument('--extra_fb_epochs', type=float, default=dextra_fb_epochs,
                        help='After each epoch of training, the fb parameters'
                             'will be trained for an extra extra_fb_epochs '
                             'epochs. Default: 0')
    tgroup.add_argument('--extra_fb_minibatches', type=int,
                        default=dextra_fb_minibatches,
                        help='After each minibatch training of the forward '
                             'parameters, we do <N> extra minibatches training'
                             'for the feedback weights. The extra minibatches '
                             'are randomly sampled from the trainingset')
    tgroup.add_argument('--only_train_first_layer', action='store_true',
                        help='Only train the forward parameters of the first '
                             'layer, while freezing all other forward'
                             ' parameters to their initialization. The feedback'
                             'parameters are all trained.')
    tgroup.add_argument('--train_only_feedback_parameters', action='store_true',
                        help='Flag indicating that only the feedback parameters'
                             'should be trained, not the forward parameters.')
    tgroup.add_argument('--clip_grad_norm', type=float, default=dclip_grad_norm,
                        help='Clip the norm of the forward and feedback weight updates if '
                             'they are bigger than the specified value. If a '
                             'value smaller than zero is provided, no gradient '
                             'clipping is done. ')
    tgroup.add_argument('--grad_deltav_cont', action='store_true',
                        help='Flag indicating that the deltav of the feedforward '
                             'gradients are being computed continuously.')
    tgroup.add_argument('--fix_grad_norm', type=float, default=-1,
                        help='Rescale the updates such that the total norm '
                             'of all concatenated updates is equal to fix_grad_norm.'
                             'If -1, no rescaling is done.')
    tgroup.add_argument('--sum_reduction', action='store_true',
                        help='Use the sum reduction instead of the mean reduction'
                             'in the losses. This option should be selected when '
                             'using DFC variants, but not when using BP variants.'
                             'In a future version, this will be taken care of '
                             'automatically, we know leave it as an option for being '
                             'compatible with the hyperparameters of the publication.')
    tgroup.add_argument('--use_bp_updates', action='store_true',
                        help='Overwrite the updates with the loss gradients '
                             '(BP updates), i.e. training the network with BP, '
                             'but still instantiate the network as defined '
                             'by network_type. This is used for debugging '
                             'purposes.')
    tgroup.add_argument('--use_diff_hebbian_updates', action='store_true',
                        help='Overwrite the updates with the differential Hebbian updates.')
    tgroup.add_argument('--use_stdp_updates', action='store_true',
                        help='Overwrite the updates with the STDP updates.')

    return tgroup


def adam_args(parser, dbeta1=0.99, dbeta2=0.99, depsilon='1e-8',
              dbeta1_fb=0.99, dbeta2_fb=0.99, depsilon_fb='1e-8'):
    """
    Training options for the Adam optimizer
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments
    Returns:
        The created argument group, in case more options should be added.
    """

    agroup = parser.add_argument_group('Training options for the '
                                       'Adam optimizer')
    agroup.add_argument('--beta1', type=float, default=dbeta1,
                        help='beta1 training hyperparameter for the adam '
                             'optimizer. Default: %(default)s')
    agroup.add_argument('--beta2', type=float, default=dbeta2,
                        help='beta2 training hyperparameter for the adam '
                             'optimizer. Default: %(default)s')
    agroup.add_argument('--epsilon', type=str, default=depsilon,
                        help='epsilon training hyperparameter for the adam '
                             'optimizer. Default: %(default)s')
    agroup.add_argument('--beta1_fb', type=float, default=dbeta1_fb,
                        help='beta1 training hyperparameter for the adam '
                             'feedback optimizer. Default: %(default)s')
    agroup.add_argument('--beta2_fb', type=float, default=dbeta2_fb,
                        help='beta2 training hyperparameter for the adam '
                             'feedback optimizer. Default: %(default)s')
    agroup.add_argument('--epsilon_fb', type=str, default=depsilon_fb,
                        help='epsilon training hyperparameter for the adam '
                             'feedback optimizer. Default: %(default)s')
    return agroup


def network_args(parser, dnum_hidden=1, dsize_hidden='5', dsize_input=10,
                 dsize_output=2, dhidden_activation='linear',
                 doutput_activation='linear', dnetwork_type='DFC',
                 dinitialization='xavier_normal',
                 tnum_hidden=1, tsize_hidden='5',):
    """
    Network options
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments
    Returns:
        The created argument group, in case more options should be added.
    """

    sgroup = parser.add_argument_group('Network options')
    sgroup.add_argument('--num_hidden', type=int, metavar='N', default=dnum_hidden,
                        help='Number of hidden layer in the ' +
                             '(student) network (depth). Default: %(default)s.')
    sgroup.add_argument('--size_hidden', type=str, metavar='N', default=dsize_hidden,
                        help='Number of units in each hidden layer of the ' +
                             '(student) network. Default: %(default)s.'
                             'If you provide a list, you can have layers of '
                             'different sizes (width).')
    sgroup.add_argument('--teacher_num_hidden', type=int, metavar='N', default=tnum_hidden,
                        help='Number of hidden layer in the ' +
                             '(teacher) network (depth). Default: %(default)s.')
    sgroup.add_argument('--teacher_size_hidden', type=str, metavar='N', default=tsize_hidden,
                        help='Number of units in each hidden layer of the ' +
                             '(teacher) network. Default: %(default)s.'
                             'If you provide a list, you can have layers of '
                             'different sizes (width).')
    sgroup.add_argument('--size_input', type=int, metavar='N', default=dsize_input,
                        help='Number of units of the input'
                             '. Default: %(default)s.')
    sgroup.add_argument('--size_output', type=int, metavar='N', default=dsize_output,
                        help='Number of units of the output'
                             '. Default: %(default)s.')
    sgroup.add_argument('--hidden_activation', type=str, default=dhidden_activation,
                        help='Activation function used for the hidden layers. '
                             'Default: $(default)s.')
    sgroup.add_argument('--output_activation', type=str, default=doutput_activation,
                        choices=['tanh', 'relu', 'linear', 'leakyrelu',
                                 'sigmoid', 'softmax', 'cap_relu'],
                        help='Activation function used for the output. '
                             'Default: $(default)s.')
    sgroup.add_argument('--no_bias', action='store_true',
                        help='Flag for not using biases in the network.')
    sgroup.add_argument('--network_type', type=str, default=dnetwork_type,
                        choices=['DFC', 'DFC_single_phase', 'DTP',
                                 'BP', 'DFA', 'MN', 'GN'],
                        help='Variant of TP that will be used to train the '
                             'network. See the layer classes for explanations '
                             'of the names. Default: %(default)s.')
    sgroup.add_argument('--initialization', type=str, default=dinitialization,
                        choices=['orthogonal', 'xavier', 'xavier_normal',
                                 'teacher'],
                        help='Type of initialization that will be used for the '
                             'forward and feedback weights of the network.'
                             'Default: %(default)s.')
    sgroup.add_argument('--fb_activation', type=str, default=None,
                        choices=['tanh', 'relu', 'linear', 'leakyrelu',
                                 'sigmoid', 'cap_relu'],
                        help='Only use with DTP. '
                             'Activation function used for the feedback targets'
                             'for the hidden layers. Default the same as '
                             'hidden_activation.')


def miscellaneous_args(parser, drandom_seed=42, ddata_random_seed=420):
    """
    Miscellaneous options
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments
    Returns:
        The created argument group, in case more options should be added.
    """

    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--no_cuda', action='store_true',
                        help='Flag to disable GPU usage.')
    mgroup.add_argument('--random_seed', type=int, metavar='N',
                        default=drandom_seed,
                        help='Random seed. Default: %(default)s.')
    mgroup.add_argument('--data_random_seed', type=int, metavar='N',
                        default=ddata_random_seed,
                        help='Random seed for data generation, relevant for '
                             'student-teacher settings. Default: %(default)s.')
    mgroup.add_argument('--cuda_deterministic', action='store_true',
                        help='Flag to make the GPU computations deterministic.'
                             'note: this slows down computation!')
    mgroup.add_argument('--hpsearch', action='store_true',
                        help='Flag indicating that the main script is running '
                             'in the context of a hyper parameter search.')
    mgroup.add_argument('--multiple_hpsearch', action='store_true',
                        help='flag indicating that main is runned in the '
                             'context of multiple_hpsearches.py')
    mgroup.add_argument('--euler_hpsearch', action='store_true',
                        help='Flag indicating that the main script is running '
                             'in the context of a hyper parameter search '
                             'in the Euler cluster.')
    mgroup.add_argument('--single_precision',
                        action='store_true',
                        help='Use single precision floats (32bits) instead of '
                             '64bit floats. This speeds up training, but can'
                             'lead to numerical issues.')
    mgroup.add_argument('--evaluate', action='store_true',
                        help="Don't stop unpromising runs, because we are "
                             "evaluating hp parameter results.")
    mgroup.add_argument('--test', action='store_true',
                        help='If active, this option will ensure that the '
                             'simulation is very fast by cutting the number '
                             'of epochs to 1, and by using only a couple of '
                             'batches. This can be used for debugging.')


def logging_args(parser, dout_dir=None, dgn_damping='0', dlog_interval=200, dsave_epoch=1):
    """
    Logging options
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments
    Returns:
        The created argument group, in case more options should be added.
    """

    if dout_dir is None:
        dout_dir = './logs/runs/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    vgroup = parser.add_argument_group('Logging options')
    vgroup.add_argument('--out_dir', type=str, default=dout_dir,
                        help='Relative path to directory where the logs are '
                             'saved.')
    vgroup.add_argument('--save_logs',
                        action='store_true',
                        help='Flag to save logs and plots by using '
                             'tensorboardX.')
    vgroup.add_argument('--save_summary_interval', type=int, default=-1,
                        help='Every how many epochs to store a separate '
                             'summary file.')
    vgroup.add_argument('--save_checkpoints', action='store_true',
                        help='If active, networks will be stored after '
                             'pre-training (if existing) and during training '
                             '(only relevant for DFC).')
    vgroup.add_argument('--pretrained_net_dir', type=str, default=None,
                        help='The path to the pre-trained network to be '
                             'loaded (only relevant for DFC).')
    vgroup.add_argument('--save_BP_angle',
                        action='store_true',
                        help='Flag indicating whether the BP updates and the'
                             'angle between those updates and the TP updates'
                             'should be computed and saved.')
    vgroup.add_argument('--save_GN_angle', action='store_true',
                        help='Flag indicating whether the GN updates and the'
                             'angle between those updates and the TP updates'
                             'should be computed and saved. Warning: this '
                             'causes a heavy extra computational load.')
    vgroup.add_argument('--save_GNT_angle',
                        action='store_true',
                        help='Flag indicating whether angle with the ideal '
                             'GNT updates should be computed. Warning, this'
                             'causes a heavy extra computational load.')
    vgroup.add_argument('--save_LU_angle',
                        action='store_true',
                        help='Flag indicating whether angle with the ideal '
                             'updates from loss_u should be computed. Warning, '
                             'this causes a heavy extra computational load.')
    vgroup.add_argument('--save_lu_loss',
                        action='store_true',
                        help='Flag indicating whether L_u loss should be saved.')
    vgroup.add_argument('--save_ratio_angle_ff_fb',
                        action='store_true',
                        help=r'Flag indicating whether ratio between the '
                             r'feeforward and feedback stimulus should be '
                             r'saved: ||Q_{i}u|| / ||W_{i}r_{i-1}||.')
    vgroup.add_argument('--save_fb_values',
                        action='store_true',
                        help=r'Flag indicating whether the '
                             r'feedback stimulus should be '
                             r'saved: ||Q_{i}u||.')
    vgroup.add_argument('--save_NDI_angle',
                        action='store_true',
                        help='Flag indicating whether angle with the analytical '
                             'NDI updates should be computed. Causes a minor '
                             'increase in computational load.')
    vgroup.add_argument('--save_BP_GNT_angle', 
                        action='store_true',
                        help='Flag indicating whether angle between the ideal '
                             'GNT updates and the BP updates should be computed. '
                             'Warning, this'
                             'causes a heavy extra computational load.')
    vgroup.add_argument('--save_GN_GNT_angle',
                        action='store_true',
                        help='Flag indicating whether angle between the ideal '
                             'GNT updates and the ideal GN updates should be computed. '
                             'Warning, this'
                             'causes a heavy extra computational load.')
    vgroup.add_argument('--save_GNT_ss_no_ss_angle', 
                        action='store_true',
                        help=r'Flag indicating whether angle between the ideal '
                             r'GNT updates computed with the steady-state value '
                             r'of r_{i-1} and the ideal GNT updates with the '
                             r'feedforward value of r_{i-1} should be computed. '
                             r'Warning, this'
                             r'causes a heavy extra computational load.')
    vgroup.add_argument('--save_condition_gn', action='store_true',
                        help='Flag indicating whether the Gauss-Newton '
                             'condition on the feedback weights should be '
                             'computed and saved.')
    vgroup.add_argument('--save_eigenvalues', action='store_true',
                        help='Flag indicating whether the stability matrices A '
                             'should be computed and their eigenvalues saved.'
                             'These matrices are evaluated at the steady state of the '
                             'controller (or NDI if --ndi is activated).'
                             'There are two values (two plots) for each layer, one for mean'
                             'and another one for standard deviation across the batch.')
    vgroup.add_argument('--save_jac_t_angle', action='store_true',
                        help='Save the angle with the transpose of the network '
                             'jacobian during pretraining of the network')
    vgroup.add_argument('--save_jac_pinv_angle', action='store_true',
                        help='Save the angle with the pseudoinverse of the network '
                             'jacobian during pretraining of the network. '
                             'The pseuodinverse will be damped with gn_damping')
    vgroup.add_argument('--save_eigenvalues_bcn', action='store_true',
                        help='Flag indicating whether the stability matrices A '
                             'should be computed and their eigenvalues saved,'
                             'before non-converged (and diverged) samples are corrected'
                             'back to the feedforward values.'
                             'This is automatically deactivated with --ndi, and '
                             '--save_eigenvalues is turned on instead '
                             '(to be evaluated at NDI solution).'
                             'There is a histogram (over samples) per each matrix A'
                             'and per epoch. Mean and std are saved in a vector for '
                             'latter plotting.')
    vgroup.add_argument('--save_norm_r', action='store_true',
                        help='Flag indicating whether the (relative) norm of the '
                             'activations of each layer should be computed and stored.'
                             'Computed as norm(r_i) - mean(norm(r)) / mean(norm(r)), '
                             'individually for each sample in the batch.'
                             'A histogram across samples is generated, and the mean and'
                             'std saved for future plotting.')
    vgroup.add_argument('--save_df', action='store_true',
                        help='Flag indicating that the computed angles should'
                             'be saved in dataframes for later use or plotting.')
    vgroup.add_argument('--gn_damping', type=str, default=dgn_damping,
                        help='Thikonov damping used for computing the '
                             'Gauss-Newton updates that are used to compute'
                             'the angles between the actual updates and the'
                             'GN updates. Default: %(default)s.')
    vgroup.add_argument('--log_interval', type=int, default=dlog_interval,
                        help='Each <log_interval> batches, the batch results'
                             'are logged to tensorboard.')
    vgroup.add_argument('--gn_damping_hpsearch', action='store_true',
                        help='Flag indicating whether a small hpsearch should'
                             'be performed to optimize the gn_damping constant'
                             'with which the gnt angles are computed.')
    vgroup.add_argument('--save_nullspace_norm_ratio', action='store_true',
                        help='Flag indicating whether the norm ratio between the'
                             'nullspace components of the parameter updates and'
                             'the updates themselves should be computed and '
                             'saved.')
    vgroup.add_argument('--save_fb_statistics_init', action='store_true',
                        help='Flag indicating that the statistics of the '
                             'feedback weights should be saved during the '
                             'initial training of the feedback weights (before'
                             'the real training of both forward and feedback'
                             'weights starts.')
    vgroup.add_argument('--compute_gn_condition_init', action='store_true',
                        help='Flag indicating that the gn_condition should '
                             'be computed at the end of the initial feedback'
                             'training and stored in the summary under the key'
                             'gn_condition_init.')
    vgroup.add_argument('--use_ss_gnt', action='store_true',
                        help=r'Use the steady state value of r_{i-1} to '
                             r'compute the ideal GNT update.')
    vgroup.add_argument('--use_nonlinear_gnt', action='store_true',
                        help=r'Use the modified version of the  '
                             r'ideal GNT update, where the nonlinearity is '
                             r'applied to the GN targets before computing the '
                             r'parameter updates.')
    vgroup.add_argument('--make_dynamics_plot', action='store_true',
                        help='Make for each epoch a plot of the dynamics of '
                             'the first minibatch during the feedback phase '
                             '(controller on).')
    vgroup.add_argument('--plot_filters', action='store_true',
                        help='Plot the effect of filtering on u, the noise and '
                             'the target rates. Important for single phase '
                             'experiments.')
    vgroup.add_argument('--make_dynamics_plot_interval', type=int, metavar='N', default=1,
                        help='If the "make_dynamics_plot" flag is on, how often (in epochs)'
                             'the dynamics will be plotted. Default: 1 (every epoch).')
    vgroup.add_argument('--plot_autoencoder_images', action='store_true',
                        help='Save plots of the input and reconstructed '
                             'images for each epoch of the autoencoder '
                             'training.')
    vgroup.add_argument('--save_stability_condition', action='store_true',
                        help='save the minimum eigenvalue of JQ during '
                             'training, to check local stability')
    vgroup.add_argument('--save_stdp_measures', action='store_true',
                        help='save the measures of interest for the '
                             'stdp rule analysis: pre_activities,'
                             'delta_post_activities, weights'
                             'weights_updates, central_activities.')
    vgroup.add_argument('--save_correlations', action='store_true',
                        help='save the agles (correlations) '
                             'between the stdp updates and the ,'
                             'BP, DFC Differential Hebbian,'
                             'and DFC udpates.')
    vgroup.add_argument('--save_epoch', type=int, default=dsave_epoch,
                        help='save the stdp measures corresponding'
                             'to this epoch.')
    vgroup.add_argument('--surprise_shuffle', action='store_true',
                        help='shuffle labels for the second epoch'
                             'to record effect on feedback signal.')
    return vgroup


def dynamical_inversion_args(parser, dalpha_di=0.001, ddt_di=0.1,
                             dtmax_di=10., depsilon_di=0.5,
                             dinitialization_K='weight_product',
                             dnoise_K=0.):
    """
    Options for the dynamical inversion process of DFC
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments
    Returns:
        The created argument group, in case more options should be added.
    """

    digroup = parser.add_argument_group('Dynamical Inversion options')
    digroup.add_argument('--ndi', action='store_true',
                         help='Non-dynamical inversion directly computes the steady-state'
                              'solution of the dynamical system, speeding up computation'
                              'under the assumption of convergence.')
    digroup.add_argument('--alpha_di', type=float, default=dalpha_di,
                         help='Alpha (leakage) of the component. Stabilizes the dynamics.'
                              'Default: %(default)s.')
    digroup.add_argument('--dt_di', type=float, default=ddt_di,
                         help='Time step for Euler steps, to compute the dynamical '
                              'inversion of targets. Larger dt leads to faster '
                              'convergence (specially if used with fast DI), but'
                              'dt>0.5 can lead to unstable, non-convergent dynamics.'
                              'Default: %(default)s.')
    digroup.add_argument('--dt_di_fb', type=float, default=-1,
                         help='Time step for Euler steps, to compute the dynamical '
                              'inversion of targets. Larger dt leads to faster '
                              'convergence (specially if used with fast DI), but'
                              'dt>0.5 can lead to unstable, non-convergent dynamics.'
                              'Default: %(default)s.')
    digroup.add_argument('--tmax_di', type=float, default=dtmax_di,
                         help='Maximum number of iterations (timesteps) performed in '
                              'dynamical inversion of targets. Default: %(default)s.')
    digroup.add_argument('--tmax_di_fb', type=float, default=-1,
                         help='Maximum number of iterations (timesteps) performed in '
                              'dynamical inversion of targets during the '
                              'feedback weight training. Default: %(default)s.')
    digroup.add_argument('--epsilon_di', type=float, default=depsilon_di,
                         help='Avoids spurious minima, but setting it too low (<0.2)'
                              ' can lead to slower computation (as some error minima '
                              'will not be detected as such). Default: %(default)s.')
    digroup.add_argument('--reset_K', action='store_true',
                         help='Resets the inversion matrix K to the product of transpose'
                              'forward weights, each time the "backward" function is called.')
    digroup.add_argument('--initialization_K', type=str,
                         default=dinitialization_K,
                         choices=['weight_product', 'orthogonal', 'xavier',
                                  'xavier_normal'],
                         help='Type of initialization that will be used for the '
                              'forward and feedback weights of the network.'
                              'Default: %(default)s.')
    digroup.add_argument('--noise_K', type=float, default=dnoise_K,
                         help='Allows to add some Gaussian to ' 
                              'the weight product initialization '
                              'of the inversion matrix K')
    digroup.add_argument('--compare_with_ndi',
                         action='store_true',
                         help='When doing dynamical control, also compute NDI (analytical solution)'
                              'in order to compare both solutions. Causes a computational overhead.')


def dfc_args(parser, dlearning_rule='nonlinear_difference',
             dk_p=0., dalpha_fb=0.5, 
             dfb_learning_rule='special_controller',
             dtime_constant_ratio=1., dapical_time_constant=-1,
             dtau_f=0.5, dtau_noise=0.05, ddecay_rate=0.8, dstdp_samples=1):
    """
    Options for the Dynamic Feedback Control Network
    Args:
        parser (argparse.ArgumentParser): argument parser
        (....): Default values for the arguments
    Returns:
        The created argument group, in case more options should be added.
    """

    dgroup = parser.add_argument_group('Dynamic Feedback Control options')
    dgroup.add_argument('--learning_rule', type=str,
                           default=dlearning_rule,
                           choices=['voltage_difference', 'derivative_matrix',
                                    'nonlinear_difference'],
                           help='Learning rule, based on the microcircuit implementation,'
                                'that will be used to update (basal) weights. Default: %(default)s.')
    dgroup.add_argument('--use_initial_activations', action='store_true',
                           help="For learning, uses the activations of the previous layer"
                                "at feedforward time, instead of the activations reached"
                                "through dynamical inversion. Activating this option makes"
                                "learning not local in time.")
    dgroup.add_argument('--use_jacobian_as_fb', action='store_true',
                        help='Use the Jacobian as the feedback weights to '
                             'control the network.')
    dgroup.add_argument('--k_p', type=float, default=dk_p,
                        help='The gain factor of the proportional control '
                             'term of the PI controller. When equal to zero'
                             '(by default), the proportional control term is '
                             'omitted and only integral control is used.'
                             'Only positive values allowed for k_p.')
    dgroup.add_argument('--k_p_fb', type=float, default=None,
                        help='The gain factor of the proportional control '
                             'term of the PI controller, used in the "sleep-phase"'
                             'for training the feedback weights. When equal to zero'
                             '(by default), the proportional control term is '
                             'omitted and only integral control is used.'
                             'Only positive values allowed for k_p_fb.')
    dgroup.add_argument('--inst_system_dynamics', action='store_true',
                        help='Flag indicating that the system dynamics, i.e.'
                             'the dynamics of the somatic compartments, should'
                             'be approximated by their instantaneous '
                             'counterparts, i.e. equivalent to having a '
                             'time constant of lim -> 0.')
    dgroup.add_argument('--alpha_fb', type=float, default=dalpha_fb,
                        help='Leakage gain of the feedback controller,'
                             'used in the sleeping phase where only the '
                             'feedback weights are trained.')
    dgroup.add_argument('--noisy_dynamics', action='store_true',
                        help='Flag indicating whether the dynamics of the '
                             'system should be corrupted slightly with '
                             'white noise with std sigma.')
    dgroup.add_argument('--fb_learning_rule', type=str,
                        default=dfb_learning_rule,
                        choices=['normal_controller', 'special_controller',
                                 'old_learning_rule'],
                        help='String indicating which learning rule '
                             'should be used for training the feedback '
                             'weights.')
    dgroup.add_argument('--teacher_linear', action='store_true',
                        help='Flag indicating that the student-teacher dataset '
                             'is created with a linear teacher.')
    dgroup.add_argument('--inst_transmission', action='store_true',
                        help='Flag indicating that we assume an instantaneous '
                             'transmission between layers, such that in '
                             'one simulation iteration, the basal voltage of '
                             'layer i at timestep t is based on the forward '
                             'propagation of the somatic voltage of layer i-1 '
                             'at timestep t, hence already incorporating the '
                             'feedback of the previous layer at the current '
                             'timestep.')
    dgroup.add_argument('--inst_transmission_fb', action='store_true',
                        help='Flag indicating that we assume an instantaneous '
                             'transmission between layers, such that in '
                             'one simulation iteration, the basal voltage of '
                             'layer i at timestep t is based on the forward '
                             'propagation of the somatic voltage of layer i-1 '
                             'at timestep t, hence already incorporating the '
                             'feedback of the previous layer at the current '
                             'timestep. This flag will only be applied '
                             'during the feedback training phase.')
    dgroup.add_argument('--time_constant_ratio', type=float,
                        default=dtime_constant_ratio,
                        help='ratio of the time constant of the'
                        'voltage dynamics w.r.t. the controller dynamics.')
    dgroup.add_argument('--time_constant_ratio_fb', type=float,
                        default=-1,
                        help='ratio of the time constant of the'
                             'voltage dynamics w.r.t. the controller dynamics.')
    dgroup.add_argument('--apical_time_constant', type=float,
                        default=dapical_time_constant,
                        help='Time constant of the apical compartment. '
                             'By default (-1), it will be set equal to '
                             'args.dt_di, such that it results in '
                             'instantaneous apical compartment dynamics.')
    dgroup.add_argument('--apical_time_constant_fb', type=float,
                        default=dapical_time_constant,
                        help='Time constant of the apical compartment during '
                             'the feedback training phase (sleep phase). '
                             'By default (-1), it will be set equal to '
                             'args.dt_di, such that it results in '
                             'instantaneous apical compartment dynamics.')
    dgroup.add_argument('--efficient_controller', action='store_true',
                        help='Use a more memory-efficient implementation of the '
                             'controller. Note that when this option is selected,'
                             'no plots of the dynamics of the feedback phase'
                             'can be saved.')
    dgroup.add_argument('--proactive_controller', action='store_true',
                        help='Use a slight variation on the forward Euler method'
                             'for simulating the controller dynamics, such that'
                             'the control input u[k+1] (which incorporates '
                             'the control error e[k]) is used to compute the '
                             'apical compartment voltage v^A[k+1], instead of '
                             'using u[k].')
    dgroup.add_argument('--simulate_layerwise', action='store_true',
                        help="Use the layerwise implementation of the simulation"
                             "of the control loop, to avoid the use of "
                             "sparse matrices, which gives problems on some"
                             "clusters. This does exactly the same computations,"
                             "only in a less efficient manner.")
    dgroup.add_argument('--include_non_converged_samples', action='store_true',
                        help='Include all the minibatch samples to update '
                             'the weights, irrespectively from whether they '
                             'are converged or not. If this flag is not activated,'
                             'only converged samples will be used for '
                             'computing the weight updates. If noisy_dynamics'
                             'is used, this flag should be True, as the noise '
                             'prevents the samples from converging exactly.')
    dgroup.add_argument('--low_pass_filter_u', action='store_true',
                        help='Low-pass filter the controller signal u '
                             'to be used in the neuronal dynamics.')
    dgroup.add_argument('--not_low_pass_filter_r', action='store_true',
                        help='Dont low-pass filter the target rates r for the '
                             'forward weight updates when having noisy dynamics.')
    dgroup.add_argument('--not_high_pass_filter_u_fb', action='store_true',
                        help='Dont high pass filter the control signal when '
                             'updating the feedback weights in single-phase '
                             'experiments.')
    dgroup.add_argument('--tau_f', type=float, default=dtau_f,
                        help='constant of exponential filter for voltage and '
                             'controller.')
    dgroup.add_argument('--tau_noise', type=float, default=dtau_noise,
                        help='constant of exponential filter for npise.')
    dgroup.add_argument('--error_as_loss_grad', action='store_true',
                        help='Compute the error e(t) as the gradient of the '
                             'loss with respect to the output activations, '
                             'instead of always using e(t) = r_L* - r_L(t). '
                             'This option does not change anything when using '
                             'MSE errors (since both are equivalent), but '
                             'has an impact in classification tasks.')
    dgroup.add_argument('--stability_tricks', action='store_true',
                        help='Flag indicating whether stability tricks are '
                             'used for feedback weights pre-training: '
                             'discarding first T/x timesteps of the controller'
                             'dynamcis.')
    dgroup.add_argument('--scaling_fb_updates', action='store_true',
                        help='Flag indicating whether the feedback weight '
                             'gradients are layerwise scalled before the '
                             'update if performed. For DFC_single_phase this '
                             'options is always True.')
    dgroup.add_argument('--not_scaling_fb_updates_single_phase', action='store_true',
                        help='Flag indicating whether the feedback weight '
                             'gradients are NOT layerwise scalled before the '
                             'update if performed for DFC_single_phase.')
    dgroup.add_argument('--at_steady_state', action='store_true',
                        help='Flag indicating whether jacobians of the network for ' 
                             'condition_2 should be calculated at steady state. ')
    dgroup.add_argument('--average_ss', action='store_true',
                        help='Flag indicating whether jacobians of the network for ' 
                             'condition_2 should be calculated using the average of the '
                             'r_target activations for the last quarter of the simulation. ')
    dgroup.add_argument('--strong_dfc_2phase', action='store_true',
                        help='To be used in combination with DFC_single_phase, to enable '
                             'Strong-DFC with two separate phases.')
    dgroup.add_argument('--decay_rate', type=float, default=ddecay_rate,
                        help='decay_rate of the filtering of the spikes '
                             'controller.')
    dgroup.add_argument('--stdp_samples', type=int, default=dstdp_samples,
                        help='number of samples used for the computation of '
                             'the mean of STDP updates to be used.')
    
    return dgroup


def post_process_args(args):
    """ Post process the command line arguments
    Args:
        args (argparse.Namespace): The config object containing all
            command line arguments
    """
    
    if args.test:
        args.epochs = 1
        args.epochs_fb = 1
        args.extra_fb_epochs = 1

    args.double_precision = not args.single_precision

    if args.apical_time_constant == -1:
        args.apical_time_constant = args.dt_di

    if args.time_constant_ratio_fb == -1:
        args.time_constant_ratio_fb = args.time_constant_ratio

    if args.tmax_di_fb == -1:
        args.tmax_di_fb = args.tmax_di

    if args.dt_di_fb == -1:
        args.dt_di_fb = args.dt_di

    if args.apical_time_constant_fb == -1:
        args.apical_time_constant_fb = args.dt_di_fb

    if args.sigma_output == -1:
        args.sigma_output = args.sigma

    if args.sigma_fb == -1:
        args.sigma_fb = args.sigma

    if args.sigma_output_fb == -1:
        args.sigma_output_fb = args.sigma_fb

    if args.time_constant_ratio < args.dt_di:
        print('Time constant ratio is smaller than dt_di, it will be '
              'set equal to dt_di to avoid instabilities.')
        args.time_constant_ratio = args.dt_di

    if args.time_constant_ratio_fb < args.dt_di_fb:
        print('Time constant ratio fb is smaller than dt_di_fb, it will be '
              'set equal to dt_di_fb to avoid instabilities.')
        args.time_constant_ratio_fb = args.dt_di_fb
    
    if args.tau_f >= 1:
        raise ValueError('tau_f for exponential filtering must be smaller than 1.')

    if args.tau_noise >= args.tau_f:
        raise ValueError('tau_noise for exponential filtering must be smaller than tau_f.')

    # convert to int needed parameters
    args.epochs_fb = int(args.epochs_fb)
    args.extra_fb_epochs = int(args.extra_fb_epochs)
    args.tmax_di_fb = int(args.tmax_di_fb)
    args.stdp_samples = int(args.stdp_samples)

    if args.out_dir is None or args.out_dir == 'None':
        args.out_dir = './logs/runs/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if args.lr_fb_init is None or args.lr_fb_init == 'None':
        args.lr_fb_init = args.lr_fb
    if args.k_p_fb is None or args.k_p_fb == 'None':
        args.k_p_fb = args.k_p

    curdir = os.path.curdir
    out_dir = os.path.join(curdir, args.out_dir)
    args.out_dir = out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Logging at {}".format(out_dir))

    if args.network_type == 'DFA':
        args.freeze_fb_weights = True
        args.fb_activation = 'linear'

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(out_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(out_dir, 'args.pickle'), 'wb') as f:
        pickle.dump(args, f)

    if args.dataset in ['mnist', 'fashion_mnist', 'cifar10']:
        args.classification = True
    else:
        args.classification = False

    if args.dataset in ['student_teacher']:
        args.regression = True
    else:
        args.regression = False

    if args.dataset in ['mnist_autoencoder']:
        args.autoencoder = True
    else:
        args.autoencoder = False

    if args.output_activation is None:
        if args.classification:
            args.output_activation = 'softmax'
        elif args.regression:
            args.output_activation = 'linear'
        elif args.autoencoder:
            args.output_activation = 'linear'
        else:
            raise ValueError('Dataset {} is not supported.'.format(
                args.dataset))

    if args.optimizer_fb is None:
        args.optimizer_fb = args.optimizer

    args.lr = utils.process_lr(args.lr)
    args.lr_fb = utils.process_lr(args.lr_fb)
    args.lr_fb_init = utils.process_lr(args.lr_fb_init)
    args.epsilon_fb = utils.process_lr(args.epsilon_fb)
    args.epsilon = utils.process_lr(args.epsilon)
    args.size_hidden = utils.process_hdim(args.size_hidden)
    args.teacher_size_hidden = utils.process_hdim(args.teacher_size_hidden)
    args.hidden_activation = utils.process_hidden_activation(args.hidden_activation)

    if args.normalize_lr:
        args.lr = args.lr/args.target_stepsize

    if args.network_type in ['TPDI', 'RTTP', 'DFC', 'DFC_single_phase']:
        args.direct_fb = True
    else:
        args.direct_fb = False

    if ',' in args.gn_damping:
        args.gn_damping = utils.str_to_list(args.gn_damping, type='float')
    else:
        args.gn_damping = float(args.gn_damping)

    if isinstance(args.size_hidden, int):
        args.size_hidden = [args.size_hidden] * args.num_hidden
    
    if isinstance(args.teacher_size_hidden, int):
        args.teacher_size_hidden = [args.teacher_size_hidden] * args.teacher_num_hidden

    if isinstance(args.hidden_activation, list):
        assert len(args.hidden_activation) == args.num_hidden

    if args.network_type == 'MN':
        args.freeze_fb_weights = True
        args.ndi = True
        args.use_initial_activations = True
        if args.learning_rule == 'nonlinear_difference':
            args.learning_rule = 'derivative_matrix'
    
    if args.classification and args.target_stepsize==0.5 and args.target_class_value==1.0:
            args.target_class_value = 0.99

    if args.network_type == 'DFC':
        args.at_steady_state = True
        if args.noisy_dynamics:
            args.average_ss = True
    
    if args.network_type == 'DFC_single_phase':
        #noisy dynamics must be active for single phase training
        # args.noisy_dynamics = True
        args.inst_transmission = True
        args.inst_transmission_fb = True
        args.proactive_controller = True
        args.scaling_fb_updates = True
        if args.not_scaling_fb_updates_single_phase:
            args.scaling_fb_updates = False
        args.simulate_layerwise = True
        if args.use_jacobian_as_fb:
            args.at_steady_state = True
        else:
            args.average_ss = True
        if args.extra_fb_epochs != 0:
            warnings.warn('The number of extra feedback epochs should ideally be '
                          'zero in single-phase experiments!')
        
        if not args.error_as_loss_grad:
            warnings.warn('Setting the error to be computed as the gradient '
                          'of the loss with respect to the output activations.')
            args.error_as_loss_grad = True
        # In the single phase setting, we don't want to have one-hot-encodings.
        if args.classification and args.target_class_value == 1:
            args.target_class_value = 0.99
        if args.ndi:
            raise ValueError('Non-dynamical inversion cannot be used for a '
                             '"DFC_single_phase" network since it uses strong '
                             'feedback and the linearization will thus not '
                             'be accurate.')
        if args.use_jacobian_as_fb and args.save_condition_gn:
            warnings.warn('Feedback weights are being fixed to the network '
                          'jacobian so condition is not computed as '
                          'it always satisfied.')
            args.save_condition_gn = False

    if args.network_type == 'GN':
        args.freeze_fb_weights = True

    if args.network_type == 'BP':
        args.normalize_lr = False
    
    if args.grad_deltav_cont == False:
        warnings.warn('Activate "grad_deltav_cont" to save a activities.')
        args.save_stdp_measures = False

    if args.save_correlations == True and args.use_stdp_updates == False and args.use_diff_hebbian_updates==False:
        warnings.warn('Deactivate "save_correlations" if neither stdp or '
        ' differential Hebbian are being used.')
        args.save_correlations = False


    args.forward_requires_grad = args.save_BP_angle or args.save_GN_angle or \
                                 args.save_GNT_angle or args.save_LU_angle or \
                                 args.save_ratio_angle_ff_fb or \
                                 args.save_fb_values or \
                                 args.gn_damping_hpsearch or \
                                 args.save_nullspace_norm_ratio or \
                                 args.ndi or \
                                 args.save_condition_gn or \
                                 args.compare_with_ndi or \
                                 args.compute_gn_condition_init or \
                                 args.save_GN_GNT_angle or \
                                 args.save_BP_GNT_angle or \
                                 args.save_GNT_ss_no_ss_angle

    args.compute_angles = args.save_BP_angle or args.save_GN_angle or \
                          args.save_GNT_angle or args.save_LU_angle or \
                          args.save_ratio_angle_ff_fb or \
                          args.save_fb_values or \
                          args.gn_damping_hpsearch or \
                          args.save_nullspace_norm_ratio or \
                          args.save_condition_gn or \
                          args.save_GN_GNT_angle or \
                          args.save_BP_GNT_angle or \
                          args.save_GNT_ss_no_ss_angle or \
                          args.save_NDI_angle


    if args.use_jacobian_as_fb:
        args.freeze_fb_weights = True
        args.epochs_fb = 0
        args.extra_fb_epochs = 0

    if args.ndi:
        if args.save_NDI_angle or args.compare_with_ndi:
            print('NDI is already active, does not make sense to compare it with itself.')
        if args.save_eigenvalues_bcn:
            args.save_eigenvalues_bcn = False
            args.save_eigenvalues = True

    if args.plot_autoencoder_images:
        assert args.dataset == 'mnist_autoencoder'
        if not os.path.exists(os.path.join(args.out_dir, 'autoencoder_images')):
            os.makedirs(os.path.join(args.out_dir, 'autoencoder_images'))

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(out_dir, 'args_postprocessed.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(os.path.join(out_dir, 'args_postprocessed.pickle'), 'wb') as f:
        pickle.dump(args, f)

def check_invalid_args(args):
    """
    Check for invalid combinations of command line arguments.
    Args:
        args (argparse.Namespace): The config object containing all
            command line arguments
    """

    if args.k_p < 0:
        raise ValueError('Only positive values for k_p are allowed')

    if args.shallow_training:
        if not args.network_type == 'BP':
            raise ValueError('The shallow_training method is only implemented'
                             'in combination with BP. Make sure to set '
                             'the network_type argument on BP.')
    if args.compute_angles:
        assert (args.save_logs or args.save_df), "If an angle is computed, it " \
                                                 "should be saved either to " \
                                                 "Tensorboard and/or a dataframe:" \
                                                 "make sure that --save_logs or " \
                                                 "--save_df is True."

    if args.target_class_value <= 0 or args.target_class_value > 1:
        raise ValueError('Target values for the correct class need to be '
                         'in the ]0, 1] range.')

    if args.dont_reuse_training_samples and args.dataset != 'student_teacher':
        warnings.warn('The option "dont_reuse_training_samples" is only '
                      'implemented for the student-teacher dataset.')
