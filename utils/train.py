"""
Collection of train and test functions.
"""

import os
from argparse import Namespace
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
import pandas as pd
from torchvision.utils import save_image
from utils import utils
import pickle
import time
from sklearn.metrics import mean_squared_error
from hypnettorch.utils import torch_ckpts as ckpts
import warnings
from utils import builders

def train(args, device, train_loader, net, writer, test_loader, summary, val_loader,
          logger):
    """
    Train the given network on the given training dataset with DTP.
    Args:
        args (Namespace): The command-line arguments.
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (DTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
        summary (dict): summary dictionary with the performance measures of the
            training and testing
        val_loader (torch.utils.data.DataLoader): The data handler for the
            validation data
        logger: The logger.
    """

    # Save initial weights distribution
    if args.save_correlations:
        # create directory if necessary
        out_dir = 'logs/adversarial_measures/weights/'
        file_name = 'normal_training_initial_weights.pickle'
        exits = os.path.exists(out_dir)
        if not exits:
            os.makedirs(out_dir)
        # saving the weights of the model 
        weights = []
        for l in range(len(net.weights)):
            w = torch.flatten(net.layers[l].weights.detach().cpu())
            weights.append(w.numpy())
        with open(os.path.join(out_dir, file_name), 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    if args.test:
        logger.info('Option "test" is active. This is a dummy run!')

    logger.info('Training network ...')
    net.train()
    net.zero_grad()

    train_var = Namespace()
    train_var.summary = summary
    train_var.forward_optimizer, train_var.feedback_optimizer, \
    train_var.feedback_init_optimizer = utils.choose_optimizer(args, net)
    if args.classification:
        if args.output_activation == 'softmax':
            if args.sum_reduction:
                train_var.loss_function = utils.cross_entropy_fn(reduction='sum')
            else:
                train_var.loss_function = utils.cross_entropy_fn()
            loss_function_name = 'cross_entropy'
        else:
            raise ValueError('MNIST can only be learned with a softmax output activation.')

    elif args.regression or args.autoencoder:
        if args.sum_reduction:
            train_var.loss_function = nn.MSELoss(reduction='sum')
        else:
            train_var.loss_function = nn.MSELoss()
        loss_function_name = 'mse'
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))

    # If the error needs to be computed as the gradient of the loss within 
    # the network, provide the name of the loss function being used. 
    if hasattr(args, 'error_as_loss_grad') and args.error_as_loss_grad:
        net.loss_function_name = loss_function_name

    train_var.batch_idx = 1
    train_var.batch_idx_fb = 1
    train_var.init_idx = 1
    train_var.total_epoch_idx = 1

    train_var.epoch_losses = np.array([])
    train_var.epoch_times = np.array([])
    train_var.epoch_losses_lu = np.array([])
    
    train_var.test_losses = np.array([])
    train_var.val_losses = np.array([])

    train_var.val_loss = None
    train_var.val_accuracy = None

    train_var.epoch_losses_lu = np.array([])

    if args.compute_gn_condition_init and args.epochs_fb > 0:
        train_var.gn_condition_init = np.array([])
    else:
        train_var.gn_condition_init = -1

    if args.classification:
        train_var.epoch_accuracies = np.array([])
        train_var.test_accuracies = np.array([])
        train_var.val_accuracies = np.array([])
    if args.save_convergence:
        train_var.converged = np.array([])
        train_var.not_converged = np.array([])
        train_var.diverged = np.array([])
    if args.compare_with_ndi:
        train_var.rel_dist_to_NDI_vector = np.array([])
    if args.save_eigenvalues:
        train_var.max_eig_mean_vector = np.array([])
        train_var.max_eig_std_vector = np.array([])
    if args.save_eigenvalues_bcn:
        train_var.max_eig_bcn_mean_vector = np.array([])
        train_var.max_eig_bcn_std_vector = np.array([])
    if args.save_norm_r:
        train_var.norm_r_mean_vector = np.array([])
        train_var.norm_r_std_vector = np.array([])
        train_var.dev_r_mean_vector = np.array([])
        train_var.dev_r_std_vector = np.array([])
    if args.save_stdp_measures:
        train_var.pre_activities_vector = np.array([])
        train_var.delta_post_activities_vector = np.array([])
        train_var.weights_vector = np.array([])
        train_var.weights_updates_vector = np.array([])
        train_var.weights_updates_vector_stdp = np.array([])
        train_var.weights_updates_vector_diff_hebbian = np.array([])
        train_var.weights_updates_vector_dfc = np.array([])
        train_var.weights_updates_vector_bp = np.array([])
        train_var.central_activities_vector = np.array([])
        train_var.ff_activities_vector = np.array([])
        train_var.true_labels_vector = np.array([])
        train_var.inputs_vector = np.array([])
        train_var.time_to_convergence = np.zeros(args.epochs)
        train_var.error_time_to_convergence = np.zeros((args.epochs,int(args.tmax_di)-1))
    if args.save_correlations:
        train_var.correlations_dfc = np.array([])
        train_var.correlations_dfc_diff_hebbian = np.array([])
        train_var.correlations_bp = np.array([])
    if args.teacher_linear:
        train_var.mse_with_lss = np.array([])

    if args.dataset == 'student_teacher':
        # transform train_loader back into unique dataset (X,Y) (currently, it's divided in batches)
        X_train = np.empty((args.batch_size*len(train_loader),net.n_in))
        Y_train = np.empty((args.batch_size*len(train_loader),net.n_out))
        for i, (inputs, targets) in enumerate(train_loader):
            for j in range(len(inputs)):
                X_train[i+j] = inputs[j]
                Y_train[i+j] = targets[j]
    
    if args.freeze_fb_weights:
        logger.info("Feedback weights are not being trained.")
    elif args.epochs_fb == 0 and args.pretrained_net_dir == None:
        logger.info("Feedback weights are not being pre-trained.")
    else:
        if args.pretrained_net_dir is not None:
            logger.info("Network with pre-trained feedback weights is being loaded.")
            cpt = ckpts.load_checkpoint(args.pretrained_net_dir, net, device=device)
            if cpt['net_state'] != 'pretrained':
                warnings.warn('Skipping network pre-training, but the indicated checkpoint ' +
                              'is in a %s and not in a pre-trained state.' % cpt['net_state'])

            if 'gn_condition_init' in cpt.keys():
                train_var.summary['gn_condition_init'] =  cpt['gn_condition_init']
            else:
                train_var.summary['gn_condition_init'] =  None

            net.to(device)
        else:
            logger.info("Feedback weights are being pre-trained.")

            train_var.epochs_init = 0
            for e_fb in range(args.epochs_fb):
                logger.info('Feedback weights training: epoch {}'.format(e_fb))
                train_var.epochs_init = e_fb
                if args.compute_gn_condition_init and e_fb == args.epochs_fb-1:
                    compute_gn_condition = True
                else:
                    compute_gn_condition = False

                train_only_feedback_parameters(args, train_var, device, train_loader,
                                               net, writer,
                                               compute_gn_condition=compute_gn_condition,
                                               init=True)
                
                if 'DFC' in args.network_type and args.save_condition_gn:
                    condition_gn = net.compute_condition_two(retain_graph=True)
                    logger.info(f'Condition 2: {condition_gn}')

                if net.contains_nans():
                    logger.info('Network contains NaNs: terminating training.')
                    break

                train_var.total_epoch_idx += 1
                if args.dont_reuse_training_samples:
                    train_loader, _, _ = builders.get_student_teacher_dataset(\
                            args, device,
                            input_data_seed=train_var.total_epoch_idx)

            logger.info(f'Feedback weights initialization done after {args.epochs_fb} epochs.')

            train_var.summary['gn_condition_init'] = np.mean(train_var.gn_condition_init)

            if args.train_only_feedback_parameters:
                logger.info('Terminating training.')
                return train_var.summary

        # Save the pre-trained network.
        if args.save_checkpoints:
            ckpts.save_checkpoint({'state_dict': net.state_dict, 'net_state': 'pretrained',
                                   'gn_condition_init': train_var.summary['gn_condition_init']},
                                   os.path.join(args.out_dir, 'ckpts/pretraining'), None)

    train_var.epochs = 0
        
    for e in range(args.epochs):
        train_var.epochs = e
        net.epoch = e

        # Reset the optimizer if required. This might enable more efficient
        # learning late in training, as a kind of learning rate scheduler.
        if args.opt_reset_interval != -1 and e % args.opt_reset_interval == 0:
            train_var.forward_optimizer, train_var.feedback_optimizer, \
            train_var.feedback_init_optimizer = utils.choose_optimizer(args, net)

        if args.reset_K and e > 0:
            net.feedbackweights_initialization()
            logger.info('Resetting forward optimizer.')

        if args.save_convergence:
            net.converged_samples_per_epoch = 0
            net.diverged_samples_per_epoch = 0
            net.not_converged_samples_per_epoch = 0
            net.epoch = e

            if args.make_dynamics_plot and e % args.make_dynamics_plot_interval == 0:
                net.makeplots = True
            else:
                net.makeplots = False

        if args.classification:
            train_var.accuracies = np.array([])
        train_var.losses = np.array([])

        if args.save_lu_loss:
            train_var.losses_lu = np.array([])

        if args.compare_with_ndi:
            net.rel_dist_to_NDI = []
        else:
            net.rel_dist_to_NDI = None

        epoch_initial_time = time.time()
        train_separate(args, train_var, device, train_loader, net, writer)
        
        if not args.freeze_fb_weights and args.lr_fb != 0.:
            for extra_e in range(args.extra_fb_epochs):
                train_only_feedback_parameters(args, train_var, device,
                                                train_loader,
                                                net, writer, log=False)
                train_var.total_epoch_idx += 1
                if args.dont_reuse_training_samples:
                    train_loader, _, _ = builders.get_student_teacher_dataset(\
                            args, device,
                            input_data_seed=train_var.total_epoch_idx)

        if 'DFC' in args.network_type and args.save_condition_gn:
            condition_gn = net.compute_condition_two(retain_graph=True)
            logger.info(f'Condition 2: {condition_gn}')

        if not args.no_val_set:
            train_var.val_accuracy, train_var.val_loss = \
                test(args, device, net, val_loader,
                         train_var.loss_function, train_var)

        train_var.test_accuracy, train_var.test_loss = \
            test(args, device, net, test_loader,
                     train_var.loss_function, train_var)

        train_var.epoch_time = time.time() - epoch_initial_time

        train_var.epoch_loss = np.mean(train_var.losses)
        train_var.epoch_loss_lu = None
        
        if args.save_lu_loss:
            train_var.epoch_loss_lu = np.mean(train_var.losses_lu)

        logger.info('Epoch {} '.format(e + 1))
        logger.info('\ttraining loss = {}'.format(np.round(train_var.epoch_loss, 6)))
        if not args.no_val_set:
            logger.info('\tval loss = {}'.format(np.round(train_var.val_loss, 6)))
        logger.info('\ttest loss = {}'.format(np.round(train_var.test_loss, 6)))

        if args.classification:
            train_var.epoch_accuracy = np.mean(train_var.accuracies)
            logger.info('\ttraining acc  = {} %'.format(np.round(train_var.epoch_accuracy * 100, 6)))
            if not args.no_val_set:
                logger.info('\tval acc  = {} %'.format(np.round(train_var.val_accuracy * 100, 6)))
            logger.info('\ttest acc  = {} %'.format(np.round(train_var.test_accuracy * 100, 6)))
        else:
            train_var.epoch_accuracy = None
        
        if args.teacher_linear and args.dataset=='student_teacher' and args.hpsearch:
            
            # compute the DFC feedforward (ff) weight mapping 
            # ff weights are saved as (n_out, n_in) so need to be transposed for forward multiplicaiton
            W_ff = 1. * torch.transpose(net._layers[0].weights.data, 0, 1)
            for i in range(1, net.depth):
                W_ff = torch.mm(W_ff, torch.transpose(net._layers[i].weights.data, 0, 1))
            # compute the least squares solution of the input-output mapping
            # W_lss = np.linalg.lstsq(X_train,Y_train)
            W_lss = np.dot((np.dot(np.linalg.inv(np.dot(X_train.T,X_train)),X_train.T)),Y_train)
            
            # compute the MSE between the DFC and the lss weights
            mse_lssm = mean_squared_error(W_ff.detach().cpu().numpy(),W_lss)
            
            train_var.mse_with_lss = mse_lssm
            logger.info('\tMSE between forward DFC and LSS mappings = {}'.format(mse_lssm))

        if args.compare_with_ndi:
            logger.info('\tmean distance to ndi  = {}'.format(np.round(np.mean(net.rel_dist_to_NDI), 18)))
        logger.info('\tepoch time = {} seconds'.format(np.round(train_var.epoch_time, 1)))

        if args.save_logs:
            if args.save_convergence: 

                avg_dist_to_NDI = None if not args.compare_with_ndi else np.mean(net.rel_dist_to_NDI)

                utils.save_logs_convergence(writer, step=e + 1, net=net,
                                            loss=train_var.epoch_loss,
                                            epoch_time=train_var.epoch_time,
                                            accuracy=train_var.epoch_accuracy,
                                            test_loss=train_var.test_loss,
                                            val_loss=train_var.val_loss,
                                            test_accuracy=train_var.test_accuracy,
                                            val_accuracy=train_var.val_accuracy,
                                            converged_samples_per_epoch=net.converged_samples_per_epoch,
                                            diverged_samples_per_epoch=net.diverged_samples_per_epoch,
                                            not_converged_samples_per_epoch=net.not_converged_samples_per_epoch,
                                            dist_to_NDI=avg_dist_to_NDI,
                                            loss_lu=train_var.epoch_loss_lu,)
            else:
                utils.save_logs(writer, step=e + 1, net=net,
                                loss=train_var.epoch_loss,
                                epoch_time=train_var.epoch_time,
                                accuracy=train_var.epoch_accuracy,
                                test_loss=train_var.test_loss,
                                val_loss=train_var.val_loss,
                                test_accuracy=train_var.test_accuracy,
                                val_accuracy=train_var.val_accuracy,
                                loss_lu=train_var.epoch_loss_lu,)

        train_var.epoch_losses = np.append(train_var.epoch_losses,
                                           train_var.epoch_loss)
        train_var.test_losses = np.append(train_var.test_losses,
                                          train_var.test_loss)
        train_var.epoch_times = np.append(train_var.epoch_times,
                                          train_var.epoch_time)
        if args.save_lu_loss:
            train_var.epoch_losses_lu = np.append(train_var.epoch_losses_lu,
                                           train_var.epoch_loss_lu)
        if not args.no_val_set:
            train_var.val_losses = np.append(train_var.val_losses,
                                             train_var.val_loss)

        if args.classification:
            train_var.epoch_accuracies = np.append(train_var.epoch_accuracies,
                                                   train_var.epoch_accuracy)
            train_var.test_accuracies = np.append(train_var.test_accuracies,
                                                  train_var.test_accuracy)
            if not args.no_val_set:
                train_var.val_accuracies = np.append(train_var.val_accuracies,
                                                     train_var.val_accuracy)

        if args.save_convergence:
            train_var.converged = np.append(train_var.converged,
                                            net.converged_samples_per_epoch)
            train_var.not_converged = np.append(train_var.not_converged,
                                                net.not_converged_samples_per_epoch)
            train_var.diverged = np.append(train_var.diverged,
                                           net.diverged_samples_per_epoch)

        if args.compare_with_ndi:
            train_var.rel_dist_to_NDI_vector = np.append(train_var.rel_dist_to_NDI_vector,
                                                         np.mean(net.rel_dist_to_NDI))

        utils.save_summary_dict(args, train_var.summary)

        if e == 2:
            if args.gn_damping_hpsearch:
                logger.info('Doing hpsearch for finding ideal GN damping constant'
                      'for computing the angle with GNT updates')
                gn_damping = gn_damping_hpsearch(args, train_var, device,
                                                 train_loader, net, writer, logger)
                args.gn_damping = gn_damping
                logger.info('Damping constants GNT angles: {}'.format(gn_damping))
                train_var.summary['gn_damping_values'] = gn_damping
                return train_var.summary

        # Save to summary.
        train_var.summary['avg_time_per_epoch'] = np.mean(train_var.epoch_times)
        train_var.summary['loss_train_last'] = train_var.epoch_loss
        train_var.summary['loss_test_last'] = train_var.test_loss
        train_var.summary['loss_train_best'] = train_var.epoch_losses.min()
        train_var.summary['loss_test_best'] = train_var.test_losses.min()
        train_var.summary['loss_train'] = train_var.epoch_losses
        train_var.summary['loss_test'] = train_var.test_losses
        best_train_epoch = train_var.epoch_loss.argmin()
        train_var.summary['epoch_best_train_loss'] = best_train_epoch
        if args.save_lu_loss:
            train_var.summary['loss_lu_train_last'] = train_var.epoch_loss_lu
            train_var.summary['loss_lu_train_best'] = train_var.epoch_losses_lu.min() 
            train_var.summary['loss_lu_train'] = train_var.epoch_losses_lu
            train_var.summary['loss_lu_at_best_train'] = \
                    train_var.epoch_losses_lu[best_train_epoch]
        if args.save_LU_angle and args.save_df:
                train_var.summary['lu_angle_last'] = net.lu_angles[0].tolist()[-1]
        if not args.no_val_set:
            train_var.summary['loss_val_last'] = train_var.val_loss
            train_var.summary['loss_val_best'] = train_var.val_losses.min()
            train_var.summary['loss_val'] = train_var.val_losses

        if args.save_convergence:
            train_var.summary['converged'] = train_var.converged
            train_var.summary['not_converged'] = train_var.not_converged
            train_var.summary['diverged'] = train_var.diverged

        if args.save_condition_gn:
            train_var.summary['gn_condition'] = condition_gn.item()

        if args.compare_with_ndi:
            train_var.summary['dist_to_NDI'] = train_var.rel_dist_to_NDI_vector
        else:
            train_var.summary['dist_to_NDI'] = np.array([0])

        if args.save_eigenvalues:
            train_var.summary['max_eig_mean'] = train_var.max_eig_mean_vector
            train_var.summary['max_eig_std'] = train_var.max_eig_std_vector
        else:
            train_var.summary['max_eig_mean'] = np.array([0])
            train_var.summary['max_eig_std'] = np.array([0])
        if args.save_eigenvalues_bcn:
            train_var.summary['max_eig_bcn_mean'] = train_var.max_eig_bcn_mean_vector
            train_var.summary['max_eig_bcn_std'] = train_var.max_eig_bcn_std_vector
        else:
            train_var.summary['max_eig_bcn_mean'] = np.array([0])
            train_var.summary['max_eig_bcn_std'] = np.array([0])

        if args.save_norm_r:
            train_var.summary['norm_r_mean'] = train_var.norm_r_mean_vector
            train_var.summary['norm_r_std'] = train_var.norm_r_std_vector
            train_var.summary['dev_r_mean'] = train_var.dev_r_mean_vector
            train_var.summary['dev_r_std'] = train_var.dev_r_std_vector
        else:
            train_var.summary['norm_r_mean'] = np.array([0])
            train_var.summary['norm_r_std'] = np.array([0])
            train_var.summary['dev_r_mean'] = np.array([0])
            train_var.summary['dev_r_std'] = np.array([0])
        
        if args.save_stdp_measures:
            train_var.summary['pre_activities_vector'] = train_var.pre_activities_vector
            train_var.summary['delta_post_activities_vector'] = train_var.delta_post_activities_vector
            train_var.summary['weights_vector'] = train_var.weights_vector
            train_var.summary['weights_updates_vector'] = train_var.weights_updates_vector
            train_var.summary['weights_updates_vector_stdp'] = train_var.weights_updates_vector_stdp
            train_var.summary['weights_updates_vector_diff_hebbian'] = train_var.weights_updates_vector_diff_hebbian
            train_var.summary['weights_updates_vector_dfc'] = train_var.weights_updates_vector_dfc
            train_var.summary['weights_updates_vector_bp'] = train_var.weights_updates_vector_bp
            train_var.summary['central_activities_vector'] = train_var.central_activities_vector
            train_var.summary['ff_activities_vector'] = train_var.ff_activities_vector
            train_var.summary['true_labels_vector'] = train_var.true_labels_vector
            train_var.summary['inputs_vector'] = train_var.inputs_vector
            train_var.summary['time_to_convergence'] = train_var.time_to_convergence
            train_var.summary['error_time_to_convergence'] = train_var.error_time_to_convergence
        else:
            train_var.summary['pre_activities_vector'] = np.array([0])
            train_var.summary['delta_post_activities_vector'] = np.array([0])
            train_var.summary['weights_vector'] = np.array([0])
            train_var.summary['weights_updates_vector'] = np.array([0])
            train_var.summary['weights_updates_vector_stdp'] = np.array([0])
            train_var.summary['weights_updates_vector_diff_hebbian'] = np.array([0])
            train_var.summary['weights_updates_vector_dfc'] = np.array([0])
            train_var.summary['weights_updates_vector_bp'] = np.array([0])
            train_var.summary['central_activities_vector'] = np.array([0])
            train_var.summary['ff_activities_vector'] = np.array([0])
            train_var.summary['true_labels_vector'] = np.array([0])
            train_var.summary['inputs_vector'] = np.array([0])
            train_var.summary['time_to_convergence'] = np.array([0])
            train_var.summary['error_time_to_convergence'] = np.array([0])
        
        if args.save_correlations:
            train_var.summary['correlations_dfc'] = train_var.correlations_dfc
            train_var.summary['correlations_dfc_diff_hebbian'] = train_var.correlations_dfc_diff_hebbian
            train_var.summary['correlations_bp'] = train_var.correlations_bp
        else:
            train_var.summary['correlations_dfc'] = np.array([0])
            train_var.summary['correlations_dfc_diff_hebbian'] = np.array([0])
            train_var.summary['correlations_bp'] = np.array([0])

        if not args.no_val_set:
            best_epoch = train_var.val_losses.argmin()
            train_var.summary['epoch_best_loss'] = best_epoch
            train_var.summary['loss_test_val_best'] = \
                train_var.test_losses[best_epoch]
            train_var.summary['loss_train_val_best'] = \
                train_var.epoch_losses[best_epoch]

        if args.classification:
            train_var.summary['acc_train_last'] = train_var.epoch_accuracy
            train_var.summary['acc_test_last'] = train_var.test_accuracy
            train_var.summary['acc_train_best'] = train_var.epoch_accuracies.max()
            train_var.summary['acc_test_best'] = train_var.test_accuracies.max()
            train_var.summary['acc_train'] = train_var.epoch_accuracies
            train_var.summary['acc_test'] = train_var.test_accuracies
            if not args.no_val_set:
                train_var.summary['acc_val'] = train_var.val_accuracies
                train_var.summary['acc_val_last'] = train_var.val_accuracy
                train_var.summary['acc_val_best'] = train_var.val_accuracies.max()
                best_epoch = train_var.val_accuracies.argmax()
                train_var.summary['epoch_best_acc'] = best_epoch
                train_var.summary['acc_test_val_best'] = \
                    train_var.test_accuracies[best_epoch]
                train_var.summary['acc_train_val_best'] = \
                    train_var.epoch_accuracies[best_epoch]

        train_var.summary['log_interval'] = args.log_interval
        if args.save_eigenvalues:
            train_var.summary['max_eig_keys'] = [k for k in net.max_eig.keys()]
        elif args.save_eigenvalues_bcn:
            train_var.summary['max_eig_keys'] = [k for k in net.max_eig_bcn.keys()]
        if args.save_norm_r:
            train_var.summary['norm_r_keys'] = [k for k in net.norm_r.keys()]
        if args.teacher_linear and args.dataset=='student_teacher' and args.hpsearch:
            train_var.summary['mse_with_lss'] = train_var.mse_with_lss

        utils.save_summary_dict(args, train_var.summary)
        if args.save_summary_interval != -1 and e % args.save_summary_interval == 0:
            # Every 10 epochs, save a separate summary file that isn't overwritten.
            utils.save_summary_dict(args, train_var.summary, epoch=e)

        # Save the training network.
        if e % 10 == 0 and args.save_checkpoints:
            store_dict = {'state_dict': net.state_dict,
                          'net_state': 'epoch_%i' % e,
                          'train_loss': train_var.epoch_loss,
                          'test_loss': train_var.test_loss}
            if args.classification:
                store_dict['train_acc'] = train_var.epoch_accuracy
                store_dict['test_acc'] = train_var.test_accuracy
            ckpts.save_checkpoint(store_dict, os.path.join(args.out_dir, 'ckpts/training'), None)

        if net.contains_nans():
            logger.info('Network contains NaNs, terminating training.')
            train_var.summary['finished'] = -1
            break

        # if e > 4 and (not args.evaluate):
        # # for stdp experiments accelarate hpsearch
        # # if e > 0 and (not args.evaluate):
        #     if args.dataset in ['mnist', 'fashion_mnist']:
        #         if train_var.epoch_accuracy < 0.3:
        #             logger.info('writing error code -1')
        #             train_var.summary['finished'] = -1
        #             break
        #     if args.dataset in ['cifar10']:
        #         if train_var.epoch_accuracy < 0.25:
        #             logger.info('writing error code -1')
        #             train_var.summary['finished'] = -1
        #             break

            # if e > 10 and args.dataset in ['student_teacher']:
            #     if np.min(train_var.epoch_losses[-10:]) >= np.min(train_var.epoch_losses[:-10]):
            #         logger.info('Loss did not improve in the last 10 epochs')
            #         train_var.summary['finished'] = -1
            #         break
    
    if args.save_stdp_measures:
        for e in range(args.epochs):
            for t in range(int(args.tmax_di)-1):
                train_var.error_time_to_convergence[e,t] = net.error_time_to_convergence[e,t]
                if net.error_time_to_convergence[e,t]<0.2 and train_var.time_to_convergence[e]==0:
                    train_var.time_to_convergence[e] = t

    # Save the final network.
    if args.save_checkpoints:
        store_dict = {'state_dict': net.state_dict,
                      'net_state': 'trained',
                      'train_loss': train_var.epoch_loss,
                      'test_loss': train_var.test_loss}
        if args.classification:
            store_dict['train_acc'] = train_var.epoch_accuracy
            store_dict['test_acc'] = train_var.test_accuracy
        ckpts.save_checkpoint(store_dict, os.path.join(args.out_dir, 'ckpts/final'), None)
    
    # Save final weights distribution
    if args.save_correlations:
        # create directory if necessary
        out_dir = 'logs/adversarial_measures/weights/'
        file_name = 'normal_training_weights.pickle'
        exits = os.path.exists(out_dir)
        if not exits:
            os.makedirs(out_dir)
        # saving the weights of the model 
        weights = []
        for l in range(len(net.weights)):
            w = torch.flatten(net.layers[l].weights.detach().cpu())
            weights.append(w.numpy())
        with open(os.path.join(out_dir, file_name), 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

    logger.info('Training network... Done! :)')
    return train_var.summary


def train_separate(args, train_var, device, train_loader, net, writer):
    """
    Train the given network on the given training dataset with DTP. For each
    epoch, first the feedback weights are trained on the whole epoch, after
    which the forward weights are trained on the same epoch (similar to Lee2105)
    Args:
        args (Namespace): The command-line arguments.
        train_var (Namespace): Structure containing training variables
        device: The PyTorch device to be used
        train_loader (torch.utils.data.DataLoader): The data handler for
            training data
        net (LeeDTPNetwork): The neural network
        writer (SummaryWriter): TensorboardX summary writer to save logs
        test_loader (DataLoader): The data handler for the test data
    """

    if not args.freeze_fb_weights and args.lr_fb != 0. and args.network_type != 'DFC_single_phase':
        for i, (inputs, targets) in enumerate(train_loader):
            if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()

            predictions = net.forward(inputs)

            train_feedback_parameters(args, net, train_var.feedback_optimizer, predictions, targets, train_var.loss_function)

            if (args.save_logs or args.save_df) and i % args.log_interval == 0:
                if args.save_condition_gn or args.save_jac_t_angle or args.save_jac_pinv_angle:
                    utils.save_batch_logs(args, writer,
                                          train_var.batch_idx_fb, net,
                                          init=False, weights='feedback')

            train_var.batch_idx_fb += 1

            if args.test and i == 1:
                break

        train_var.total_epoch_idx += 1
        if args.dont_reuse_training_samples:
            train_loader, _, _ = builders.get_student_teacher_dataset(\
                    args, device, input_data_seed=train_var.total_epoch_idx)

    for i, (inputs, targets) in enumerate(train_loader):
        if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        if args.autoencoder:
            targets = inputs.detach()
        if net.epoch<=20 and args.surprise_shuffle:
            for i in range(len(targets)):
                if torch.argmax(targets[i]) != 9:
                    targets[i] = torch.ones(10)*0.0011
                    targets[i, torch.argmax(targets[i])+1] = 0.9900
                else:
                    targets[i] = torch.ones(10)*0.0011
                    targets[i, 0] = 0.9900

        predictions = net.forward(inputs)

        if i % args.log_interval == 0:

            if args.save_eigenvalues:       net.save_eigenvalues = True
            else:                           net.save_eigenvalues = False

            if args.save_eigenvalues_bcn:   net.save_eigenvalues_bcn = True
            else:                           net.save_eigenvalues_bcn = False

            if args.save_norm_r:            net.save_norm_r = True
            else:                           net.save_norm_r = False

            if args.save_stdp_measures:     net.save_stdp_measures = True
            else:                           net.save_stdp_measures = False

            if args.save_correlations:      net.save_correlations = True
            else:                           net.save_correlations = False

            if args.save_NDI_angle:         net.save_NDI_updates = True
            else:                           net.save_NDI_updates = False

        else:
            net.save_eigenvalues = False
            net.save_eigenvalues_bcn = False
            net.save_norm_r = False
            net.save_stdp_measures = False
            net.save_correlations = False
            net.save_NDI_updates = False


        train_var.batch_accuracy, train_var.batch_loss, train_var.batch_loss_lu = \
            train_forward_parameters(args, net, predictions, targets,
                                     train_var.loss_function,
                                     train_var.forward_optimizer, 
                                     feedback_optimizer=train_var.feedback_optimizer, 
                                     writer=writer,
                                     step=train_var.batch_idx_fb)

        if args.network_type== 'DFC_single_phase' and args.strong_dfc_2phase and (not args.freeze_fb_weights):
            train_feedback_parameters(args, net, train_var.feedback_optimizer, predictions, targets,
                                      train_var.loss_function, init=True)

        if args.classification:
            train_var.accuracies = np.append(train_var.accuracies,
                                             train_var.batch_accuracy)
        train_var.losses = np.append(train_var.losses,
                                     train_var.batch_loss.item())
        if args.save_lu_loss:
            train_var.losses_lu = np.append(train_var.losses_lu,
                                     train_var.batch_loss_lu.item())

        if i % args.log_interval == 0:
            if args.save_logs or args.save_df:
                if args.compute_angles:
                    utils.save_angles(args, writer, train_var.batch_idx, net, train_var.batch_loss, predictions)
                if args.save_eigenvalues:
                    net.save_eigenvalues_to_tensorboard(writer, train_var.batch_idx)
                if args.save_eigenvalues_bcn:
                    net.save_eigenvalues_bcn_to_tensorboard(writer, train_var.batch_idx)
                if args.save_norm_r and args.save_logs:
                    net.save_norm_r_to_tensorboard(writer, train_var.batch_idx)
                weights_to_log = 'forward'
                if args.freeze_fb_weights or args.network_type == 'DFC_single_phase':
                    # In these two cases, the loop above for the feedback
                    # weights hasn't run, so we need to do the logs here.
                    weights_to_log = 'both'
                if args.save_condition_gn or args.save_jac_t_angle or \
                                                     args.save_jac_pinv_angle:
                    utils.save_batch_logs(args, writer, train_var.batch_idx,
                                          net, init=False,
                                          weights=weights_to_log)

            if args.save_eigenvalues:
                train_var.max_eig_mean_vector, train_var.max_eig_std_vector = \
                    summary_vector_append_mean_std(net.max_eig,
                                                   train_var.max_eig_mean_vector,
                                                   train_var.max_eig_std_vector)
            if args.save_eigenvalues_bcn:
                train_var.max_eig_bcn_mean_vector, train_var.max_eig_bcn_std_vector = \
                    summary_vector_append_mean_std(net.max_eig_bcn,
                                                   train_var.max_eig_bcn_mean_vector,
                                                   train_var.max_eig_bcn_std_vector)
            if args.save_norm_r:
                train_var.norm_r_mean_vector, train_var.norm_r_std_vector = \
                    summary_vector_append_mean_std(net.norm_r,
                                                   train_var.norm_r_mean_vector,
                                                   train_var.norm_r_std_vector)
                train_var.dev_r_mean_vector = np.append(train_var.dev_r_mean_vector,
                                                        np.mean(net.dev_r))
                train_var.dev_r_std_vector = np.append(train_var.dev_r_std_vector,
                                                       np.std(net.dev_r))


        train_var.batch_idx += 1

        if not args.freeze_forward_weights:
            train_var.forward_optimizer.step()
        if not args.freeze_fb_weights and args.lr_fb != 0. and args.network_type == 'DFC_single_phase':
            if not args.strong_dfc_2phase:
                train_var.feedback_optimizer.step()

        if args.test and i == 1:
            break
        
        if args.save_stdp_measures and (net.epoch==args.save_epoch or (args.save_epoch==None and net.epoch==args.epochs-1)) and i==len(train_loader)-2:
            train_var.true_labels_vector = targets
            train_var.inputs_vector = inputs
        
    if args.save_stdp_measures and (net.epoch==args.save_epoch or (args.save_epoch==None and net.epoch==args.epochs-1)):
        train_var.pre_activities_vector = net.pre_activities
        train_var.delta_post_activities_vector = net.delta_post_activities
        train_var.weights_vector = net.weights
        train_var.weights_updates_vector = net.weights_updates
        train_var.weights_updates_vector_stdp = net.weights_updates_stdp
        train_var.weights_updates_vector_diff_hebbian = net.weights_updates_diff_hebbian
        train_var.weights_updates_vector_dfc = net.weights_updates_dfc
        train_var.weights_updates_vector_bp = net.weights_updates_bp
        train_var.central_activities_vector = net.central_activities
        train_var.ff_activities_vector = net.ff_activities
    
    if args.save_correlations and net.epoch==args.epochs-1:
        train_var.correlations_dfc = net.dfc_angles_list
        train_var.correlations_dfc_diff_hebbian = net.dfc_diff_hebbian_angles_list
        train_var.correlations_bp = net.correlations_bp
    
    train_var.total_epoch_idx += 1
    if args.dont_reuse_training_samples:
        train_loader, _, _ = builders.get_student_teacher_dataset(\
                args, device, input_data_seed=train_var.total_epoch_idx)

def summary_vector_append_mean_std(batch_dict, vector_mean, vector_std):
    """
    Takes the results from one batch (of max eig or norm_r)
    and appends n values of mean and std (where n is the number
    of keys in batch dict).
    :param: batch_dict : dictionary containing results for one metric for one batch
    :param: vector_mean: vector of means to which the means of batch_dict data will be appendend
    :param: vector_std: vector of means to which the standard deviations of batch_dict data will be appendend
    :return: `vector_mean`, `vector_std` with the new values appended
    """

    keys = [k for k in batch_dict.keys()]
    mean = np.zeros((len(keys), 1))
    std = np.zeros((len(keys), 1))
    for i, k in enumerate(keys):
        mean[i] = np.mean(batch_dict[k])
        std[i] = np.std(batch_dict[k])
    vector_mean = np.append(vector_mean, mean)
    vector_std = np.append(vector_std, std)
    return vector_mean, vector_std


def train_forward_parameters(args, net, predictions, targets, loss_function,
                                 forward_optimizer, writer=None, step=None,
                                 feedback_optimizer=None):
    """
    Train the forward parameters on the current mini-batch.
    """

    if predictions.requires_grad == False:
        predictions.requires_grad = True

    save_target = args.compute_angles
    forward_optimizer.zero_grad()
    if args.network_type == 'DFC_single_phase':
        feedback_optimizer.zero_grad()

    loss = loss_function(predictions, targets)

    net.backward(loss, targets, args.target_stepsize, save_target=save_target,
                 writer=writer if args.save_logs and args.plot_filters \
                 else None, step=step)
                 
    if args.save_lu_loss:
        batch_loss_lu = utils.loss_function_lu(args, net, args.use_jacobian_as_fb)

    if args.use_bp_updates:
        net.set_grads_to_bp(loss, retain_graph=True)

    if args.classification:
        batch_accuracy = utils.accuracy(predictions, targets)
    else:
        batch_accuracy = None
    if args.sum_reduction: # take the mean of the loss,
        # for fair comparison when mean_reduction is used (for BP)
        batch_loss = loss/args.batch_size
        if args.regression or args.autoencoder:
            # divide by number of output neurons
            batch_loss = batch_loss/args.size_output
        if args.save_lu_loss:
            # divide by total number of neurons in the network (except input)
            total_num_neurons = np.sum([net.layers[i].activations.shape[1] for i in range(net.depth)])
            batch_loss_lu = batch_loss_lu/total_num_neurons
    else:
        batch_loss = loss
        # if args.save_lu_loss:
        #     batch_loss_lu = loss_lu

    if args.sum_reduction: # take the mean of the loss,
        # for fair comparison when mean_reduction is used (for BP)
        batch_loss = loss/args.batch_size
        if args.regression or args.autoencoder:
            batch_loss = batch_loss/args.size_output
    else:
        batch_loss = loss
    
    if not args.save_lu_loss:
        batch_loss_lu = None

    if args.fix_grad_norm > 0:
        utils.fix_grad_norm_(forward_optimizer.parameters,
                             fixed_norm=args.fix_grad_norm,
                             norm_type=2.)
        if args.network_type == 'DFC_single_phase':
            utils.fix_grad_norm_(feedback_optimizer.parameters,
                                 fixed_norm=args.fix_grad_norm,
                                 norm_type=2.)
    if net.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(net.get_forward_parameter_list(),
                                       max_norm=net.clip_grad_norm)
        if args.network_type == 'DFC_single_phase':
            torch.nn.utils.clip_grad_norm_(net.get_feedback_parameter_list(),
                                           max_norm=net.clip_grad_norm)


    return batch_accuracy, batch_loss, batch_loss_lu


def train_feedback_parameters(args, net, feedback_optimizer, predictions, targets, loss_function, init=False):
    """
    Train the feedback parameters on the current mini-batch.
    """

    loss = loss_function(predictions, targets)

    feedback_optimizer.zero_grad()
    
    net.compute_feedback_gradients(loss, targets, args.target_stepsize, init=init)

    if args.fix_grad_norm > 0:
        utils.fix_grad_norm_(feedback_optimizer.parameters,
                                 fixed_norm=args.fix_grad_norm,
                                 norm_type=2.)

    if net.clip_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(net.get_feedback_parameter_list(),
                                       max_norm=net.clip_grad_norm)

    feedback_optimizer.step()


def test(args, device, net, test_loader, loss_function, train_var=None):
    """
    Compute the test loss and accuracy on the test dataset
    Args:
        args: command line inputs
        net: network
        test_loader (DataLoader): dataloader object with the test dataset
    Returns: Tuple containing:
        - Test accuracy
        - Test loss
    """

    loss = 0
    if args.save_lu_loss:
        loss_lu = 0
    if args.classification:
        accuracy = 0
    
    nb_batches = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):

            if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()

            predictions = net.forward(inputs)

            loss += loss_function(predictions, targets).item()
            
            if args.classification:
                accuracy += utils.accuracy(predictions, targets)

            if args.plot_autoencoder_images and i == 0:
                input_images = torch.reshape(inputs,
                                             (args.batch_size, 1, 28, 28))[0:5]
                reconstructed_images = torch.reshape(predictions, (
                    args.batch_size, 1, 28, 28))[0:5]
                save_image(input_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/input_epoch_{}.png'.format(
                        train_var.epochs)))
                save_image(reconstructed_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/reconstructed_epoch_{}.png'.format(
                        train_var.epochs)))
            nb_batches += 1

            if args.test and i == 1:
                break

    loss /= nb_batches
    if args.sum_reduction: # take the mean of the loss,
        # for fair comparison when mean_reduction is used (for BP)
        loss /= args.batch_size
        if args.regression or args.autoencoder:
            loss /= args.size_output

    if args.classification:
        accuracy /= nb_batches
    else:
        accuracy = None

    return accuracy, loss


def train_only_feedback_parameters(args, train_var, device, train_loader,
                                     net, writer, log=True,
                                     compute_gn_condition=False, init=False):
    """
    Train only the feedback parameters for the given amount of epochs.
    This function is used to initialize the network in a 'pseudo-inverse'
    condition.
    """

    for i, (inputs, targets) in enumerate(train_loader):
        if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
            inputs, targets = inputs.to(device), targets.to(device)
        if not args.network_type == 'DDTPConv':
            inputs = inputs.flatten(1, -1)
        if args.autoencoder:
            targets = inputs.detach()

        predictions = net.forward(inputs)

        train_feedback_parameters(args, net, train_var.feedback_init_optimizer, predictions, targets, train_var.loss_function, init)
        
        if (args.save_logs or args.save_df) and i % args.log_interval == 0:
            if args.save_condition_gn or args.save_jac_t_angle or args.save_jac_pinv_angle:
                if init:
                    utils.save_batch_logs(args, writer, train_var.init_idx, net,
                          init=init, statistics=args.save_fb_statistics_init,
                          weights='feedback')
                    train_var.init_idx += 1

        if compute_gn_condition and i % args.log_interval == 0:
            gn_condition = net.compute_condition_two(retain_graph=False)
            train_var.gn_condition_init = np.append(train_var.gn_condition_init,
                                                    gn_condition.item())

        if args.test and i == 1:
            break

def train_bp(args, device, train_loader, net, writer, test_loader, summary,
                 val_loader, logger):
    logger.info('Training network ...')
    net.train()
    forward_optimizer = utils.OptimizerList(args, net)

    if args.classification:
        if args.output_activation == 'softmax':
            loss_function = utils.cross_entropy_fn()
        else:
            raise ValueError('The mnist dataset can only be combined with a '
                             'softmax output activation.')

    elif args.regression or args.autoencoder:
        loss_function = nn.MSELoss()
    else:
        raise ValueError('The provided dataset {} is not supported.'.format(
            args.dataset
        ))

    epoch_losses = np.array([])
    test_losses = np.array([])
    val_losses = np.array([])
    val_loss = None
    val_accuracy = None

    if args.classification:
        epoch_accuracies = np.array([])
        test_accuracies = np.array([])
        val_accuracies = np.array([])
    
    for e in range(args.epochs):
        nb_batches = 0
        if args.classification:
            running_accuracy = 0
        else:
            running_accuracy = None
        running_loss = 0

        # Reset the optimizer if required. This might enable more efficient
        # learning late in training, as a kind of learning rate scheduler.
        if args.opt_reset_interval != -1 and e % args.opt_reset_interval == 0:
            forward_optimizer = utils.OptimizerList(args, net)

        for i, (inputs, targets) in enumerate(train_loader):
            
            if args.dataset not in ['mnist', 'fashion_mnist', 'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'BPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()

            forward_optimizer.zero_grad()
            predictions = net(inputs)
            loss = loss_function(predictions, targets)
            loss.backward()

            if args.fix_grad_norm > 0:
                utils.fix_grad_norm_(net.parameters(),
                                     fixed_norm=args.fix_grad_norm,
                                     norm_type=2.)

            forward_optimizer.step()

            running_loss += loss.item()

            if args.classification:
                running_accuracy += utils.accuracy(predictions, targets)

            nb_batches += 1

            if args.test and i == 1:
                break

            if args.save_stdp_measures and i==len(train_loader)-2:
                summary['true_labels_vector'] = targets
                summary['inputs_vector'] = inputs
                summary['ff_activities_vector'] = net.dummy_forward(inputs, args.batch_size)

        if not args.no_val_set:
            val_accuracy, val_loss = test_bp(args, device, net, val_loader,
                                                 loss_function, epoch=e)

        test_accuracy, test_loss = test_bp(args, device, net, test_loader,
                                               loss_function, epoch=e)

        epoch_loss = running_loss/nb_batches
        if args.classification:
            epoch_accuracy = running_accuracy/nb_batches
        else:
            epoch_accuracy = None

        logger.info('Epoch {} '.format(e + 1))
        logger.info('\ttraining loss = {}'.format(np.round(epoch_loss,6)))
        if not args.no_val_set:
            logger.info('\tval loss = {}.'.format(np.round(val_loss,6)))
        logger.info('\ttest loss = {}.'.format(np.round(test_loss,6)))

        if args.classification:
            logger.info('\ttraining acc  = {}%'.format(np.round(epoch_accuracy * 100,6)))
            if not args.no_val_set:
                logger.info('\tval acc  = {}%'.format(np.round(val_accuracy * 100,6)))
            logger.info('\ttest acc  = {}%'.format(np.round(test_accuracy * 100,6)))

        if args.save_logs:
            utils.save_logs(writer, step=e + 1, net=net,
                            loss=epoch_loss,
                            epoch_time=2.,
                            accuracy=epoch_accuracy,
                            test_loss=test_loss,
                            test_accuracy=test_accuracy,
                            val_loss=val_loss,
                            val_accuracy=val_accuracy)

        epoch_losses = np.append(epoch_losses,
                                           epoch_loss)
        test_losses = np.append(test_losses,
                                          test_loss)
        if not args.no_val_set:
            val_losses = np.append(val_losses, val_loss)

        if args.classification:
            epoch_accuracies = np.append(
                epoch_accuracies,
                epoch_accuracy)
            test_accuracies = np.append(test_accuracies,
                                                  test_accuracy)
            if not args.no_val_set:
                val_accuracies = np.append(val_accuracies, val_accuracy)

        # if e > 4:
        #     if args.dataset in ['mnist', 'fashion_mnist']:
        #         if epoch_accuracy < 0.4:
        #             logger.info('writing error code -1')
        #             summary['finished'] = -1
        #             break
        #     if args.dataset in ['cifar10']:
        #         if epoch_accuracy < 0.25:
        #             logger.info('writing error code -1')
        #             summary['finished'] = -1
        #             break

        summary['loss_train_last'] = epoch_loss
        summary['loss_test_last'] = test_loss
        summary['loss_train_best'] = epoch_losses.min()
        summary['loss_test_best'] = test_losses.min()
        summary['loss_train'] = epoch_losses
        summary['loss_test'] = test_losses
        if not args.no_val_set:
            summary['loss_val_last'] = val_loss
            summary['loss_val_best'] = val_losses.min()
            summary['loss_val'] = val_losses
            best_epoch = val_losses.argmin()
            summary['epoch_best_loss'] = best_epoch
            summary['loss_test_val_best'] = \
                test_losses[best_epoch]
            summary['loss_train_val_best'] = \
                epoch_losses[best_epoch]

        if args.classification:
            summary['acc_train_last'] = epoch_accuracy
            summary['acc_test_last'] = test_accuracy
            summary[
                'acc_train_best'] = epoch_accuracies.max()
            summary[
                'acc_test_best'] = test_accuracies.max()
            summary['acc_train'] = epoch_accuracies
            summary['acc_test'] = test_accuracies
            if not args.no_val_set:
                summary['acc_val'] = val_accuracies
                summary['acc_val_last'] = val_accuracy
                summary['acc_val_best'] = val_accuracies.max()
                best_epoch = val_accuracies.argmax()
                summary['epoch_best_acc'] = best_epoch
                summary['acc_test_val_best'] = \
                    test_accuracies[best_epoch]
                summary['acc_train_val_best'] = \
                    epoch_accuracies[best_epoch]

        utils.save_summary_dict(args, summary)
        if args.save_summary_interval != -1 and e % args.save_summary_interval == 0:
            # Every 10 epochs, save a separate summary file that isn't overwritten.
            utils.save_summary_dict(args, summary, epoch=e)

    logger.info('Training network ... Done')
    return summary


def test_bp(args, device, net, test_loader, loss_function, epoch):
    loss = 0
    if args.classification:
        accuracy = 0
    nb_batches = 0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            if args.dataset not in ['mnist', 'fashion_mnist',
                                    'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'BPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()
            predictions = net(inputs)
            loss += loss_function(predictions, targets).item()
            if args.classification:
                accuracy += utils.accuracy(predictions, targets)

            if args.plot_autoencoder_images and i == 0:
                input_images = torch.reshape(inputs,
                                             (args.batch_size,1, 28, 28))[0:5]
                reconstructed_images = torch.reshape(predictions, (
                    args.batch_size,1, 28, 28))[0:5]
                save_image(input_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/input_epoch_{}.png'.format(
                        epoch)))
                save_image(reconstructed_images, os.path.join(
                    args.out_dir,
                    'autoencoder_images/reconstructed_epoch_{}.png'.format(
                        epoch)))

            nb_batches += 1

            if args.test and i == 1:
                break

    loss /= nb_batches
    if args.classification:
        accuracy /= nb_batches
    else:
        accuracy = None
    return accuracy, loss


def gn_damping_hpsearch(args, train_var, device, train_loader, net, writer, logger):
    freeze_forward_weights_copy = args.freeze_forward_weights
    args.freeze_forward_weights = True
    damping_values = np.logspace(-5., 1., num=7, base=10.0)
    damping_values = np.append(0, damping_values)
    average_angles = np.empty((len(damping_values), net.depth))

    for k, gn_damping in enumerate(damping_values):

        logger.info('testing damping={}'.format(gn_damping))
        angles_df = pd.DataFrame(columns=[i for i in range(0, net.depth)])
        step=0

        for i, (inputs, targets) in enumerate(train_loader):

            if args.dataset not in ['mnist', 'fashion_mnist',
                                    'mnist_autoencoder']:
                inputs, targets = inputs.to(device), targets.to(device)
            if not args.network_type == 'DDTPConv':
                inputs = inputs.flatten(1, -1)
            if args.autoencoder:
                targets = inputs.detach()

            predictions = net.forward(inputs)

            acc, loss, loss_lu = \
                train_forward_parameters(args, net, predictions, targets,
                                        train_var.loss_function,
                                        train_var.forward_optimizer)

            if  i % args.log_interval == 0:
                net.save_gnt_angles(writer, step, predictions,
                                    loss, gn_damping,
                                    retain_graph=False,
                                    custom_result_df=angles_df)
                step += 1

            if args.test and i == 1:
                break

        average_angles[k,:] = angles_df.mean(axis=0)

    optimal_damping_constants_layerwise = damping_values[average_angles.argmin(axis=0)]
    optimal_damping_constant = damping_values[average_angles.mean(axis=1).argmin(axis=0)]
    logger.info('average angles:')
    logger.info(average_angles)
    logger.info('optimal damping constant: {}'.format(optimal_damping_constant))
    logger.info('optimal damping constants layerwise: {}'.format(optimal_damping_constants_layerwise))

    file_path = os.path.join(args.out_dir, 'optimal_gnt_damping_constant.txt')
    with open(file_path, 'w') as f:
        f.write('average angles:\n')
        f.write(str(average_angles) + '\n')
        f.write('optimal damping constant: {} \n'.format(optimal_damping_constant))
        f.write('optimal damping constants layerwise: {} \n'.format(optimal_damping_constants_layerwise))

    args.freeze_forward_weights = freeze_forward_weights_copy

    return optimal_damping_constant