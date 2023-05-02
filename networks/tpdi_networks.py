from importlib.machinery import SourcelessFileLoader

from jsonschema import draft6_format_checker
from networks.tpdi_layers import TPDILayer, DFCLayer, GNLayer
from networks.abstract_network import AbstractNetwork
import torch
import torch.nn as nn
import numpy as np
from utils.utils import compute_batch_jacobian, dist
from utils import utils
import pandas as pd
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime
import os
# import torch.linalg

class DFCNetwork(AbstractNetwork):
    """
    Implements Dynamic Feedback Control,
    using network dynamics to invert targets for all hidden layers
    simultaneously.
    """

    def __init__(self, n_in, n_hidden, n_out, epochs, batch_size, activation='relu',
                 output_activation='linear', bias=True, sigma=0.36,
                 sigma_output=0.36, sigma_fb=0.01, sigma_output_fb=0.1,
                 forward_requires_grad=False, initialization='xavier_normal',
                 reset_K=False, initialization_K='weight_product', noise_K=1e-3,
                 save_df=False, target_stepsize=0.01,
                 ndi=False, alpha_di=0.001, dt_di=0.02, dt_di_fb=0.001,
                 tmax_di=500, tmax_di_fb=10, epsilon_di=0.3,
                 compare_with_ndi=False,
                 out_dir=r'.\logs',
                 learning_rule='nonlinear_difference',
                 use_initial_activations=False,
                 clip_grad_norm=-1,
                 inst_system_dynamics=False,
                 k_p=2.0, alpha_fb=0.5, noisy_dynamics=False,
                 fb_learning_rule='normal_controller',
                 inst_transmission=False, inst_transmission_fb=False,
                 time_constant_ratio=0.2, time_constant_ratio_fb=0.005,
                 apical_time_constant=-1, apical_time_constant_fb=None,
                 grad_deltav_cont=False,
                 k_p_fb=0.,
                 efficient_controller=False,
                 proactive_controller=False,
                 save_NDI_updates=False,
                 save_eigenvalues=False,
                 save_eigenvalues_bcn=False,
                 save_norm_r=False,
                 save_stdp_measures=False,
                 save_correlations=False,
                 save_epoch=None,
                 simulate_layerwise=False,
                 include_non_converged_samples=False,
                 low_pass_filter_u=False,
                 tau_f=0.9,
                 tau_noise=0.8,
                 decay_rate=0.8,
                 classification=False,
                 use_jacobian_as_fb=False,
                 stability_tricks=False,
                 freeze_fb_weights=False,
                 scaling_fb_updates=False,
                 at_steady_state=False,
                 average_ss=False,
                 not_low_pass_filter_r=False,
                 use_diff_hebbian_updates=False,
                 use_stdp_updates=False,
                 stdp_samples=1,):
        super().__init__(n_in=n_in,
                         n_hidden=n_hidden,
                         n_out=n_out,
                         activation=activation,
                         output_activation=output_activation,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         initialization=initialization,
                         save_df=save_df,                       
                         clip_grad_norm=clip_grad_norm)

        self._layers = self.set_layers(n_in, n_hidden, n_out,
                                       activation, output_activation,
                                       bias, forward_requires_grad,
                                       initialization)
        
        layer_output_dims = [l.weights.shape[0] for l in self.layers]
        self._r_target = [torch.zeros((int(tmax_di), batch_size, l)) for l in layer_output_dims]
        layer_output_dims_with_input = [n_in] + layer_output_dims[:-1]
        self._pre_activities = [torch.zeros((int(tmax_di), batch_size, l)) for l in layer_output_dims_with_input]
        self._delta_post_activities =[torch.zeros((int(tmax_di), batch_size, l)) for l in layer_output_dims]
        self._central_activities = [torch.zeros((int(tmax_di), batch_size, l)) for l in layer_output_dims]
        self._ff_activities = [torch.zeros((batch_size, l)) for l in layer_output_dims]
        self._weights = [torch.zeros_like(l.weights.data) for l in self.layers] 
        self._weights_updates = [torch.zeros_like(l.weights.data) for l in self.layers]
        self._weights_updates_stdp = [torch.zeros_like(l.weights.data) for l in self.layers]
        self._weights_updates_diff_hebbian = [torch.zeros_like(l.weights.data) for l in self.layers]
        self._weights_updates_dfc = [torch.zeros_like(l.weights.data) for l in self.layers]
        self._weights_updates_bp = [torch.zeros_like(l.weights.data) for l in self.layers]
        self._error_time_to_convergence = torch.zeros((epochs, int(tmax_di)-1))
        self._u = torch.zeros(batch_size, n_out)
        self._correlations_dfc = 0
        self._correlations_dfc_diff_hebbian = 0
        self._correlations_bp = 0
        self._dfc_angles_list = np.zeros(len(self.layers))
        self._dfc_total_angle = 0
        self._dfc_diff_hebbian_angles_list = np.zeros(len(self.layers))
        self._dfc_diff_hebbian_total_angle = 0
        self._step = 0
        self._stdp_samples = stdp_samples

        self._n_in = n_in
        self._n_hidden = n_hidden
        self._n_out = n_out
        self._target_stepsize = target_stepsize
        self._activation = activation
        self._num_epochs = epochs
        self._batch_size = batch_size
        self._sigma = sigma
        self._sigma_output = sigma_output
        self._sigma_fb = sigma_fb
        self._sigma_output_fb = sigma_output_fb
        self._reset_K = reset_K
        self.initialization_K = initialization_K
        self._noise_K = noise_K
        self._save_df = save_df
        self._ndi = ndi
        self._alpha_di = alpha_di
        self._dt_di = dt_di
        self._dt_di_fb = dt_di_fb
        self._tmax_di = tmax_di
        self._tmax_di_fb = tmax_di_fb
        self._epsilon_di = epsilon_di
        self.compare_with_ndi = compare_with_ndi
        self.makeplots = False
        self._out_dir = out_dir 
        self.learning_rule = learning_rule
        self.use_initial_activations = use_initial_activations
        self._clip_grad_norm = clip_grad_norm
        self._inst_system_dynamics = inst_system_dynamics
        self._k_p = k_p
        self._k_p_fb = k_p_fb
        self._alpha_fb = alpha_fb
        self._noisy_dynamics = noisy_dynamics
        self._fb_learning_rule = fb_learning_rule
        self._inst_transmission = inst_transmission
        self._inst_transmission_fb = inst_transmission_fb
        self._time_constant_ratio = time_constant_ratio
        self._time_constant_ratio_fb = time_constant_ratio_fb
        self._apical_time_constant = apical_time_constant
        self._apical_time_constant_fb = apical_time_constant_fb
        self._grad_deltav_cont = grad_deltav_cont
        self._efficient_controller = efficient_controller
        self._proactive_controller = proactive_controller
        self._save_NDI_updates = save_NDI_updates
        self._save_eigenvalues = save_eigenvalues
        self._save_eigenvalues_bcn = save_eigenvalues_bcn
        self._save_norm_r = save_norm_r
        self._save_stdp_measures = save_stdp_measures
        self._save_correlations = save_correlations
        self._save_epoch = save_epoch
        self._include_non_converged_samples = include_non_converged_samples
        self._low_pass_filter_u = low_pass_filter_u
        self._tau_f = tau_f
        self._alpha_r = dt_di / tau_f
        self._alpha_u = dt_di / tau_f
        self._decay_rate = decay_rate
        self._alpha_noise = dt_di / tau_noise
        self._tau_noise = tau_noise
        self._classification = classification
        self._loss_function_name = 'mse'
        self._use_jacobian_as_fb = use_jacobian_as_fb
        self._stability_tricks = stability_tricks
        self._freeze_fb_weights = freeze_fb_weights
        self._scaling_fb_updates = scaling_fb_updates
        self._at_steady_state = at_steady_state
        self._average_ss = average_ss
        self._not_low_pass_filter_r = not_low_pass_filter_r
        self._use_diff_hebbian_updates = use_diff_hebbian_updates
        self._use_stdp_updates = use_stdp_updates
        
        if simulate_layerwise:
            self._simulation_mode = 'layerwise'
        else:
            self._simulation_mode = 'blockwise'
        
        if ndi:
            print('Non-dynamical inversion is active.')
        else:
            print('Full dynamical inversion is active.')

            self.converged_samples_per_epoch = 0
            self.diverged_samples_per_epoch = 0
            self.not_converged_samples_per_epoch = 0
            self.last_savefig_time = 0 

        if grad_deltav_cont:
            print('Delta_v gradients are computed continuously.')
        
        if compare_with_ndi:
            print('Also computing analytical solution (NDI) for comparison (this causes a computational overhead).')
        
        self.feedbackweights_initialization()
        
        if save_df:
            self.condition_gn = pd.DataFrame(
                columns=[0])
            self.condition_gn_init = pd.DataFrame(columns=[0])

    @property
    def r_target(self):
        """Getter for read-only attribute :attr:`r_target`."""
        return self._r_target

    @r_target.setter
    def r_target(self, value):
        """Setter for attribute :attr:`r_target`."""
        self._r_target = value
    
    @property
    def pre_activities(self):
        """Getter for read-only attribute :attr:`pre_activities`."""
        return self._pre_activities

    @pre_activities.setter
    def pre_activities(self, value):
        """Setter for attribute :attr:`pre_activities`."""
        self._pre_activities = value

    @property
    def delta_post_activities(self):
        """Getter for read-only attribute :attr:`delta_post_activities`."""
        return self._delta_post_activities

    @delta_post_activities.setter
    def delta_post_activities(self, value):
        """Setter for attribute :attr:`delta_post_activities`."""
        self._delta_post_activities = value

    @property
    def weights(self):
        """Getter for read-only attribute :attr:`weights`."""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Setter for attribute :attr:`weights`."""
        self._weights = value
    
    @property
    def central_activities(self):
        """Getter for read-only attribute :attr:`central_activities`."""
        return self._central_activities

    @central_activities.setter
    def central_activities(self, value):
        """Setter for attribute :attr:`central_activities`."""
        self._central_activities = value
    
    @property
    def ff_activities(self):
        """Getter for read-only attribute :attr:`ff_activities`."""
        return self._ff_activities

    @ff_activities.setter
    def ff_activities(self, value):
        """Setter for attribute :attr:`ff_activities`."""
        self._ff_activities = value

    @property
    def weights_updates(self):
        """Getter for read-only attribute :attr:`weights_updates`."""
        return self._weights_updates

    @weights_updates.setter
    def weights_updates(self, value):
        """Setter for attribute :attr:`weights_updates`."""
        self._weights_updates = value

    @property
    def weights_updates_stdp(self):
        """Getter for read-only attribute :attr:`weights_updates_stdp`."""
        return self._weights_updates_stdp

    @weights_updates_stdp.setter
    def weights_updates_stdp(self, value):
        """Setter for attribute :attr:`weights_updates_stdp`."""
        self._weights_updates_stdp = value

    @property
    def weights_updates_diff_hebbian(self):
        """Getter for read-only attribute :attr:`weights_updates_diff_hebbian`."""
        return self._weights_updates_diff_hebbian

    @weights_updates_diff_hebbian.setter
    def weights_updates(self, value):
        """Setter for attribute :attr:`weights_updates_diff_hebbian`."""
        self._weights_updates_diff_hebbian = value
    
    @property
    def weights_updates_dfc(self):
        """Getter for read-only attribute :attr:`weights_updates_dfc`."""
        return self._weights_updates_dfc

    @weights_updates_dfc.setter
    def weights_updates_dfc(self, value):
        """Setter for attribute :attr:`weights_updates_dfc`."""
        self._weights_updates_dfc = value
    
    @property
    def weights_updates_bp(self):
        """Getter for read-only attribute :attr:`weights_updates_bp`."""
        return self._weights_updates_bp

    @weights_updates_bp.setter
    def weights_updates_bp(self, value):
        """Setter for attribute :attr:`weights_updates_bp`."""
        self._weights_updates_bp = value

    @property
    def error_time_to_convergence(self):
        """Getter for read-only attribute :attr:`error_time_to_convergence`."""
        return self._error_time_to_convergence

    @error_time_to_convergence.setter
    def error_time_to_convergence(self, value):
        """Setter for attribute :attr:`error_time_to_convergence`."""
        self._error_time_to_convergence = value
    
    @property
    def correlations_bp(self):
        """Getter for read-only attribute :attr:`correlations_bp`."""
        return self._correlations_bp
        
    @correlations_bp.setter
    def correlations_bp(self, value):
        """Setter for attribute :attr:`correlations_bp`."""
        self._correlations_bp = value
    
    @property
    def correlations_dfc(self):
        """Getter for read-only attribute :attr:`correlations_dfc`."""
        return self._correlations_dfc
        
    @correlations_dfc.setter
    def correlations_dfc(self, value):
        """Setter for attribute :attr:`correlations_dfc`."""
        self._correlations_dfc = value
    
    @property
    def correlations_dfc_diff_hebbian(self):
        """Getter for read-only attribute :attr:`correlations_dfc_diff_hebbian`."""
        return self._correlations_dfc_diff_hebbian
        
    @correlations_dfc_diff_hebbian.setter
    def correlations_dfc_diff_hebbian(self, value):
        """Setter for attribute :attr:`correlations_dfc_diff_hebbian`."""
        self._correlations_dfc_diff_hebbian = value
    
    @property
    def dfc_angles_list(self):
        """Getter for read-only attribute :attr:`dfc_angles_list`."""
        return self._dfc_angles_list
        
    @dfc_angles_list.setter
    def dfc_angles_list(self, value):
        """Setter for attribute :attr:`dfc_angles_list`."""
        self._dfc_angles_list = value
    
    @property
    def dfc_total_angle(self):
        """Getter for read-only attribute :attr:`dfc_total_angle`."""
        return self._dfc_total_angle
        
    @dfc_total_angle.setter
    def dfc_total_angle(self, value):
        """Setter for attribute :attr:`dfc_total_angle`."""
        self._dfc_total_angle = value

    @property
    def dfc_diff_hebbian_angles_list(self):
        """Getter for read-only attribute :attr:`dfc_diff_hebbian_angles_list`."""
        return self._dfc_diff_hebbian_angles_list
        
    @dfc_diff_hebbian_angles_list.setter
    def dfc_diff_hebbian_angles_list(self, value):
        """Setter for attribute :attr:`dfc_diff_hebbian_angles_list`."""
        self._dfc_diff_hebbian_angles_list = value
    
    @property
    def dfc_diff_hebbian_total_angle(self):
        """Getter for read-only attribute :attr:`dfc_diff_hebbian_total_angle`."""
        return self._dfc_diff_hebbian_total_angle
        
    @dfc_diff_hebbian_total_angle.setter
    def dfc_diff_hebbian_total_angle(self, value):
        """Setter for attribute :attr:`dfc_diff_hebbian_total_angle`."""
        self._dfc_diff_hebbian_total_angle = value

    @property
    def step(self):
        """Getter for read-only attribute :attr:`step`."""
        return self._step
        
    @step.setter
    def step(self, value):
        """Setter for attribute :attr:`step`."""
        self._step = value
    
    @property
    def stdp_samples(self):
        """ Getter for read-only attribute :attr:`stdp_samples`"""
        return self._stdp_samples

    @property
    def n_in(self):
        """Getter for read-only attribute :attr:`n_in`."""
        return self._n_in
    
    @property
    def n_hidden(self):
        """Getter for read-only attribute :attr:`n_hidden`."""
        return self._n_hidden

    @property
    def n_out(self):
        """Getter for read-only attribute :attr:`n_out`."""
        return self._n_out
    
    @property
    def target_stepsize(self):
        """Getter for read-only attribute :attr:`target_stepsize`."""
        return self._target_stepsize

    @property
    def activation(self):
        """Getter for read-only attribute :attr:`activation`."""
        return self._activation
    
    @property
    def num_epochs(self):
        """Getter for read-only attribute :attr:`num_epochs`."""
        return self._num_epochs

    @property
    def batch_size(self):
        """Getter for read-only attribute :attr:`batch_size`."""
        return self._batch_size
    
    @property
    def sigma(self):
        """Getter for read-only attribute :attr:`sigma`."""
        return self._sigma

    @property
    def sigma_output(self):
        """Getter for read-only attribute :attr:`sigma_output`."""
        return self._sigma_output

    @property
    def sigma_fb(self):
        """Getter for read-only attribute :attr:`sigma`."""
        return self._sigma_fb

    @property
    def sigma_output_fb(self):
        """Getter for read-only attribute :attr:`sigma_output_fb`."""
        return self._sigma_output_fb

    @property
    def ndi(self):
        """Getter for read-only attribute :attr:`ndi`."""
        return self._ndi

    @property
    def alpha_di(self):
        """Getter for read-only attribute :attr:`alpha_di`."""
        return self._alpha_di

    @property
    def dt_di(self):
        """Getter for read-only attribute :attr:`dt_di`."""
        return self._dt_di

    @property
    def dt_di_fb(self):
        """Getter for read-only attribute :attr:`dt_di_fb`."""
        return self._dt_di_fb

    @property
    def tmax_di(self):
        """Getter for read-only attribute :attr:`tmax_di`."""
        return self._tmax_di

    @property
    def tmax_di_fb(self):
        """Getter for read-only attribute :attr:`tmax_di_fb`."""
        return self._tmax_di_fb

    @property
    def epsilon_di(self):
        """Getter for read-only attribute :attr:`epsilon_di`."""
        return self._epsilon_di

    @property
    def noise_K(self):
        """Getter for read-only attribute :attr:`epsilon_di`."""
        return self._noise_K

    @property
    def out_dir(self):
        """Getter for read-only attribute :attr:`out_dir`."""
        return self._out_dir

    @property
    def k_p(self):
        """Getter for read-only attribute :attr:`k_p`"""
        return self._k_p

    @property
    def k_p_fb(self):
        """Getter for read-only attribute :attr:`k_p_fb`"""
        return self._k_p_fb

    @property
    def inst_system_dynamics(self):
        """Getter for read-only attribute :attr:`inst_system_dynamics`"""
        return self._inst_system_dynamics

    @property
    def alpha_fb(self):
        """Getter for read-only attribute :attr:`alpha_fb`"""
        return self._alpha_fb

    @property
    def fb_learning_rule(self):
        """Getter for read-only attribute :attr:`fb_learning_rule`"""
        return self._fb_learning_rule

    @property
    def noisy_dynamics(self):
        """Getter for read-only attribute :attr:`noisy_dynamics`"""
        return self._noisy_dynamics

    @property
    def inst_transmission(self):
        """ Getter for read-only attribute :attr:`inst_transmission`"""
        return self._inst_transmission

    @property
    def inst_transmission_fb(self):
        """ Getter for read-only attribute :attr:`inst_transmission_fb`"""
        return self._inst_transmission_fb

    @property
    def time_constant_ratio(self):
        """ Getter for read-only attribute :attr:`time_constant_ratio`"""
        return self._time_constant_ratio

    @property
    def time_constant_ratio_fb(self):
        """ Getter for read-only attribute :attr:`time_constant_ratio_fb`"""
        return self._time_constant_ratio_fb

    @property
    def apical_time_constant(self):
        """ Getter for read-only attribute :attr:`apical_time_constant`"""
        return self._apical_time_constant

    @property
    def apical_time_constant_fb(self):
        """ Getter for read-only attribute :attr:`apical_time_constant_fb`"""
        return self._apical_time_constant_fb

    @property
    def full_Q(self):
        """ Getter for matrix :math:`\bar{Q}` containing the concatenated
        feedback weights."""
        return torch.cat([l.feedbackweights for l in self.layers], dim=0)

    @property
    def grad_deltav_cont(self):
        """ Getter for read-only attribute :attr:`grad_deltav_cont`"""
        return self._grad_deltav_cont

    @property
    def efficient_controller(self):
        """ Getter for read-only attribute :attr:`efficient_controller`"""
        return self._efficient_controller

    @property
    def proactive_controller(self):
        """ Getter for read-only attribute :attr:`proactive_controller`"""
        return self._proactive_controller

    @property
    def save_NDI_updates(self):
        """ Getter for read-only attribute :attr:`save_NDI_updates`"""
        return self._save_NDI_updates

    @save_NDI_updates.setter
    def save_NDI_updates(self, bool_value):
        """ Setter for attribute save_NDI_updates."""
        self._save_NDI_updates = bool_value

    @property
    def save_eigenvalues(self):
        """ Getter for read-only attribute :attr:`save_eigenvalues`"""
        return self._save_eigenvalues

    @save_eigenvalues.setter
    def save_eigenvalues(self, bool_value):
        """ Setter for attribute save_eigenvalues."""
        self._save_eigenvalues = bool_value

    @property
    def save_eigenvalues_bcn(self):
        """ Getter for read-only attribute :attr:`save_eigenvalues_bcn`"""
        return self._save_eigenvalues_bcn

    @save_eigenvalues_bcn.setter
    def save_eigenvalues_bcn(self, bool_value):
        """ Setter for attribute save_eigenvalues_bcn."""
        self._save_eigenvalues_bcn = bool_value

    @property
    def save_norm_r(self):
        """ Getter for read-only attribute :attr:`save_norm_r`"""
        return self._save_norm_r

    @property
    def save_stdp_measures(self):
        """ Getter for read-only attribute :attr:`save_stdp_measures`"""
        return self._save_stdp_measures
    
    @property
    def save_correlations(self):
        """ Getter for read-only attribute :attr:`save_correlations`"""
        return self._save_correlations

    @property
    def save_epoch(self):
        """ Getter for read-only attribute :attr:`save_epoch`"""
        return self._save_epoch
    
    @property
    def use_diff_hebbian_updates(self):
        """ Getter for read-only attribute :attr:`use_diff_hebbian_updates`"""
        return self._use_diff_hebbian_updates

    @property
    def use_stdp_updates(self):
        """ Getter for read-only attribute :attr:`use_stdp_updates`"""
        return self._use_stdp_updates

    @property
    def include_non_converged_samples(self):
        """ Getter for read-only attribute :attr:`include_non_converged_samples`"""
        return self._include_non_converged_samples

    @property
    def low_pass_filter_u(self):
        """ Getter for read-only attribute :attr:`low_pass_filter_u`"""
        return self._low_pass_filter_u

    @property
    def alpha_r(self):
        """ Getter for read-only attribute :attr:`alpha_r`"""
        return self._alpha_r
    
    @property
    def tau_f(self):
        """ Getter for read-only attribute :attr:`tau_f`"""
        return self._tau_f
    
    @property
    def decay_rate(self):
        """ Getter for read-only attribute :attr:`decay_rate`"""
        return self._decay_rate

    @property
    def alpha_u(self):
        """ Getter for read-only attribute :attr:`alpha_u`"""
        return self._alpha_u

    @property
    def alpha_noise(self):
        """ Getter for read-only attribute :attr:`alpha_noise`"""
        return self._alpha_noise
    
    @property
    def tau_noise(self):
        """ Getter for read-only attribute :attr:`tau_noise`"""
        return self._tau_noise
    
    @property
    def classification(self):
        """ Getter for read-only attribute :attr:`classification`"""
        return self._classification

    @save_norm_r.setter
    def save_norm_r(self, bool_value):
        """ Setter for attribute save_norm_r."""
        self._save_norm_r = bool_value

    @save_stdp_measures.setter
    def save_stdp_measures(self, bool_value):
        """ Setter for attribute save_stdp_measures."""
        self._save_stdp_measures = bool_value
    
    @save_correlations.setter
    def save_correlations(self, bool_value):
        """ Setter for attribute save_correlations."""
        self._save_correlations = bool_value
    
    @save_epoch.setter
    def save_epoch(self, value):
        """ Setter for attribute save_epoch."""
        self._save_epoch = value

    @property
    def simulation_mode(self):
        """ Getter for read-only attribute :attr:`simulation_mode`"""
        return self._simulation_mode

    @property
    def loss_function_name(self):
        """ Getter for read-only attribute :attr:`loss_function_name`"""
        return self._loss_function_name

    @loss_function_name.setter
    def loss_function_name(self, value):
        """Setter for loss_function_name"""
        self._loss_function_name = value

    @property
    def use_jacobian_as_fb(self):
        """ Getter for read-only attribute :attr:`use_jacobian_as_fb`"""
        return self._use_jacobian_as_fb

    @property
    def stability_tricks(self):
        """ Getter for read-only attribute :attr:`stability_tricks`"""
        return self._stability_tricks

    @property
    def freeze_fb_weights(self):
        """ Getter for read-only attribute :attr:`freeze_fb_weights`"""
        return self._freeze_fb_weights
    
    @property
    def scaling_fb_updates(self):
        """ Getter for read-only attribute :attr:`scaling_fb_updates`"""
        return self._scaling_fb_updates

    @property
    def at_steady_state(self):
        """ Getter for read-only attribute :attr:`at_steady_state`"""
        return self._at_steady_state

    @property
    def average_ss(self):
        """ Getter for read-only attribute :attr:`average_ss`"""
        return self._average_ss

    @property
    def not_low_pass_filter_r(self):
        """ Getter for read-only attribute :attr:`not_low_pass_filter_r`"""
        return self._not_low_pass_filter_r
    
    @property
    def u(self):
        """Getter for read-only attribute :attr:`u`"""
        return self._u

    @u.setter
    def u(self, value):
        """Setter for u"""
        self._u = value

    @property
    def state_dict(self):
        """A dictionary containing the current state of the network,
        incliding forward and backward weights."""
        forward_weights = [layer.weights.data for layer in self._layers]
        feedback_weights = [layer.feedbackweights.data for layer in self._layers]

        state_dict = {'forward_weights': forward_weights,
                      'feedback_weights': feedback_weights}

        return state_dict


    def load_state_dict(self, state_dict):
        """Load a state into the network.

        This function sets the forward and backward weights.

        Args:
            state_dict (dict): The state with forward and backward weights.
        """
        for l, layer in enumerate(self._layers):
            layer.weights.data = state_dict['forward_weights'][l]
            layer.feedbackweights.data = state_dict['feedback_weights'][l]


    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization):
        
        n_all = [n_in] + n_hidden + [n_out]

        if isinstance(activation, str):
            activation = [activation]*len(n_hidden)


        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):

            layers.append(DFCLayer(n_all[i - 1], n_all[i], n_out,
                                   bias=bias,
                                   forward_requires_grad=forward_requires_grad,
                                   forward_activation=activation[i-1],
                                   initialization=initialization,
                                   clip_grad_norm=self.clip_grad_norm))
        layers.append(DFCLayer(n_all[-2], n_all[-1], n_out,
                               bias=bias, forward_requires_grad=forward_requires_grad,
                               forward_activation=output_activation,
                               initialization=initialization,
                               clip_grad_norm=self.clip_grad_norm))

        return layers


    def feedbackweights_initialization(self):
        """
        Initializes the inversion matrix K of each layer to the product of
        the forward weights (transposed) of subsequent layers.
        """

        print('Calculating the inversion matrices K_i.')

        if self.initialization_K == 'weight_product':
            for i in range(self.depth - 1):
                K = 1.*torch.transpose(self._layers[i+1].weights.data, 0, 1)
                for j in range(i + 2, self.depth):
                    K = torch.mm(K, torch.transpose(self._layers[j].weights.data, 0, 1))

                if self.noise_K > 0.:
                    K += torch.normal(mean=0., std=self.noise_K, size=K.shape)

                self.layers[i].feedbackweights = K
            self.layers[-1].feedbackweights = \
                torch.eye(self.layers[-1].feedbackweights.shape[0])
        else:
            for l in self.layers[:-1]:
                if self.initialization_K == "orthogonal":
                    gain = np.sqrt(6. / (l.feedbackweights.shape[0] + l.feedbackweights.shape[1]))
                    nn.init.orthogonal_(l.feedbackweights, gain=gain)
                elif self.initialization_K == 'xavier':
                    nn.init.xavier_uniform_(l.feedbackweights)
                elif self.initialization_K == 'xavier_normal':
                    nn.init.xavier_normal_(l.feedbackweights)
                else:
                    raise ValueError('Provided K initialization "{}" is not '
                                     'supported.'.format(self.initialization_K))
        
        self.layers[-1].feedbackweights = \
                torch.eye(self.layers[-1].feedbackweights.shape[0])
    

    def backward(self, loss, targets, target_lr, save_target=True, writer=None,
                 step=None):
        """
        Run the feedback phase of the network, where the network is pushed
        to the output target by the controller. Compute the update of
        the forward weights of the network accordingly and save it in
        ``self.layers[i].weights.grad`` for each layer i.
        Args:
            loss (torch.Tensor): The output loss of the feedforward pass.
            target_lr (float): The stepsize that is used to compute the
                output target
            save_target (bool): Flag indicating whether the equilibrium
                activations of the network should be saved.
            writer: The tensorboard writer. Placeholders for classes that
                inherit from here.
            step: The training step, for tensorboard.
        """
        """
        Note: backward is now implemented network-level, as it perform 
            simultaneous inversion and control of all targets
        """

        if self.classification and self.target_stepsize==0.5:
            output_target = targets
        else:
            output_target = self.compute_output_target(loss, target_lr)

        v_feedforward = [l.linearactivations for l in self.layers]
        r_feedforward = [l.activations for l in self.layers]

        if self.ndi:
            u, v_targets, r_targets, target_lastlayer, delta_v = \
                self.non_dynamical_inversion(output_target, self.alpha_di)
        else:
            u, v_targets, r_targets, target_lastlayer, delta_v, (vs_time, vb_time, r_targets_time) = \
                self.dynamical_inversion(output_target, alpha=self.alpha_di,
                                         dt=self.dt_di,
                                         tmax=self.tmax_di,
                                         epsilon=self.epsilon_di,
                                         makeplots=self.makeplots,
                                         savedir=self.out_dir,
                                         compare_with_ndi=self.compare_with_ndi)
        if self.save_eigenvalues:
            A, max_eig, keys = self.compute_A_matrices(v_targets, linear=True)
            self.max_eig = max_eig

        if self.save_norm_r:
            self.compute_norm_r(r_feedforward)

        if self.compare_with_ndi:
            u_ndi, v_targets_ndi, r_targets_ndi, target_ndi, delta_v_ndi = \
                self.non_dynamical_inversion(output_target, self.alpha_di)
            self.rel_dist_to_NDI.append(torch.mean(dist(target_lastlayer, target_ndi)/dist(r_feedforward[-1], target_ndi)).detach().cpu().numpy())

        if self.save_correlations:
            # compute internally the dfc angles
            dfc_updates_list =[torch.zeros_like(self.layers[i].get_forward_gradients()[0]) for i in range(self.depth)]
            current_updates_list = [torch.zeros_like(self.layers[i].get_forward_gradients()[0]) for i in range(self.depth)]
            dfc_diff_hebbian_updates_list = [torch.zeros_like(self.layers[i].get_forward_gradients()[0]) for i in range(self.depth)]

        for i in range(self.depth):
            delta_v_i = delta_v[i]

            if i == 0:
                r_previous = self.input
            else:
                if self.use_initial_activations:
                    r_previous = r_feedforward[i - 1]
                else:
                    r_previous = r_targets[i - 1]
            
            if self.noisy_dynamics and not self.not_low_pass_filter_r:
                #compute lowpass filter of r (average out the injected noise in ff phase)
                r_targets_time_filtered_i = torch.zeros_like(r_targets_time[i])
                r_targets_time_filtered_i[0] = r_targets_time[i][0].clone()
                for t in range(1,int(self.tmax_di)):
                    #exponential smoothing
                    r_targets_time_filtered_i[t] = self.alpha_r*r_targets_time[i][t] + (1-self.alpha_r)*r_targets_time_filtered_i[t-1]
                r_targets_time[i] = r_targets_time_filtered_i

            if self.grad_deltav_cont:
                if self.efficient_controller:
                    raise ValueError('Argument "efficient_controller" must be False for dfc method.')
                    pass 
                else:
                    # if i == 0:
                    #     r_previous_time = self.input.unsqueeze(0).expand(
                    #         int(self.tmax_di), self.input.shape[0], self.input.shape[1])

                    # else:
                    #     r_previous_time = r_targets_time[i - 1]
                    
                    central_activities_i = self.layers[i].forward_activationfunction(vs_time[i])
                    # test without filtering of pre activities (targets)
                    if i == 0:
                        r_previous_time = self.input.unsqueeze(0).expand(
                            int(self.tmax_di), self.input.shape[0], self.input.shape[1])
                    else:
                        r_previous_time = self.layers[i-1].forward_activationfunction(vs_time[i-1])

                    weights_updates = self.layers[i].compute_forward_gradients_deltav_continuous(
                        vs_time[i], vb_time[i], central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
                        
            else:
                weights_updates = self.layers[i].compute_forward_gradients(delta_v_i, r_previous, learning_rule=self.learning_rule)
                        
            self.layers[i].target = v_targets[i]
            self.layers[i].linearactivations_ss = v_targets[i]
            self.layers[i].activations_ss = r_targets[i]
            self.u = u

            if self.save_NDI_updates:
                u_ndi, v_targets_ndi, r_targets_ndi, target_ndi, delta_v_ndi = \
                    self.non_dynamical_inversion(output_target, self.alpha_di)
                delta_v_i_ndi = delta_v_ndi[i]
                self.layers[i].compute_forward_gradients(delta_v_i_ndi, r_previous,
                                                                learning_rule=self.learning_rule,
                                                                saving_ndi_updates=True)

            if self.grad_deltav_cont and self.save_stdp_measures and (self.epoch==self.save_epoch-1 or (self.save_epoch==None and self.epoch==self.num_epochs-1)):
                # save the feedforward activities (nonlinear activations) 
                self.pre_activities[i] = r_previous_time
                # save difference between central and basal compartments (linear activations)
                self.delta_post_activities[i] = vs_time[i] - vb_time[i]
                # save weights values 
                self.weights[i] = self.layers[i].weights.detach().cpu()
                # save stdp weights updates values 
                self.weights_updates[i] = weights_updates.detach().cpu()
                
                # save central compartment activties (after nonlinearity is applied)
                self.central_activities[i] = self.layers[i].forward_activationfunction(vs_time[i])
                # save feedforward activities (after nonlinearity is applied)
                # y = self.forward(self.input, dummy_reset=True)
                # self.ff_activities[i] = self.layers[i].activations
            
            # compute layerwise angles for dfc updates
            out_dfc = self.compute_dfc_angles(i, vs_time[i], vb_time[i], central_activities_i, r_previous_time)
            dfc_stdp_gradients = out_dfc[0]
            dfc_gradients = out_dfc[1]
            dfc_diff_hebbian_gradients = out_dfc[2]
            dfc_angles = out_dfc[3]
            dfc_diff_hebbian_angles = out_dfc[4]

            if self.save_stdp_measures and self.epoch==self.num_epochs-1:
                self.weights_updates_bp[i] = self.layers[i].compute_bp_update(loss, retain_graph=True)[0]
                self.weights_updates_stdp[i] = dfc_stdp_gradients.detach()
                self.weights_updates_diff_hebbian[i] = dfc_diff_hebbian_gradients.detach()
                self.weights_updates_dfc[i] = dfc_gradients.detach()
            
            if self.save_correlations:
                current_updates_list[i] = dfc_stdp_gradients.detach()
                dfc_updates_list[i] = dfc_gradients.detach()
                dfc_diff_hebbian_updates_list[i] = dfc_diff_hebbian_gradients.detach()
                self.dfc_angles_list[i] = dfc_angles
                self.dfc_diff_hebbian_angles_list[i] = dfc_diff_hebbian_angles
        
        if self.save_correlations:
            # compute total angles for dfc udpates
            current_updates_concat = utils.vectorize_tensor_list(current_updates_list)
            dfc_updates_concat = utils.vectorize_tensor_list(dfc_updates_list)
            dfc_total_angle = utils.compute_angle(current_updates_concat, dfc_updates_concat)
            self.dfc_total_angle = dfc_total_angle
            # compute total angles for dfc differential hebbian angles
            dfc_diff_hebbian_updates_concat = utils.vectorize_tensor_list(dfc_diff_hebbian_updates_list)
            dfc_diff_hebbian_total_angle = utils.compute_angle(current_updates_concat, dfc_diff_hebbian_updates_concat)
            self.dfc_diff_hebbian_total_angle = dfc_diff_hebbian_total_angle

        if self.grad_deltav_cont and self.save_correlations and self.epoch==self.num_epochs-1:
            # save bp agles
            self.correlations_bp = self.bp_angles_network.at[self.step, 0]
            # save dfc agles
            self.correlations_dfc = self.dfc_angles_network.at[self.step, 0]
            # save dfc_diff_hebbian agles
            self.correlations_dfc_diff_hebbian = self.dfc_diff_hebbian_angles_network.at[self.step, 0]

        self.makeplots = False 


    def compute_norm_r(self, r_feedforward):
        r"""
        All these operations are performed PER SAMPLE.
        Batch averaging only occurs in the very end, when saving to Tensorboard.
        to get the mean, initialize to the first layer,
        then add the others (0 to L-1) and then divide
        the metric "deviation of r" (dev_r) has also been introduced, which averages
        the previous across all layers (but still per sample):
        $$ dev_r = sqrt[\suxm_{i=0}^{L-1}(||r_i||_2 - mean(||r||_2))^2 / L ] /mean(||r||_2) $$
        """

        self.norm_r = dict()
        mean = torch.norm(self.input, dim=1)
        for i in range(self.depth - 1):
            mean += torch.norm(r_feedforward[i], dim=1)
        mean = (mean / self.depth).detach()
        self.norm_r["mean"] = mean.cpu().numpy()
        input_deviation = torch.norm(self.input.detach(), dim=1) - mean
        input_rel_dev = input_deviation / mean
        self.norm_r["input"] = input_rel_dev.cpu().numpy()
        var_norm_r = input_deviation ** 2
        for i in range(self.depth - 1):
            layer_deviation = torch.norm(r_feedforward[i].detach(), dim=1) - mean
            layer_rel_dev = layer_deviation / mean
            self.norm_r["layer " + str(i + 1)] = layer_rel_dev.cpu().numpy()
            var_norm_r += layer_deviation ** 2
        self.dev_r = np.sqrt(((1 / self.depth) * var_norm_r / mean).detach().cpu().numpy())

    @torch.no_grad()
    def non_dynamical_inversion(self, output_target, alpha, retain_graph=False):
        r"""
        Compute the analytical solution for the network activations in the
        feedback phase, when the controller pushes the network to reach the
        output target. The following formulas are used:

        ..math::
            u &= (\bar{J}\bar{Q} + \alpha I)^{-1} \delta_L \\
            \Delta \bar{v} = \bar{Q} u

        with :math:`u` the control input at steady-state, :math:`\Delta v` the
        apical compartment voltage at steady-state, :math:`\delta_L` the
        difference between the output target and the output of the network at
        the feedforward sweep (without feedback). For the other symbols, see the
        paper.
        Args:
            output_target (torch.Tensor): The output target of the network
            alpha (float): the leakage term of the controller
            retain_graph (bool): Flag indicating whether the autograd graph
                should be retained for later use.
        Returns (tuple): An ordered tuple containing
            u_ndi (torch.Tensor): :math:`u`, the steady state control input
            v_target (list): :math:`\bar{v}_{ss} = \bar{v}^- + \Delta_\bar{v}`
                The steady-state voltage activations of the somatic compartments,
                split in a list that contains v_{ss} for each layer
            r_target (list): :math:`\bar{r}_{ss} = \phi(\bar{v}_{ss}`
                The steady-state firing rates of the neurons,
                split in a list that contains r_{ss} for each layer
            target_ndi (torch.Tensor): The steady state output activation
                of the network.
            delta_v_ndi_split (list): a list containing :math:`\Delta v_i` for
                each layer
        """
        # Compute the Jacobian.
        Q = self.full_Q
        # J = self.compute_full_jacobian(linear=True, retain_graph=retain_graph)
        J = self.compute_full_jacobian(linear=True, steady_state=self.at_steady_state, retain_graph=retain_graph, r_targets=self.r_target, average_ss=self.average_ss)

        # Compute the error.
        deltaL = self.compute_error(output_target, self.layers[-1].activations)

        # Analytically compute the steady-state control signal.
        device = output_target.device
        if self.use_jacobian_as_fb:
            u_ndi = torch.solve(deltaL.unsqueeze(2), \
                        torch.matmul(J, J.transpose(1, 2)) + \
                        alpha * torch.eye(J.shape[1]).to(device))\
                        [0].squeeze(-1)
            delta_v_ndi = torch.matmul(J.transpose(1, 2), \
                                u_ndi.unsqueeze(2)).squeeze(-1)
        else:
            u_ndi = torch.solve(deltaL.unsqueeze(2), torch.matmul(J, Q) + \
                        alpha * torch.eye(J.shape[1]).to(device))\
                        [0].squeeze(-1)
            delta_v_ndi = torch.matmul(u_ndi, Q.t())
        delta_v_ndi_split = utils.split_in_layers(self, delta_v_ndi)

        # Compute the targets across layers.
        r_target_previous = [self.input] 
        v_target = []
        for i in range(len(self.layers)):
            v_target.append(delta_v_ndi_split[i] + torch.matmul(r_target_previous[i], self.layers[i].weights.t()))
            if self.layers[i].use_bias:
                v_target[i] += self.layers[i].bias.unsqueeze(0).expand_as(v_target[i])
            r_target_previous.append(self.layers[i].forward_activationfunction(v_target[i]))

        r_target = r_target_previous[1:]
        target_ndi = r_target[-1]

        return u_ndi, v_target, r_target, target_ndi, delta_v_ndi_split


    def dummy_forward(self, h, i):
        """
        Propagates the activations h of layer i forward to the output of the
        network, without saving activations and linear activations in the layer
        objects.
        Args:
            h (torch.Tensor): activations
            i (int): index of the layer of which h are the activations
        Returns: output of the network with h as activation for layer i
        """

        y = h

        for layer in self.layers[i + 1:]:
            y = layer.dummy_forward(y)

        return y

    @torch.no_grad()
    def dynamical_inversion(self, output_target, alpha=0.001, dt=0.3, tmax=100,
                            epsilon=0.5, makeplots=False,
                            compare_with_ndi=True, savedir=r'.\logs'):
        """
        Performs DFC in real time, that is, controlling all hidden layers simultaneously.
        """
        
        tmax = np.round(tmax).astype(int)
        batch_size = self.layers[0].activations.shape[0]
        plot_title = '' 
        error_str = None
        error = False

        r_feedforward = [l.activations for l in self.layers]
        v_feedforward = [l.linearactivations for l in self.layers]

        if not self.efficient_controller:
            r_target, u, (va, vb, vs), sample_error = \
                self.controller(output_target, alpha, dt, tmax,
                                mode=self.simulation_mode,
                                inst_system_dynamics=self.inst_system_dynamics,
                                k_p=self.k_p,
                                noisy_dynamics=self.noisy_dynamics,
                                inst_transmission=self.inst_transmission,
                                time_constant_ratio=self.time_constant_ratio,
                                apical_time_constant=self.apical_time_constant,
                                sparse=True,
                                proactive_controller=self.proactive_controller,
                                sigma=self.sigma,
                                sigma_output=self.sigma_output,)

            if makeplots:
                plot_title = plot_title + '_Epoch{epoch}_'.format(epoch=self.epoch)
                if error: print(error_str)

                if compare_with_ndi:
                    u_ndi, v_targets_ndi, r_targets_ndi, target_ndi, delta_v_ndi = \
                        self.non_dynamical_inversion(output_target, alpha)
                    self.create_plots_w_ndi(r_target, va, vb, vs, r_feedforward, r_target[-1], output_target,
                                            u_ndi, delta_v_ndi, savedir=savedir, title=plot_title)
                else:
                    self.create_plots_di(r_target, va, vb, vs, r_feedforward, r_target[-1], output_target,
                                         savedir=savedir, title=plot_title)

            converged, diverged = self.check_convergence(r_target, r_feedforward, output_target,
                                                             u, sample_error, epsilon, batch_size)

            if self.save_eigenvalues_bcn:
                vs_bcn = [v[-1] for v in vs]
                A, max_eig, keys = self.compute_A_matrices(vs_bcn, linear=True)
                self.max_eig_bcn = max_eig

            if not self.include_non_converged_samples:
                indices = converged == 0
                indices = utils.bool_to_indices(indices)

                for i in range(self.depth):
                    vs[i][:, indices, :] = v_feedforward[i][indices, :]
                    vb[i][:, indices, :] = v_feedforward[i][indices, :]
                    va[i][:, indices, :] = 0.
                    r_target[i][:, indices, :] = r_feedforward[i][indices]
                u[:, indices, :] = 0.

            r_target_ss = [r[-1] for r in r_target]
            target_ss = r_target_ss[-1]
            u_ss = u[-1]
            va_ss = [v[-1] for v in va]
            vb_ss = [v[-1] for v in vb]
            vs_ss = [v[-1] for v in vs]
            delta_v_ss = [vs_ss[i] - vb_ss[i] for i in range(len(vs_ss))]

            return u_ss, vs_ss, r_target_ss, target_ss, delta_v_ss, (vs, vb, r_target)

        else:
            r_target, u, (va, vb, vs), sample_error = \
                self.controller_efficient(output_target, alpha, dt, tmax,
                                mode=self.simulation_mode,
                                inst_system_dynamics=self.inst_system_dynamics,
                                k_p=self.k_p,
                                noisy_dynamics=self.noisy_dynamics,
                                inst_transmission=self.inst_transmission,
                                time_constant_ratio=self.time_constant_ratio,
                                apical_time_constant=self.apical_time_constant,
                                sparse=True,
                                continuous_updates_forward=self.grad_deltav_cont,
                                continuous_updates_feedback=False,
                                proactive_controller=self.proactive_controller,
                                sigma=self.sigma,
                                sigma_output=self.sigma_output,)

            if self.save_eigenvalues_bcn:
                vs_bcn = [v[-1] for v in vs]
                A, max_eig, keys = self.compute_A_matrices(vs_bcn, linear=True)
                self.max_eig_bcn = max_eig

            converged, diverged = self.check_convergence(r_target, r_feedforward, output_target,
                                               u, sample_error, epsilon, batch_size)

            if not self.include_non_converged_samples:
                indices = converged == 0
                indices = utils.bool_to_indices(indices)

                for i in range(self.depth):
                    vs[i][indices, :] = v_feedforward[i][indices]
                    vb[i][indices, :] = v_feedforward[i][indices]
                    va[i][indices, :] = 0.
                    r_target[i][indices, :] = r_feedforward[i][indices]
                u[indices, :] = 0.

            r_target_ss = r_target
            target_ss = r_target_ss[-1]
            u_ss = u
            va_ss = va
            vb_ss = vb
            vs_ss = vs
            delta_v_ss = [vs_ss[i] - vb_ss[i] for i in range(len(vs_ss))]

            return u_ss, vs_ss, r_target_ss, target_ss, delta_v_ss, (vs, vb, r_target)


    def check_convergence(self, r_target, r_feedforward, output_target, u,
                          sample_error, epsilon, batch_size):
        threshold_convergence = 1e-5
        threshold_divergence = 1

        if not self.efficient_controller:
            u = u[-1]
            r_target = r_target[-1][-1]
        else:
            r_target = r_target[-1]


        diff = self.compute_loss((output_target - self.alpha_di * u), r_target)
        norm = torch.norm(r_feedforward[-1], dim=1).detach()

        converged = ((diff / norm) < threshold_convergence) * (sample_error[-1] < epsilon ** 2 * sample_error[0])
        diverged = (diff / norm) > threshold_divergence
        self.converged_samples_per_epoch += sum(converged).detach().cpu().numpy()
        self.diverged_samples_per_epoch += sum(diverged).detach().cpu().numpy()
        self.not_converged_samples_per_epoch += (batch_size - sum(converged) - sum(diverged)).detach().cpu().numpy()

        return converged, diverged


    def create_plots_di(self, r_target, va, vb, vs, r_feedforward, r_target_L, output_target, savedir='./logs', title=""):
        L = len(r_target)
        fig, axs = plt.subplots(L, 2, figsize=(15, 10))
        lw = 3
        alpha = 0.1

        for i in range(L):
            ax1 = axs[i, 0]
            if len(r_target[i].shape)>2:
                # take the last iteration which is just the steady state one
                r_target_i = r_target[i][-1]
            else:
                r_target_i = r_target[i]

            deviation_per_sample = self.compute_loss(r_feedforward[i], r_target_i, axis=1).cpu().numpy()
            # deviation_per_sample = self.compute_loss(r_feedforward[i], r_target[i], axis=2).cpu().numpy()
            ax1.plot(deviation_per_sample, color='k', alpha=alpha)
            # ax1.plot(np.mean(deviation_per_sample, axis=1), color='k', linewidth=lw)
            ax1.set_title('Deviation between feedforward and target activations')
            if i < L-1:
                ax2 = axs[i, 1]
                diff = self.compute_loss(va[i] + vb[i], vs[i], axis=2).cpu().numpy()
                ax2.semilogy(diff, color='orange', alpha=alpha)
                ax2.semilogy(np.mean(diff, axis=1), color='orange', linewidth=lw)
                ax2.set_title('Diff between ^v and (vb + va) = (Wi*r_i-1 + Qi*u)')
        if len(r_target_L.shape)>2:
                # take the last iteration which is just the steady state one
                r_target_L = r_target_L[-1]
        error_per_sample = self.compute_loss(output_target, r_target_L, axis=1).cpu().numpy()
        # error_per_sample = self.compute_loss(output_target, r_target_L, axis=2).cpu().numpy() 
        axs[L-1,1].semilogy(error_per_sample, color='red', alpha=alpha)
        # axs[L-1,1].semilogy(np.mean(error_per_sample, axis=1), color='red', linewidth=lw)
        axs[L-1,1].set_title('Error signal = output target - ^r_L')

        now = datetime.now()
        savepath = os.path.join(savedir, title + now.strftime("_%Y_%m_%d__%H_%M_%S_%f") + '.png')
        plt.savefig(savepath)
        print('Figure saved in '+ savepath)
        plt.close()


    def create_plots_w_ndi(self, r_target, va, vb, vs, r_feedforward, r_target_L, output_target, u_ndi, delta_v_ndi,
                           savedir='./logs', title=""):

        L = len(r_target)
        fig, axs = plt.subplots(L, 3, figsize=(15, 10))
        lw = 3
        alpha = 0.1 
        eps = 1e-20 

        for i in range(L):
            ax1 = axs[i, 0]
            deviation_per_sample = self.compute_loss(r_feedforward[i], r_target[i], axis=2).cpu().numpy() + eps
            ax1.plot(deviation_per_sample, color='k', alpha=alpha)
            ax1.plot(np.mean(deviation_per_sample, axis=1), color='k', linewidth=lw)
            ax1.set_title('Deviation between feedforward and target activations')

            if i < L-1:
                ax2 = axs[i, 1]
                diff = self.compute_loss(va[i] + vb[i], vs[i], axis=2).cpu().numpy() + eps
                ax2.semilogy(diff, color='orange', alpha=alpha)
                ax2.semilogy(np.mean(diff, axis=1), color='orange', linewidth=lw)
                ax2.set_title('Diff between ^v and (vb + va) = (Wi*r_i-1 + Qi*u)')

                ax3 = axs[i, 2]
                diff_ndi = self.compute_loss(va[i], delta_v_ndi[i], axis=2).cpu().numpy() + eps
                ax3.semilogy(diff_ndi, color='blue', alpha=alpha)
                ax3.semilogy(np.mean(diff_ndi, axis=1), color='blue', linewidth=lw)
                ax3.set_title('Diff delta_v_ndi and va=Qu ')


        error_per_sample = self.compute_loss(output_target, r_target_L, axis=2).cpu().numpy() + eps
        axs[L-1,1].semilogy(error_per_sample, color='red', alpha=alpha)
        axs[L-1,1].semilogy(np.mean(error_per_sample, axis=1), color='red', linewidth=lw)
        axs[L-1,1].set_title('Error signal = output target - ^r_L')
        fig.delaxes(axs[L-1,2])

        now = datetime.now()
        savepath = os.path.join(savedir, title + now.strftime("_%Y_%m_%d__%H_%M_%S_%f") + '.png')
        plt.savefig(savepath)
        print('Figure saved in '+ savepath)
        plt.close()


    def compute_error(self, output_target, r_target):
        r"""Compute the error e(t) in the predictions.

        By default this error is computed as in the DFC paper according to:

        ..math::

            e(t) = r_L^* - r_L(t)

        For a mean-squared error (MSE) loss
        :math:`\mathcal{L} = \frac{1}{2}\snorm{r_L^* - r_L(t)}_{2}^{2}`,
        this can be seen as the gradient of the loss with respect to the output
        activations :math:`r_L(t)`. This notion can be generalized to other
        losses, and we can instead write the error as:

        ..math::

            e(t) = -\left.\pder{r_L{\mathcal{L}}\right\rvert_{r_L=r_L(t)}

        In this function we hard-code the solution of this equation for the
        MSE loss mentioned above as well as for the cross-entropy loss. Which
        one of these is used will be determined by the attribute
        ``loss_function_name``, which by default is the MSE loss.

        So for cross-entropy loss, we return the following error:

        ..math::

            e(t) = r_L^* - softmax(r_L(t))

        Args:
            output_target (torch.Tensor): The desired output :math:`r_L^*`.
            r_target (torch.Tensor): The current output :math:`r_L(t)`.

        Returns:
            (torch.Tensor): The error :math:`e(t)`.
        """
        assert output_target.shape == r_target.shape

        if self.loss_function_name == 'mse':
            return output_target - r_target
        elif self.loss_function_name == 'cross_entropy':
            return output_target - torch.softmax(r_target, dim=1)
        else:
            raise ValueError('Loss function %s ' % self.loss_function_name + \
                             'not recognized.')


    def compute_loss(self, output_target, r_target, axis=1):
        r"""Compute the loss in the predictions.

        This function is mostly used to check for convergence.
        By default this error is computed as in the DFC paper for each sample
        according to:

        ..math::

            \mathcal{L} = \frac{1}{2}\snorm{r_L^* - r_L(t)}_{2}^{2}

        However, if ``loss_function_name==cross_entropy`` we compute the
        following:

        ..math::

            \mathcal{L} = - (r_L^* * \log softmax(r_L(t))

        Args:
            output_target (torch.Tensor): The desired output :math:`r_L^*`.
            r_target (torch.Tensor): The current output :math:`r_L(t)`.
            axis (int): The axis across which to compute the norm.

        Returns:
            (torch.Tensor): The list of loss values in the mini-batch.
        """
        assert output_target.shape == r_target.shape

        if self.loss_function_name == 'mse':
            return torch.norm(output_target - r_target, dim=axis, p=2).detach()
        elif self.loss_function_name == 'cross_entropy':
            return utils.cross_entropy(r_target, output_target).detach()
        else:
            raise ValueError('Loss function %s ' % self.loss_function_name + \
                             'not recognized.')


    def controller(self, output_target, alpha, dt, tmax, mode='blockwise',
                   inst_system_dynamics=False, k_p=0., noisy_dynamics=False,
                   sparse=True, inst_transmission=False, time_constant_ratio=1.,
                   apical_time_constant=-1, proactive_controller=False,
                   sigma=0.01, sigma_output=0.01):
        r"""
        Simulate the feedback control loop for tmax timesteps. The following
        continuous time ODEs are simulated
        with time interval ``dt``:

        ..math::
            \frac{\tau_v}{\tau_u}\frac{d v_i(t)}{dt} = \
                -v_i(t) + W_i r_{i-1}(t) + b_i + Q_i u(t) \\
            \frac{d u(t)}{dt} = e(t) + k_p\frac{d e(t)}{dt} - \alpha u(t) \\
            e(t) = r_L^* - r_L(t)

        Note that we use a ratio :math:`\frac{\tau_v}{\tau_u}` instead of two
        separate time constants for :math:`v` and :math`u`, as a scaling of
        both timeconstants can be absorbed in the simulation timestep ``dt``.
        IMPORTANT: ``time_constant_ratio`` should never be taken smaller than
        ``dt``, as the the forward Euler method will become unstable by
        default (the simulation steps will start to 'overshoot').

        If ``inst_transmission=False``, the forward Euler method is used to
        simulate the differential equation. If ``inst_transmission=True``, a
        slight modification is made to the forward Euler method, assuming that
        we have instant transmission from one layer to the next: the basal
        voltage of layer i at timestep ``t`` will already be based on the
        forward propagation of the somatic voltage of layer i-1 at timestep ``t``,
        hence including the feedback of layer i-1 at timestep ``t``.
        It is recommended to put ``inst_transmission=True`` when the
        ``time_constant_ratio`` is approaching ``dt``, as then we are
        approaching the limit of instantaneous system dynamics in the simulation
        where inst_transmission is always used (See below).

        If ``inst_system_dynamics=True``, we assume that the time constant of the
        system (i.e. the network) is much smaller than that of the controller
        and we approximate this by replacing the dymical equations for v_i by
        their instantaneous equivalents:

        ..math::
            v_i(t) = W_i r_{i-1}(t) + b_i + Q_i u(t)

        Note that ``inst_transmission`` will always be put on True (overridden)
        in combination with inst_system_dynamics.

        If ``proactive_controller=True``, the control input u[k+1] will be used
        to compute the apical voltages v^A[k+1], instead of the control
        input u[k]. This is a slight variation on the forward Euler method and
        and corresponds to the conventional discretized control schemes.

        If ``noisy_dynamics=True``, noise is added to the apical compartment of
        the neurons. We now simulate the apical compartment with its own dynamics,
        as the ``normal_controller`` feedback learning rule needs access to the
        noisy apical compartment. We use the following stochastic differential
        equation for the apical compartment:
        ..math::
            \tau_A d v_i^A = (-v_i^A + Q_i u)dt + \sigma dW
        with W the Wiener process (Brownian motion) with covariance matrix I.
        This is simulated with the Euler-Maruyama method:
        ..math::
            v_i^A[k+1] = v_i^A[k] + \Delta t / \tau_A (-v_i^A[k] + Q_i u[k]) + \
                \sigma / sqrt(\Delta t / \tau_A) \Delta W

        with :math:`\Delta W` drawn from the zero-mean Gaussian distribution with
        covariance I. The other dynamical
        equations in the system remain the same, except that :math:`Q_i u` is
        replaced by :math:`v_i^A`

        ..math::
            \tau_v \frac{d v_i(t)}{dt} = -v_i(t) + W_i r_{i-1}(t) + b_i + v_i^A

        One can opt for instantaneous apical compartment dynamics by putting
        its timeconstant :math:`tau_A` (``apical_time_constant``) equal to
        ``dt``. This is not encouraged for training the feedback weights with
        ``normal_controller`` fb learning rule, but can be used for simulating
        noisy system dynamics for training the forward weights, resulting in:
        ..math::
            \tau_v d v_i(t)} = (-v_i(t) + W_i r_{i-1}(t) + b_i + Q_i u(t) )dt +\
                \sigma dW

        which can again be similarly discretized with the Euler-Maruyama method.

        Note that for training the feedback weights with the ``normal_controller``
        learning rule, it is recommended to put ``inst_transmission=True``, such
        that the noise of all layers can influence the output at the current
        timestep, instead of having to wait for a couple of timesteps, depending
        on the layer depth.

        Note that in
        the current implementation, we interpret that the noise is added in
        the apical compartment, and that the basal and somatic compartments
        are not noisy. At some point we might want to also add noise in the
        somatic and basal compartments for physical realism.

        Args:
            output_target (torch.Tensor): The output target :math:`r_L^*` that is used by
                the controller to compute the control error :math:`e(t)`.
            alpha (float): The leakage term of the controller
            dt (float): the time interval used in the forward Euler method
            tmax (int): the maximum number of timesteps
            mode ['blockwise', 'layerwise']: String indicating whether the
                dynamics should be simulated in a layerwise manner where we
                loop through the layers, or in a blockwise manner where we use
                block matrices and concatenated vectors to compute the updates
                in one go. Both options produce the same results, it's only a
                matter of efficiency. Use 'blockwise' preferably. If
                ``inst_system_dynamics`` and/or ``inst_transmission` is used,
                ``mode=layerwise`` will be used (overwritten if necessary).
            inst_system_dynamics (bool): Flag indicating whether we should
                replace the system dynamics by their instantaneous counterpart.
                If True, `inst_transmission`` will be overwritten to ``True``.
            k_p (float): The positive gain parameter for the proportional part
                of the controller. If it is equal to zero (which is the default),
                no proportional control will be used, only integral control.
            noisy_dynamics (bool): Flag indicating whether noise should be
                added to the dynamcis.
            sparse (bool): Flag indicating whether, when computing all updates in
                one go (mode 'blockwise'), the weights matrix W is a sparse matrix
                or a block diagonal matrix. Both produce identical results, and the
                sparse version is moderately to substantially faster, and therefore
                preferred.
            inst_transmission (bool): Flag indicating whether the modified
                version of the forward Euler method should be used, where it is
                assumed that there is instant transmission between layers (but
                not necessarily instant voltage dynamics). See the docstring
                above for more information.
            time_constant_ratio (float): ratio of the time constant of the
                voltage dynamics w.r.t. the controller dynamics.
            apical_time_constant (float): time constant of the apical
                compartment. If apical_time_constant is -1, we assume that
                the user does not want
                to model the apical compartment dynamics, but assumes instant
                transmission to the somatic compartment instead (i.e. apical time
                constant of zero).

        Returns (tuple): Ordered tuple containing
            r_target (list): A list with at index ``i`` a torch.Tensor of
                dimension :math:`t_{max}\times B \times n_i` containing the
                firing rates of layer i for each timestep.
            u (torch.Tensor): A tensor of dimension
                :math:`t_{max}\times B \times n_L` containing the control input
                for each timestep
            (v_a, v_b, v_s) (tuple): A tuple with 3 elements, each containing
                a list with at index ``i`` a torch.Tensor of
                dimension :math:`t_{max}\times B \times n_i` containing the
                voltage levels of the apical, basal or somatic compartments
                respectively.
            sample_error (torch.Tensor): A tensor of dimension
                :math:`t_{max} \times B` containing the L2 norm of the error
                e(t) at each timestep.
        """

        if k_p < 0:
            raise ValueError('Only positive values for k_p are allowed')

        if inst_system_dynamics:
            inst_transmission = True

        if inst_transmission:
            mode = 'layerwise'

        if apical_time_constant == -1:
            apical_time_constant = dt

        assert apical_time_constant > 0

        batch_size = output_target.shape[0]
        tmax = int(tmax)
        
        L = len(self.layers) 
        layer_input_dims = [l.weights.shape[1] for l in self.layers]
        layer_output_dims = [l.weights.shape[0] for l in self.layers]

        # If hidden activations are linear, then J doen't depend on the samples
        if self.use_jacobian_as_fb and self.activation == 'linear':
            # J = self.compute_full_jacobian(linear=True)
            J = self.compute_full_jacobian(linear=True, steady_state=self.at_steady_state, retain_graph=False, r_targets=self.r_target, average_ss=self.average_ss)

        if mode == 'layerwise':
            size_output = output_target.shape[1]
            u = torch.zeros((tmax, batch_size, size_output))
            va = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # apical voltage = Ki u
            vb = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # basal voltage = Wi h_target_i-1
            vs = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # somatic voltage
            noise_filtered = [torch.zeros((batch_size, l)) for l in layer_output_dims] # exponentially filtered white noise
            if k_p > 0:
                u_int = torch.zeros((tmax, batch_size, size_output))
            r_target = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # just v_soma after non-linearity
            sample_error = torch.ones((tmax, batch_size)) * 10
            
            # # reset activations in every layer to dummy feedforward activations 
            # if self.use_diff_hebbian_updates or self.use_stdp_updates:
            #     y = self.forward(self.input, dummy_reset=True)

            for i in range(L):
                vb[i][0, :] = self.layers[i].linearactivations
                vs[i][0, :] = self.layers[i].linearactivations
                r_target[i][0, :] = self.layers[i].activations
            sample_error[0] = self.compute_loss(output_target, r_target[-1][0, :])

            for t in range(tmax - 1):

                e = self.compute_error(output_target, r_target[-1][t])

                # save time to convergence measures
                if self.save_stdp_measures:
                    # self.error_time_to_convergence[self.epoch,t] += torch.abs(torch.mean(e)*torch.pow(torch.tensor(10), torch.tensor(16)))
                    self.error_time_to_convergence[self.epoch,t] += torch.abs(torch.mean(e))

                # If hidden activations are nonlinear, then J does depend on the
                # samples (derivative of their activations).
                if self.use_jacobian_as_fb and self.activation != 'linear':
                    # For efficiency, we only compute J every few timesteps.
                    # The smaller dt, the less often we want to log.
                    log_freq = 1 / (dt * 100)
                    if log_freq < 1:
                        log_freq = 1
                    if t == 0 or t % log_freq == 0:
                        # J = self.compute_full_jacobian(linear=True)
                        J = self.compute_full_jacobian(linear=True, steady_state=self.at_steady_state, retain_graph=False, r_targets=self.r_target, average_ss=self.average_ss)

                if k_p > 0.: 
                    u_int[t + 1] = u_int[t] + dt * (e - alpha * u[t])
                    u[t + 1] = u_int[t + 1] + k_p * e
                else:
                    u[t + 1] = u[t] + dt * (e - alpha * u[t])

                def layer_iteration(i):
                    if i == 0:
                        r_previous = self.input
                    else:
                        if inst_transmission:
                            r_previous = r_target[i - 1][t + 1]
                        else:
                            r_previous = r_target[i - 1][t]

                    a = r_previous.mm(self.layers[i].weights.t())
                    if self.layers[i].bias is not None:
                        a += self.layers[i].bias.unsqueeze(0).expand_as(a)
                    vb[i][t + 1, :] = a

                    def get_control_signal(t, u_aux):
                        """Get the control signal Qu for the given timestep.

                        By default, this computes :math:`Qu` but in case the option
                        `use_jacobian_as_fb``is active, this computes :math:`Ju`.

                        Args:
                            t (int): The timestep.
                            u_aux (torch.Tensor): The control u to use. Can be
                                low-pass filtered or not, depending on
                                `low_pass_filter_u`.

                        Returns:
                            (torch.Tensor): The control signal.
                        """

                        if self.use_jacobian_as_fb:
                            batch_size = u_aux.shape[1]
                            n_out = u_aux.shape[2]

                            # Select the correct Jacobian block.
                            J_sq = J.view(batch_size * n_out, J.shape[-1])
                            Ji = utils.split_in_layers(self, J_sq)[i]
                            Ji = Ji.view(batch_size, n_out, Ji.shape[-1])

                            return torch.matmul(u_aux[t].unsqueeze(1), Ji).squeeze()
                        else:
                            return torch.mm(u_aux[t], \
                                            self.layers[i].feedbackweights.t())

                    # Get the control signal.
                    control_signal = get_control_signal(\
                                        t + 1 if proactive_controller else t, u)

                    va[i][t + 1, :] = va[i][t,:] + dt / apical_time_constant * \
                                          (- va[i][t, :] + control_signal)
                    va[i][t + 1, :] = control_signal

                    if noisy_dynamics:
                        if i == self.depth-1:
                            sigma_copy = sigma_output
                        else:
                            sigma_copy = sigma
                        
                        if self.target_stepsize==0.5:
                            new_noise = torch.randn_like(va[i][t+1,:])
                            # Warning: for very small dt, we might need to change the implementation for
                            # numerical stability and work with tau_noise*sqrt(dt) instead of
                            # alpha_noise/sqrt(dt).
                            alpha_noise = self.alpha_noise # TODO should be dt / self.tau_noise
                            noise_filtered[i] = (alpha_noise/np.sqrt(dt)) * new_noise + \
                                                (1-alpha_noise) * noise_filtered[i]
                            va[i][t + 1, :] += sigma_copy * noise_filtered[i]
                        else:
                            va[i][t + 1, :] += sigma_copy * \
                                            np.sqrt(dt)/apical_time_constant * \
                                            torch.randn_like(va[i][t+1,:])

                    if inst_system_dynamics: 
                        vs[i][t + 1, :] = va[i][t + 1, :] + vb[i][t + 1, :]
                    else: 
                        vs[i][t + 1, :] = vs[i][t, :] + dt / time_constant_ratio \
                                          * (va[i][t + 1, :] + vb[i][t + 1, :] -
                                             vs[i][t, :])

                    r_target[i][t + 1, :] = self.layers[i].forward_activationfunction(vs[i][t + 1, :])
                    # save r_tartgets for computation of condition 2
                    self.r_target = r_target

                    # We need to store these to compute correctly the Jacobian.
                    if self.use_jacobian_as_fb:
                        self.layers[i].linearactivations = vs[i][t + 1, :]
                        self.layers[i].activations = r_target[i][t + 1, :]

                if not inst_transmission:
                    for i in range(L - 1, 0 - 1, -1):
                        layer_iteration(i)

                else:
                    for i in range(L):
                        layer_iteration(i)

                sample_error[t + 1] = self.compute_loss(output_target, r_target[-1][t + 1, :])
        
            if self.target_stepsize==0.5:
                # As we are standard using noisy dynamics, the last value of u will be noisy, and we should
                # average over u to cancel out the noise. I assume that in the last quarter of the simulation,
                # u has converged, so we can average over that interval.
                interval_length=int(tmax/4)
                self.u = torch.sum(u[-interval_length:-1,:,:], dim=0)/float(interval_length)

            return r_target, u, (va, vb, vs), sample_error

        elif mode == 'blockwise':

            start_output_limits = [sum(layer_output_dims[:i]) for i in range(L)]
            end_output_limits   = [sum(layer_output_dims[:i + 1]) for i in range(L)]
            start_input_limits  = [sum(layer_input_dims[:i]) for i in range(L)] 
            end_input_limits    = [sum(layer_input_dims[:i + 1]) for i in range(L)] 
            total_output_dim_hidden = sum(layer_output_dims[:-1])
            total_output_dim = sum(layer_output_dims)
            size_output = output_target.shape[1]
            
            va = torch.zeros((tmax, batch_size, total_output_dim))
            vb = torch.zeros((tmax, batch_size, total_output_dim))
            vs = torch.zeros((tmax, batch_size, total_output_dim))
            u = torch.zeros((tmax, batch_size, size_output))
            if k_p > 0:
                u_int = torch.zeros((tmax, batch_size, size_output))
            if noisy_dynamics:
                xi = torch.zeros((batch_size, total_output_dim))
            r_target = torch.zeros((tmax, batch_size, total_output_dim))  # just non-lin(vs)
            sample_error = torch.ones((tmax, batch_size)) * 10
            
            weight_list = [l.weights for l in self.layers]
            if sparse:
                idx = [[] , []]
                vals = []
                for i, w in enumerate(weight_list):
                    idx[0].append(np.arange(start_output_limits[i], end_output_limits[i]).repeat(w.shape[1]))
                    idx[1].append(np.tile(np.arange(start_input_limits[i], end_input_limits[i]), w.shape[0]))
                    vals.append(w.reshape(-1,))
                idx[0] = np.concatenate(idx[0])
                idx[1] = np.concatenate(idx[1])
                vals = torch.cat(vals)
                W = torch.sparse_coo_tensor(idx, vals).detach()  # already checked W2.to_dense() == W
            else: 
                W = torch.zeros((sum([w.shape[0] for w in weight_list]), sum([w.shape[1] for w in weight_list])))
                for i, w in enumerate(weight_list):
                    W[start_output_limits[i]:end_output_limits[i], start_input_limits[i]:end_input_limits[i]] = w

            if self.use_bias:
                bias_list = [l.bias for l in self.layers]
                B = torch.zeros((sum([b.shape[0] for b in bias_list]), ))
                for i, b in enumerate(bias_list):
                    B[start_output_limits[i]:end_output_limits[i]] = b

            Q = self.full_Q

            vb[0] = torch.cat([l.linearactivations for l in self.layers], dim=1)
            vs[0] = vb[0]
            r_target[0] = torch.cat([l.activations for l in self.layers], dim=1)
            r_output = r_target[0, :, -size_output:]
            sample_error[0] = self.compute_loss(output_target, r_output)

            for t in range(tmax - 1):
                e = self.compute_error(output_target, \
                                       r_target[t, :, -size_output:])
                if k_p > 0.: 
                    u_int[t+1] = u_int[t] + dt * (e - alpha * u[t])
                    u[t+1] = u_int[t+1] + k_p * e
                else: 
                    u[t+1] = u[t] + dt * (e - alpha * u[t])

                r_input = torch.cat([self.input, r_target[t][:, :end_output_limits[L-2]]], dim=1)
                b = (W @ r_input.t()).t()
                if self.layers[0].use_bias:
                    b += B.unsqueeze(0).expand_as(vb[t + 1])
                vb[t + 1] = b

                if not proactive_controller:
                    va[t+1] = va[t] + dt/apical_time_constant * \
                              ( -va[t] + torch.mm(u[t], Q.t()))
                else:
                    va[t + 1] = va[t] + dt / apical_time_constant * \
                                (-va[t] + torch.mm(u[t+1], Q.t()))
                if noisy_dynamics:
                    noise = sigma * np.sqrt(dt)/apical_time_constant * \
                                           torch.randn_like(va[t+1])
                    noise[:, -size_output:] = sigma_output/sigma * noise[:, -size_output:]
                    va[t+1] += noise

                if inst_system_dynamics: 
                    vs[t + 1] = va[t + 1] + vb[t + 1]
                else:
                    vs[t+1] = vs[t] + dt/time_constant_ratio * (-vs[t] + va[t+1] + vb[t+1])
                for i in range(self.depth):
                    r_target[t+1, :, start_output_limits[i]:end_output_limits[i]] = \
                        self.layers[i].forward_activationfunction(
                            vs[t+1, :, start_output_limits[i]:end_output_limits[i]])

                r_output = r_target[t + 1, :, -size_output:]
                sample_error[t+1] = self.compute_loss(output_target, r_output)

            r_target_split = [r_target[:,:, start_output_limits[i]:end_output_limits[i]] for i in range(L)]
            va_split = [va[:,:, start_output_limits[i]:end_output_limits[i]] for i in range(L)]
            vb_split = [vb[:,:, start_output_limits[i]:end_output_limits[i]] for i in range(L)]
            vs_split = [vs[:,:, start_output_limits[i]:end_output_limits[i]] for i in range(L)]

            # save r_tartgets for computation of condition 2
            self.r_target = r_target

            return r_target_split, u, (va_split, vb_split, vs_split), sample_error


    def controller_efficient(self, output_target, alpha, dt, tmax, mode='blockwise',
                   inst_system_dynamics=False, k_p=0., noisy_dynamics=False,
                   sparse=True, inst_transmission=False, time_constant_ratio=1.,
                   apical_time_constant=-1, continuous_updates_forward=False,
                   continuous_updates_feedback=False,
                   proactive_controller=False, sigma=0.01, sigma_output=0.01):
        r"""
        Simulate the feedback control loop for tmax timesteps.
        In contrast with :function:`controller`, this implementation is more
        efficient,  as it does not save the activations of the layers
        during the feedback phase. Note that no plots can be made of the
        dynamics in this case.

        See docstring :func:`tpdi_networks.DFCNetwork.controller` for
        an explanation of what the controller does.

        Args:
            (....): See :func:`tpdi_networks.DFCNetwork.controller`
            continuous_updates_forward (bool): Flag indicating that the weight update
                at each timestep of the feedback phase should be computed
                and accumulated. As in this efficient controller implementation
                the activations are not saved, the weight updates are computed
                in each timestep and accumulated in the ``.grad`` attribute of the
                forward parameters.
            continuous_updates_feedback (bool): Flag indicating that the
                feedback weight updates should be computed and stored in
                the ``.grad`` attribute of the :attr:`feedbackweights` of the
                layers for each timestep. This feature will be used by the
                feedback learning rule ``normal_controller``.

        Returns (tuple): Ordered tuple containing
            r_target (list): A list with at index ``i`` a torch.Tensor of
                dimension :math:`B \times n_i` containing the
                firing rates of layer i for the last timestep.
            u (torch.Tensor): A tensor of dimension
                :math:`B \times n_L` containing the control input
                for the last timestep timestep
            (v_a, v_b, v_s) (tuple): A tuple with 3 elements, each containing
                a list with at index ``i`` a torch.Tensor of
                dimension :math:`B \times n_i` containing the
                voltage levels of the apical, basal or somatic compartments
                respectively for the last timestep.
            sample_error (torch.Tensor): A tensor of dimension
                :math:`2 \times B` containing the L2 norm of the error
                e(t) at the first and last timestep.
        """

        if k_p < 0:
            raise ValueError('Only positive values for k_p are allowed')
        if inst_system_dynamics:
            inst_transmission = True

        if inst_transmission:
            mode = 'layerwise'

        if apical_time_constant is None:
            apical_time_constant = dt

        r_feedforward = [l.activations for l in self.layers]

        batch_size = output_target.shape[0]
        tmax = int(tmax)
        L = len(self.layers)
        layer_input_dims = [l.weights.shape[1] for l in self.layers]
        layer_output_dims = [l.weights.shape[0] for l in self.layers]

        if mode == 'layerwise':
            size_output = output_target.shape[1]
            u = torch.zeros((batch_size, size_output))
            va = [torch.zeros((batch_size, l)) for l in
                  layer_output_dims]
            vb = [torch.zeros((batch_size, l)) for l in
                  layer_output_dims]
            vs = [torch.zeros((batch_size, l)) for l in
                  layer_output_dims]
            if k_p > 0:
                u_int = torch.zeros((batch_size, size_output))
            r_target = [torch.zeros((batch_size, l)) for l in
                        layer_output_dims] 
            sample_error = torch.ones((2, batch_size)) * 10

            for i in range(L):
                vb[i] = self.layers[i].linearactivations
                vs[i] = self.layers[i].linearactivations
                r_target[i] = self.layers[i].activations
            sample_error[0] = self.compute_loss(output_target, r_target[-1])

            for t in range(tmax - 1):
                
                e = self.compute_error(output_target, r_target[-1])

                if proactive_controller:
                    if k_p > 0.: 
                        u_int = u_int + dt * (e - alpha * u)
                        u = u_int + k_p * e
                    else: 
                        u = u + dt * (e - alpha * u)

                if continuous_updates_feedback and t > 0:
                    if k_p > 0.:
                        u_int_dummy = u_int + dt * (e - alpha * u)
                        u_dummy = u_int_dummy + k_p * e
                    else:
                        u_dummy = u + dt * (e - alpha * u)

                    for i in range(self.depth):
                        self.layers[i].compute_feedback_gradients(
                            -va[i],
                            u_dummy,
                            sigma=sigma,
                            scale=1. / (tmax - 2))

                def layer_iteration(i):
                    if i == 0:
                        r_previous = self.input
                    else:
                        r_previous = r_target[i - 1]

                    a = r_previous.mm(self.layers[i].weights.t())
                    if self.layers[i].bias is not None:
                        a += self.layers[i].bias.unsqueeze(0).expand_as(a)
                    vb[i] = a

                    va[i] = va[i] + dt / apical_time_constant * \
                        (- va[i] + torch.mm(u, self.layers[i].feedbackweights.t()))

                    if noisy_dynamics:
                        if i == self.depth-1:
                            sigma_copy = sigma_output
                        else:
                            sigma_copy = sigma
                        va[i] += sigma_copy * np.sqrt(dt)/ apical_time_constant * \
                                           torch.randn_like(va[i])

                    if inst_system_dynamics: 
                        vs[i] = va[i] + vb[i]
                    else: 
                        vs[i] = vs[i] + dt / time_constant_ratio \
                                          * (va[i] + vb[i] - vs[i])

                    r_target[i] = self.layers[i].forward_activationfunction(vs[i])
                    
                if not inst_transmission:
                    for i in range(L - 1, 0 - 1, -1):
                        layer_iteration(i)

                else:
                    for i in range(L):
                        layer_iteration(i)

                if not proactive_controller:
                    if k_p > 0.: 
                        u_int = u_int + dt * (e - alpha * u)
                        u = u_int + k_p * e
                    else: 
                        u = u + dt * (e - alpha * u)

                if continuous_updates_forward:
                    delta_v = [vs[i] - vb[i] for i in range(len(vs))]

                    sample_error[1] = self.compute_loss(output_target,
                                                        r_target[-1])
                    converged, diverged = self.check_convergence(
                            r_target, r_feedforward, output_target, u, sample_error,
                            epsilon=0.5, batch_size=batch_size)

                    for i in range(self.depth):
                        if i == 0:
                            r_previous = self.input
                        else:
                            r_previous = r_target[i - 1]

                        if not self.include_non_converged_samples:
                            delta_v[i][diverged == 1, :] = 0.

                        self.layers[i].compute_forward_gradients(
                            delta_v[i],
                            r_previous,
                            learning_rule=self.learning_rule,
                            scale=1. / (tmax - 1))


            sample_error[1] = self.compute_loss(output_target, r_target[-1])

            # save r_tartgets for computation of condition 2
            self.r_target = r_target

            return r_target, u, (va, vb, vs), sample_error

        elif mode == 'blockwise':

            start_output_limits = [sum(layer_output_dims[:i]) for i in range(L)]
            end_output_limits = [sum(layer_output_dims[:i + 1]) for i in
                                 range(L)]
            start_input_limits = [sum(layer_input_dims[:i]) for i in
                                  range(L)]
            end_input_limits = [sum(layer_input_dims[:i + 1]) for i in
                                range(L)]
            total_output_dim_hidden = sum(layer_output_dims[:-1])
            total_output_dim = sum(layer_output_dims)
            size_output = output_target.shape[1]

            va = torch.zeros((batch_size, total_output_dim))
            vb = torch.zeros((batch_size, total_output_dim))
            vs = torch.zeros((batch_size, total_output_dim))
            u = torch.zeros((batch_size, size_output))
            if k_p > 0:
                u_int = torch.zeros((batch_size, size_output))
            r_target = torch.zeros(
                (batch_size, total_output_dim)) 
            sample_error = torch.ones((2, batch_size)) * 10
            
            weight_list = [l.weights for l in self.layers]
            if sparse:
                idx = [[], []]
                vals = []
                for i, w in enumerate(weight_list):
                    idx[0].append(np.arange(start_output_limits[i],
                                            end_output_limits[i]).repeat(
                        w.shape[1]))
                    idx[1].append(np.tile(
                        np.arange(start_input_limits[i], end_input_limits[i]),
                        w.shape[0]))
                    vals.append(w.reshape(-1, ))
                idx[0] = np.concatenate(idx[0])
                idx[1] = np.concatenate(idx[1])
                vals = torch.cat(vals)
                W = torch.sparse_coo_tensor(idx,
                                            vals).detach()
            else: 
                W = torch.zeros((sum([w.shape[0] for w in weight_list]),
                                 sum([w.shape[1] for w in weight_list])))
                for i, w in enumerate(weight_list):
                    W[start_output_limits[i]:end_output_limits[i],
                    start_input_limits[i]:end_input_limits[i]] = w

            if self.use_bias:
                bias_list = [l.bias for l in self.layers]
                B = torch.zeros((sum([b.shape[0] for b in bias_list]),))
                for i, b in enumerate(bias_list):
                    B[start_output_limits[i]:end_output_limits[i]] = b

            Q = self.full_Q

            vb = torch.cat([l.linearactivations for l in self.layers], dim=1)
            vs = vb
            r_target = torch.cat([l.activations for l in self.layers], dim=1)
            r_output = r_target[:, -size_output:]
            sample_error[0] = self.compute_loss(output_target, r_output)

            for t in range(tmax - 1):
                e = self.compute_error(output_target, \
                                       r_target[:, -size_output:])
                if proactive_controller:
                    if k_p > 0.: 
                        u_int = u_int + dt * (e - alpha * u)
                        u = u_int + k_p * e
                    else: 
                        u = u + dt * (e - alpha * u)

                if continuous_updates_feedback and t > 0:
                    if k_p > 0.: 
                        u_int_dummy = u_int + dt * (e - alpha * u)
                        u_dummy = u_int_dummy + k_p * e
                    else: 
                        u_dummy = u + dt * (e - alpha * u)

                    va_list = utils.split_in_layers(self, va)
                    
                    for i in range(self.depth):
                        self.layers[i].compute_feedback_gradients(
                            -va_list[i],
                            u_dummy,
                            sigma=sigma,
                            scale=1./(tmax-2))

                r_input = torch.cat(
                    [self.input, r_target[:, :end_output_limits[L - 2]]],
                    dim=1)
                b = (W @ r_input.t()).t()
                if self.layers[0].use_bias:
                    b += B.unsqueeze(0).expand_as(vb)
                vb = b

                va = va + dt / apical_time_constant * (
                            -va + torch.mm(u, Q.t()))
                if noisy_dynamics:
                    noise = sigma * np.sqrt(dt) / apical_time_constant * \
                            torch.randn_like(va)
                    noise[:, -size_output:] = sigma_output / sigma * noise[:,
                                                                      -size_output:]
                    va += noise

                vs = vs + dt / time_constant_ratio * (
                            -vs + va + vb)

                for i in range(self.depth):
                    r_target[:, start_output_limits[i]:end_output_limits[i]] = \
                        self.layers[i].forward_activationfunction(
                            vs[:, start_output_limits[i]:end_output_limits[i]])

                if not proactive_controller:
                    if k_p > 0.: 
                        u_int = u_int + dt * (e - alpha * u)
                        u = u_int + k_p * e
                    else:
                        u = u + dt * (e - alpha * u)

                if continuous_updates_forward:
                    delta_v = vs - vb
                    delta_v_list = utils.split_in_layers(self, delta_v)
                    r_target_list = utils.split_in_layers(self, r_target)

                    sample_error[1] = self.compute_loss(output_target,
                                                        r_target_list[-1])
                    converged, diverged = self.check_convergence(
                        r_target_list, r_feedforward, output_target, u, sample_error,
                        epsilon=0.5, batch_size=batch_size)

                    for i in range(self.depth):
                        if i == 0:
                            r_previous = self.input
                        else:
                            r_previous = r_target_list[i-1]

                        if not self.include_non_converged_samples:
                            delta_v_list[i][diverged == 1, :] = 0.

                        self.layers[i].compute_forward_gradients(
                            delta_v_list[i],
                            r_previous,
                            learning_rule=self.learning_rule,
                            scale=1./(tmax-1))

            r_output = r_target[:, -size_output:]
            sample_error[1] = self.compute_loss(output_target, r_output)

            r_target_split = [
                r_target[:, start_output_limits[i]:end_output_limits[i]] for
                i in range(L)]
            va_split = [va[:, start_output_limits[i]:end_output_limits[i]]
                        for i in range(L)]
            vb_split = [vb[:, start_output_limits[i]:end_output_limits[i]]
                        for i in range(L)]
            vs_split = [vs[:, start_output_limits[i]:end_output_limits[i]]
                        for i in range(L)]

            # save r_tartgets for computation of condition 2
            self.r_target = r_target

            return r_target_split, u, (va_split, vb_split, vs_split), sample_error


    def compute_feedback_gradients(self, loss, targets, target_lr, init=False):
        """
        Compute the gradients of the feedback weights for each layer.
        ``self.fb_learning_rule`` determines which learning is used for training
        the feedback weights.
        """

        if self.fb_learning_rule == 'normal_controller':
            self.compute_feedback_gradients_normal_controller(loss, target_lr, targets)
        elif self.fb_learning_rule == 'special_controller':
            self.compute_feedback_gradients_special_controller()
        elif self.fb_learning_rule == 'old_learning_rule':
            self.compute_feedback_gradients_old()
        else:
            raise ValueError('Feedback learning rule "{}" not '
                             'recognized.'.format(self.fb_learning_rule))


    def compute_feedback_gradients_normal_controller(self, loss, target_lr, targets):
        r"""
        Compute the gradients of the feedback weights for each layer,
        according to the following update rule:

        ..math::
            \frac{d Q_i}{dt} = -\frac{1}{\sigma^2} \mathbf{v}^A_i \mathbf{u}^T 
        when the output target is equal to the feedforward output of the
        network (i.e. without feedback) and when noise is applied to the
        network dynamics. Hence, the controller will try to 'counter' the
        noisy dynamics, such that the output is equal to the unnoisy output.
        """

        if self.classification and self.target_stepsize==0.5:
            output_target = targets
        else:
            output_target = self.layers[-1].activations.data
        batch_size = output_target.shape[0]

        r_target, u, (va, vb, vs), sample_error = \
            self.controller(output_target=output_target,
                            alpha=self.alpha_fb,
                            dt=self.dt_di_fb,
                            tmax=self.tmax_di_fb,
                            inst_system_dynamics=self.inst_system_dynamics,
                            k_p=self.k_p_fb,
                            noisy_dynamics=True,
                            inst_transmission=self.inst_transmission_fb,
                            time_constant_ratio=self.time_constant_ratio_fb,
                            apical_time_constant=self.apical_time_constant_fb,
                            proactive_controller=self.proactive_controller,
                            mode=self.simulation_mode,
                            sigma=self.sigma_fb,
                            sigma_output=self.sigma_output_fb)

        u = u[1:,:,:]
        if self.target_stepsize==0.5:
            u_aux = torch.zeros_like(u)
            u_aux[0] = u[0]
            # exponential smoothing
            for t in range(1, len(u)):
                u_aux[t] = self.alpha_u * u[t] + (1 - self.alpha_u) * u_aux[t-1]
            # subtract the exponential smoothing (high pass filter: average out
            # the target clamping)
            u_filtered = u - u_aux
            u = u_filtered

        for i, layer in enumerate(self.layers):
            va_i = va[i][:-1, :, :]

            # compute a layerwise scaling_fb_updates for the feedback weights
            if self.scaling_fb_updates:
                # scale the update for each layer according to the theory
                scaling = (1 + self.time_constant_ratio_fb * (self.alpha_noise / self.dt_di_fb)) \
                        ** (len(self.layers) - i - 1)
            else:
                scaling = 1.
            
            # get the amount of noise used.
            sigma_i = self.sigma_fb
            if i == len(self.layers) - 1:
                sigma_i = self.sigma_output_fb
            
            layer.compute_feedback_gradients_continuous(va_i, u,
                        sigma=sigma_i, scaling=scaling)


    def compute_feedback_gradients_special_controller(self):
        r"""
        Compute the gradients of the feedback weights for each layer,
        according to the following update rule:

        ..math::
            \Delta Q_i = \frac{1}{\sigma^2} \Delta \mathbf{v}_i \mathbf{u}^T
        """

        noise = []
        for layer in self.layers:
            noise.append(self.sigma_fb * torch.randn_like(layer.activations))
        error = self.noisy_dummy_forward(self.input, noise) - \
                self.layers[-1].activations

        for i, layer in enumerate(self.layers):

            # get the amount of noise used.
            sigma_i = self.sigma_fb
            if i == len(self.layers) - 1:
                sigma_i = self.sigma_output_fb

            layer.compute_feedback_gradients(noise[i], error, sigma=sigma_i)


    def compute_feedback_gradients_old(self):
        r"""
        Compute the gradients of the feedback weights for each layer,
        according to the following update rule:

        ..math::
            \Delta Q_i = -\frac{1}{\sigma^2} \Delta \mathbf{v}_i \mathbf{u}^T
        """

        noise = []
        for layer in self.layers:
            noise.append(self.sigma_fb * torch.randn_like(layer.activations))
        output_target = self.noisy_dummy_forward(self.input, noise)

        if self.ndi:
            u_ndi, v_target, r_target, target_ndi, delta_v_ndi_split = \
                self.non_dynamical_inversion(output_target, self.alpha_di)

        else:
            raise NotImplementedError('The dynamical inversion is not '
                                      'implemented yet.')
        for i, layer in enumerate(self.layers):

            # get the amount of noise used.
            sigma_i = self.sigma_fb
            if i == len(self.layers) - 1:
                sigma_i = self.sigma_output_fb
 
            layer.compute_feedback_gradients_old(delta_v_ndi_split[i], u_ndi,
                                             sigma=sigma_i)

    def noisy_dummy_forward(self, input_network, noise):
        """
        Propagates the input forward to the output of the network,
        while adding noise to the linear activations of each layer
        before propagating it to the next.
        This method does not save any activations or linear activations in
        the layer objects.
        Args:
            input_network (torch.Tensor): input of the network
            noise (list): A list containing noise for each layer, with a
                structure consistent with :func:`networks.utils.split_in_layers`
        Returns (torch.Tensor): The resulting output activation of the network.
        """

        y = input_network

        for i, layer in enumerate(self.layers):
            y = layer.noisy_dummy_forward(y, noise[i])

        return y


    def compute_full_jacobian_tilde(self, linear=True, retain_graph=False):
        r"""
        Compute the linear transformation of the full Jacobian for each batch
        sample according to

        ..math::
            \tilde{J} \triangleq V_{\bar{Q}}^T \bar{J} U_{\bar{Q}}

        with :math:`U_{\bar{Q}}` and :math:`V_{\bar{Q}}` the singular vectors
        of :math:`\bar{Q}` (i.e. from its SVD).
        Args:
            (....): see docstring of :func:`compute_full_jacobian`
        Returns (torch.Tensor): A :math:`B \times n_L \times \sum_{l=1}^L n_l`
            dimensional tensor,
            with B the minibatch size and n_l the dimension of layer l,
            containing the linearly transformed Jacobian of the network
            output w.r.t. the
            concatenated activations (pre or post-nonlinearity) of all layers,
            for each minibatch sample
        """

        # full_jacobians = self.compute_full_jacobian(linear=True, retain_graph=retain_graph)
        full_jacobians = self.compute_full_jacobian(linear=True, steady_state=self.at_steady_state, retain_graph=retain_graph, r_targets=self.r_target, average_ss=self.average_ss)
        full_jacobians_tilde = torch.zeros_like(full_jacobians)

        U, S, V = torch.svd(self.full_Q, some=False)

        for b in range(full_jacobians.shape[0]):
            full_jacobians_tilde[b, :, :] = V.T.mm(full_jacobians[b,:,:].mm(U))

        return full_jacobians_tilde


    def compute_A_matrices(self, linear_activations, forward=True, linear=True):
        r"""
        Compute the different A matrices that encode the differential
        equations of the network, so that the eigenvalues can be extracted
        and analyzed.

        ..math::
            z = \begin{pmatrix} \Delta v \\ u \end{pmatrix}  \\
            \dot{z} = \frac{d}{dt} z = A z +
            \begin{pmatrix} 0 \\ \frac{1}{\tau_w} \delta_L \end{pmatrix}

        where A takes different forms depending on the selected dynamics.

        The four possible A are returned:
        * :math:`\tilde{A}_I = \begin{pmatrix} \frac{-1}{\tau_v} (I - \hat{J}) & \frac{1}{\tau_v} \bar{Q} \\
         \frac{-1}{\tau_u} J_L^o & \frac{-\alpha}{\tau_u} I \end{pmatrix}`
        * :math:`A_I = \begin{pmatrix} \frac{-1}{\tau_v} (I - \hat{J}) & \frac{1}{\tau_v} (I - \hat{J}) \bar{Q} \\
         \frac{-1}{\tau_u} \bar{J} & \frac{-\alpha}{\tau_u} I \end{pmatrix}`
        * :math:`A_I^{inst} = \frac{-1}{\tau_u} (\bar{J}\bar{Q} + \alpha I)`
        * :math:`A_{PI} = \begin{pmatrix} \frac{-1}{\tau_v} (I - \hat{J}) & \frac{1}{\tau_v} (I - \hat{J}) \bar{Q} \\
        \frac{-1}{\tau_u} (-\bar{J} + \frac{K_p}{\tau_v} J_L^o) & -\frac{1}{\tau_u}(\alpha I+\frac{K_p}{\tau_v} J_L Q_L)
        \end{pmatrix}`
        
        :param linear_activations: linear activations :math:`v_i` that are to be used. It can either be
        the feedforward activations, or the steady state activations (from dynamical inversion).
        Must be given as a list, one element per layer.
        :param forward: flag which signals whether we are analysing forward dynamics (if True) or feedback dynamics
        (if False). Sets the values alpha, k_p and time_constant_ratio to their corresponding forward or
        feedback values.
        :param linear: whether the network is linear. Passed on to compute the Jacobian.
        :return A: A dictionary which contains the matrices :math:`\tilde{A}_I`, :math:`A_I`, :math:`A_I^{inst}`
                   and :math:`A_{PI}` as described above.
        :return max_eig: A dictionary with identical keys, containing the maximum eigenvalue (real part) of each
                        matrix in A, for each batch sample (in a 1D np array).
        :return keys: list of keys under which the A matrices and the respective max eigenvalues are stored
        """

        if forward:
            alpha = self.alpha_di
            k_p = self.k_p
            tcr = self.time_constant_ratio
        else:
            alpha = self.alpha_fb
            k_p = self.k_p_fb
            tcr = self.time_constant_ratio_fb

        L = self.depth
        layer_jacobians = [None for i in range(L)]
        vectorized_nonlinearity_derivative = [self.layers[i].compute_vectorized_jacobian(linear_activations[i]) for i in range(L)]
        output_activation = self.layers[-1].activations
        batch_size = output_activation.shape[0]
        output_size = output_activation.shape[1]
        
        layer_jacobians[-1] = \
            torch.eye(output_size).repeat(batch_size, 1, 1).reshape(batch_size, output_size, output_size)
        if linear:
            layer_jacobians[-1] = layer_jacobians[-1] \
                                  * vectorized_nonlinearity_derivative[-1].view(batch_size, output_size, 1)

        for i in range(L - 1 - 1, 0 - 1, -1): 
            if linear:
                layer_jacobians[i] = layer_jacobians[i + 1].matmul(self.layers[i + 1].weights) \
                                     * vectorized_nonlinearity_derivative[i].unsqueeze(1)
            else:
                layer_jacobians[i] = (layer_jacobians[i + 1] * vectorized_nonlinearity_derivative[i + 1].unsqueeze(1)) \
                                     .matmul(self.layers[i + 1].weights)

        J_bar = torch.cat(layer_jacobians, dim=2)
        B = batch_size
        sum_i = J_bar.shape[2]

        J_L0 = torch.zeros(J_bar.shape)
        J_L0[:, :, -layer_jacobians[-1].shape[2]:] = layer_jacobians[-1]

        J_hat = torch.zeros((B, sum_i, sum_i))
        io = torch.tensor([[l.weights.shape[0], l.weights.shape[1]] for l in self.layers])
        limits = [torch.sum(io[:i, 0]) for i in range(L + 1)]
        for i in range(1, L):
            J_hat[:, limits[i]:limits[i+1], limits[i-1]:limits[i]] = \
                self.layers[i].weights.unsqueeze(0) * vectorized_nonlinearity_derivative[i-1].unsqueeze(1)

        a_11 = (1/tcr) * (torch.eye(sum_i) - J_hat)
        A = dict()
        A["A_tildeI"] = torch.cat((
                         torch.cat((-1*a_11, (1/tcr)*self.full_Q.expand(B, self.full_Q.shape[0], self.full_Q.shape[1])), dim=2),  # first row
                         torch.cat((-1*J_L0, -alpha * torch.eye(output_size).expand(B, output_size, output_size)), dim=2)  # second row
                        ), dim=1)

        A["A_I"] = torch.cat((
                    torch.cat((-1*a_11, a_11 @ self.full_Q), dim=2), 
                    torch.cat((-1*J_bar, -alpha * torch.eye(output_size).expand(B, output_size, output_size)), dim=2)  # second row
                   ), dim=1) 

        A["A_instI"] = -1*(torch.matmul(J_bar, self.full_Q) + alpha*torch.eye(output_size))

        A["A_PI"] = torch.cat((
                     torch.cat( (-1*a_11, a_11 @ self.full_Q), dim=2), 
                     torch.cat( (-1*J_bar + k_p/tcr * J_L0,
                                -1*(alpha * torch.eye(output_size).expand(B, output_size, output_size)
                                +(k_p/tcr)*layer_jacobians[-1]@self.layers[-1].feedbackweights)),dim=2)
                    ), dim=1)

        max_eig = dict()
        keys = [k for k in A.keys()]
        for k in keys:
            max_eig[k] = np.zeros((B,))

        for b in range(B):
            for k in keys:
                max_eig[k][b] = torch.max(torch.eig(A[k][b]).eigenvalues[:, 0]).detach().cpu().numpy()  # the first value is the real part

        return A, max_eig, keys


    def save_eigenvalues_to_tensorboard(self, writer, step):
        """
        Save the maximum eigenvalues of the different variations of A
        to Tensorboard. Generates a plot of the average and a histogram across samples.
        :param: writer: Tensorboard object to which the data will be written
        :param: step: x-axis index used for tensorboard (normally train_var.batch_idx)
        """

        categ = "max_eig"
        for k in self.max_eig.keys():
            writer.add_scalar(
                tag=categ + "/" + k,
                scalar_value=np.mean(self.max_eig[k]),
                global_step=step)
            writer.add_histogram(
                tag=categ+"/"+k,
                values=self.max_eig[k],
                global_step=step)


    def save_eigenvalues_bcn_to_tensorboard(self, writer, step):
        """
        Save the maximum eigenvalues of the different variations of A
        (before corrections for non-convergence / divergence take place)
        to Tensorboard. Saves both mean and std across the batch.
        :param: writer: Tensorboard object to which the data will be written
        :param: step: x-axis index used for tensorboard (normally train_var.batch_idx)
        """

        categ = "max_eig_bcn"
        for k in self.max_eig_bcn.keys():
            writer.add_scalar(
                tag=categ + "/" + k + "_mean",
                scalar_value=np.mean(self.max_eig_bcn[k]),
                global_step=step)
            writer.add_histogram(
                tag=categ + "/" + k,
                values=self.max_eig_bcn[k],
                global_step=step)


    def save_norm_r_to_tensorboard(self, writer, step):
        """
        Save the modulus of r (post nonlinearity activations) of each layer
        (as relative deviations from the mean of all layers)
        to Tensorboard. Saves both mean and std across the batch.
        :param: writer: Tensorboard object to which the data will be written
        :param: step: x-axis index used for tensorboard (normally train_var.batch_idx)
        """
        """
        iterate over the keys in norm_r and save to Tensorboard
        note that deviations can have positive or negative value
        values distributed around the mean will generate 0 mean 
        but a large std.
        """

        categ = 'norm_r'
        for k in self.norm_r.keys():
            writer.add_scalar(
                tag=categ + "/" + k + "_mean",
                scalar_value=np.mean(self.norm_r[k]),
                global_step=step)
            writer.add_histogram(
                tag=categ + "/" + k,
                values=self.norm_r[k],
                global_step=step)
        writer.add_scalar(
            tag=categ + "/" + "dev_r_mean",
            scalar_value=np.mean(self.dev_r),
            global_step=step)
        writer.add_histogram(
            tag=categ + "/" + "dev_r",
            values=self.dev_r,
            global_step=step)


    def split_full_jacobian_tilde(self, full_jacobian_tilde):
        """
        Split :math:`\tilde{J}` into the block matrices
        :math:`\tilde{J} = [\tilde{J}_1 \tilde{J}_2]` with :math:`\tilde{J}_1`
        a square matrix. Do this for each batch sample.
        Args:
            full_jacobian_tilde (torch.Tensor): the linearly transformed
                full jacobian computed by :func:`compute_full_jacobian_tilde`
        Returns (tuple): A tuple (J_tilde_1, J_tilde_2), with J_tilde_1 and
            J_tilde_2 containing the above specified matrices for each batch
            sample.
        """

        n_L = full_jacobian_tilde.shape[1]
        return (full_jacobian_tilde[:,:,:n_L], full_jacobian_tilde[:,:,n_L:])


    def save_ndi_angles(self, writer, step,
                        save_dataframe=True, save_tensorboard=True):
        """
        Compute the angle between the actual weight updates of the model
        (e.g. resulting from TPDI) on the one hand, and the weight updates
        resulting from ideal inversion (analytical solution, NDI) on
        other hand. These have been stored during training in
        self.layers[i].ndi_update_weights / _bias.
        Save the angle in the tensorboard X writer
        (if ``save_tensorboard=true``) and in the
        corresponding dataframe (if ``save_dataframe=True``)
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard

            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
        """

        ndi_param_updates = []
        net_params = self.get_forward_parameter_list()
        net_param_updates = [p.grad for p in net_params]
        for i in range(self.depth):
            parameter_update = self.layers[i].get_forward_gradients()
            weights_angle = utils.compute_angle(self.layers[i].ndi_updates_weights,
                                                parameter_update[0])
            ndi_param_updates.append(self.layers[i].ndi_updates_weights)
            if self.use_bias:
                bias_angle = utils.compute_angle(self.layers[i].ndi_updates_bias,
                                                 parameter_update[1])
                ndi_param_updates.append(self.layers[i].ndi_updates_bias)

            if save_tensorboard:
                name = 'layer {}'.format(i + 1)
                writer.add_scalar(
                    tag='{}/weight_ndi_angle'.format(name),
                    scalar_value=weights_angle,
                    global_step=step)
                if self.use_bias:
                    writer.add_scalar(
                        tag='{}/bias_ndi_angle'.format(name),
                        scalar_value=bias_angle,
                        global_step=step
                    )
            if save_dataframe:
                self.ndi_angles.at[step, i] = weights_angle.item()

        total_angle = utils.compute_angle(utils.vectorize_tensor_list(ndi_param_updates),
                                          utils.vectorize_tensor_list(net_param_updates))

        if save_tensorboard:
            name = 'total_alignment/ndi_angle'
            writer.add_scalar(
                tag=name,
                scalar_value=total_angle,
                global_step=step
            )
        if save_dataframe:
            self.ndi_angles_network.at[step, 0] = total_angle.item()

    
    def save_dfc_angles(self, writer, step,
                        save_dataframe=True, save_tensorboard=True):
        """
        Compute the angle between the actual weight updates of the model
        (e.g. resulting from TPDI) on the one hand, and the weight updates
        resulting from the original DFC model on
        other hand.
        Save the angle in the tensorboard X writer
        (if ``save_tensorboard=true``) and in the
        corresponding dataframe (if ``save_dataframe=True``)
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard

            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
        """
        self.step = step
        for i in range(len(self.layers)):
            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(
                    tag='{}/weight_dfc_angle'.format(name),
                    scalar_value=self.dfc_angles_list[i],
                    global_step=step)
            if save_dataframe:
                self.dfc_angles.at[step, i] = self.dfc_angles_list[i].item()
        if save_tensorboard:
            writer.add_scalar(
                tag='total_alignment/dfc_angles',
                scalar_value=self.dfc_total_angle,
                global_step=step)
        if save_dataframe:
            self.dfc_angles_network.at[step, 0] = self.dfc_total_angle.item()


    def compute_dfc_angles(self, i, vs_time_i, vb_time_i, central_activities_i, r_previous_time):
        dfc_stdp_udpates = self.layers[i].compute_forward_gradients_deltav_continuous_stdp(
                        vs_time_i, vb_time_i, central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
        dfc_udpates = self.layers[i].compute_forward_gradients_deltav_continuous_dfc(
                        vs_time_i, vb_time_i, central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
        dfc_diff_hebbian_udpates = self.layers[i].compute_forward_gradients_deltav_continuous_diff_hebbian(
                        vs_time_i, vb_time_i, central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
        
        if self.save_correlations:
            dfc_angles = utils.compute_angle(dfc_udpates, dfc_stdp_udpates)
            dfc_diff_hebbian_angles = utils.compute_angle(dfc_diff_hebbian_udpates, dfc_stdp_udpates)
        else:
            dfc_angles = None
            dfc_diff_hebbian_angles = None

        return dfc_stdp_udpates, dfc_udpates, dfc_diff_hebbian_udpates, dfc_angles, dfc_diff_hebbian_angles


    def save_dfc_diff_hebbian_angles(self, writer, step,
                        save_dataframe=True, save_tensorboard=True):
        """
        Compute the angle between the actual weight updates of the model
        (e.g. resulting from TPDI) on the one hand, and the weight updates
        resulting from the original DFC model on
        other hand.
        Save the angle in the tensorboard X writer
        (if ``save_tensorboard=true``) and in the
        corresponding dataframe (if ``save_dataframe=True``)
        Args:
            writer: Tensorboard writer
            step (int): x-axis index used for tensorboard

            save_dataframe (bool): Flag indicating whether a dataframe of the angles
                should be saved in the network object
            save_tensorboard (bool): Flag indicating whether the angles should
                be saved in Tensorboard
        """
        # self.step = step
        for i in range(len(self.layers)):
            if save_tensorboard:
                name = 'layer {}'.format(i+1)
                writer.add_scalar(
                    tag='{}/weight_dfc_diff_hebbian_angle'.format(name),
                    scalar_value=self.dfc_diff_hebbian_angles_list[i],
                    global_step=step)
            if save_dataframe:
                self.dfc_diff_hebbian_angles.at[step, i] = self.dfc_diff_hebbian_angles_list[i].item()
        if save_tensorboard:
            writer.add_scalar(
                tag='total_alignment/dfc_diff_hebbian_angles',
                scalar_value=self.dfc_diff_hebbian_total_angle,
                global_step=step)
        if save_dataframe:
            self.dfc_diff_hebbian_angles_network.at[step, 0] = self.dfc_diff_hebbian_total_angle.item()


    def compute_dfc_diff_hebbian_angles(self, i, vs_time_i, vb_time_i, central_activities_i, r_previous_time):
        dfc_diff_hebbian_udpates = self.layers[i].compute_forward_gradients_deltav_continuous_diff_hebbian(
                        vs_time_i, vb_time_i, central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
        dfc_stdp_udpates = self.layers[i].compute_forward_gradients_deltav_continuous_stdp(
                        vs_time_i, vb_time_i, central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
        
        weights_angle = utils.compute_angle(dfc_diff_hebbian_udpates, dfc_stdp_udpates)

        return dfc_diff_hebbian_udpates, dfc_stdp_udpates, weights_angle


    def compute_condition_two(self, retain_graph=False):
        r"""
        ..math::
            \frac{\|\tilde{J}_2\|_F}{\|\tilde{J}\|_F}
        to keep track whether condition 2 is (approximately) satisfied.
        If the minibatch size is bigger than 1, the mean over the minibatch
        is returned.
        Returns:
        """

        jacobians = self.compute_full_jacobian(linear=True,
                steady_state=self.at_steady_state, retain_graph=retain_graph,
                    r_targets=self.r_target, average_ss=self.average_ss)
        
        Q = self.full_Q
        projected_Q_fro = []
        for b in range(jacobians.shape[0]):
            jac = jacobians[b,:,:]
            projection_matrix = torch.matmul(jac.T,
                torch.matmul(torch.inverse(torch.matmul(jac, jac.T)), jac))
            projected_Q_fro.append(torch.norm(torch.matmul(projection_matrix, Q), p='fro'))

        projected_Q_fro = torch.stack(projected_Q_fro)

        Q_fro = torch.norm(Q, p='fro')

        condition_two_ratio = projected_Q_fro/Q_fro

        return torch.mean(condition_two_ratio)


    def get_feedback_parameter_list(self):
        """
        Returns (list): a list with all the feedback parameters (weights and
            biases) of the network. Note that the first hidden layer does not
            need feedback parameters, so they are not put in the list.
        """
        
        parameterlist = []
        for layer in self.layers:
            parameterlist.append(layer.feedbackweights)
        return parameterlist


    def save_feedback_batch_logs(self, args, writer, step, init=False,
                                 retain_graph=False, save_tensorboard=True,
                                 save_dataframe=True, save_statistics=False,
                                 damping=0):
        """
        Save the logs for the current minibatch on tensorboardX.
        Args:
            args (argsparse.Namespace): cmd line arguments
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
            retain_graph (bool): Flag indicating whether the Pytorch Autograd
                computational graph should be retained or not.
            save_statistics: Flag indicating whether the statistics of the
                feedback weights should be saved (e.g. gradient norms)
            damping (float): damping constant used for computing the damped
                pseudoinverse of the jacobian.
        """
        if args.save_jac_pinv_angle:
            self.save_jacobian_pinv_angle(writer, step, retain_graph=True,
                                    save_tensorboard=save_tensorboard,
                                    save_dataframe=save_dataframe,
                                    damping=damping, init=init)
        if args.save_jac_t_angle:
            self.save_jacobian_transpose_angle(writer, step, retain_graph=True,
                                        save_tensorboard=save_tensorboard,
                                        save_dataframe=save_dataframe, init=init)
        if args.save_condition_gn:
            condition_2 = self.compute_condition_two(retain_graph=retain_graph)
            if save_tensorboard:
                if init:
                    writer.add_scalar(tag='feedback_pretraining/condition_2',
                                    scalar_value=condition_2,
                                    global_step=step)
                else:
                    writer.add_scalar(tag='feedback_training/condition_2',
                                    scalar_value=condition_2,
                                    global_step=step)

            if save_dataframe:
                if init:
                    self.condition_gn_init.at[step, 0] = condition_2.item()
                else:
                    self.condition_gn.at[step, 0] = condition_2.item()

        if save_statistics:
            for i, layer in enumerate(self.layers):
                name = 'layer_' + str(i+1)
                layer.save_feedback_batch_logs(writer, step, name, init=init)

        if args.save_stability_condition:
            eig_min = self.compute_stability_condition()
            if save_tensorboard:
                if init:
                    name = 'feedback_pretraining/minimum_eigval_JQ'
                else:
                    name = 'feedback_training/minimum_eigval_JQ'
                writer.add_scalar(
                    tag=name,
                    scalar_value=eig_min,
                    global_step=step
                )


    def save_forward_batch_logs(self, args, writer, step, save_statistics=False,
                                save_tensorboard=True):
        """Save the forward weight logs for the current minibatch on tensorboardX.

        Args:
            args (argsparse.Namespace): cmd line arguments
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            save_statistics: Flag indicating whether the statistics of the
                feedback weights should be saved (e.g. gradient norms)
        """
        if save_tensorboard and save_statistics:
            for i, layer in enumerate(self.layers):
                name = 'layer_' + str(i+1)
                layer.save_forward_batch_logs(writer, step, name,
                                              no_gradient=i == 0)

    def to(self, device):
        """Overwrite `to` method so that also the feedback weights are moved.

        Args:
            device (str): The device where to send the weights.
        """
        super().to(device)
        for layer in self.layers:
            layer.feedbackweights = layer.feedbackweights.to(device)
            if layer.feedbackweights.grad is not None:
                layer.feedbackweights.grad = layer.feedbackweights.grad.to(device)

class GNNetwork(DFCNetwork):
    """
    Network implementing Gauss Newton updates (computed separately for
    each minibatch sample and afterwards averaged over the minibatch).
    Dummy methods whose purpose is to make sure the abstract methods of the superclass
    are overridden.
    """

    def set_layers(self, n_in, n_hidden, n_out, activation, output_activation,
                   bias, forward_requires_grad, initialization):
        n_all = [n_in] + n_hidden + [n_out]

        if isinstance(activation, str):
            activation = [activation] * len(n_hidden)

        layers = nn.ModuleList()
        for i in range(1, len(n_all) - 1):
            layers.append(GNLayer(n_all[i - 1], n_all[i],
                                  bias=bias,
                                  forward_requires_grad=forward_requires_grad,
                                  forward_activation=activation[i - 1],
                                  initialization=initialization,
                                  clip_grad_norm=self.clip_grad_norm,
                                  size_output=n_out))
        layers.append(GNLayer(n_all[-2], n_all[-1],
                              bias=bias,
                              forward_requires_grad=forward_requires_grad,
                              forward_activation=output_activation,
                              initialization=initialization,
                              clip_grad_norm=self.clip_grad_norm,
                              size_output=n_out))

        return layers


    def backward(self, loss, targets, target_lr, save_target=False,
                 writer=None, step=None):
        # compute the output error
        output_activation =  self.layers[-1].activations
        output_error = torch.autograd.grad(loss, output_activation,
                                           retain_graph=True)[0].detach()

        # Compute GN updates
        gn_parameter_updates = self.compute_gn_parameter_update(
            output_error=output_error,
            damping=self.alpha_di,
            retain_graph=True
        )
        for i in range(self.depth):
            if self.use_bias:
                gn_weight_update = gn_parameter_updates[2 * i]
                gn_bias_update = gn_parameter_updates[2 * i + 1]
                self.layers[i].compute_forward_gradients(gn_weight_update,
                                                         gn_bias_update)
            else:
                gn_weight_update = gn_parameter_updates[i]
                self.layers[i].compute_forward_gradients(gn_weight_update)


class DFC_single_phase_Network(DFCNetwork):
    """
    Network that always udpates the feedfoward and feedback
    weights simultaneously in one single phase.
    """
    
    def __init__(self, *args, pretrain_without_controller=False,
                 not_high_pass_filter_u_fb=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.pretrain_without_controller = pretrain_without_controller
        self.not_high_pass_filter_u_fb = not_high_pass_filter_u_fb
        if self.simulation_mode == 'blockwise':
            raise ValueError('Blockwise computation of controller is not implemented.')
    
    
    def backward(self, loss, targets, target_lr, save_target=True, writer=None,
                 step=None):
        """
        Run the feedback phase of the network, where the network is pushed
        to the output target by the controller. Compute the update of
        the forward weights.
        
        Args:
            loss (torch.Tensor): The output loss of the feedforward pass.
            target_lr (float): The stepsize that is used to compute the
                output target
            save_target (bool): Flag indicating whether the equilibrium
                activations of the network should be saved.
            writer: The tensorboard writer.
            step: The training step, for tensorboard.
        """
        output_target = targets

        v_feedforward = [l.linearactivations for l in self.layers]
        r_feedforward = [l.activations for l in self.layers]

        u, va, vb, vs, r_target = \
            self.dynamical_inversion(output_target, 
                                     alpha=self.alpha_di,
                                     dt=self.dt_di,
                                     tmax=self.tmax_di,
                                     epsilon=self.epsilon_di,
                                     makeplots=self.makeplots,
                                     savedir=self.out_dir,
                                     compare_with_ndi=self.compare_with_ndi,
                                     writer=writer, step=step)

        if self.noisy_dynamics and not self.not_low_pass_filter_r:
            # compute lowpass filter of r (average out the injected noise in ff phase)
            r_target_filtered = [torch.zeros_like(r) for r in r_target]
            # exponential smoothing
            for l in range(self.depth):
                r_target_filtered[l][0] = r_target[l][0].clone()
                for t in range(1,int(self.tmax_di)):
                    r_target_filtered[l][t] = self.alpha_r*r_target[l][t] + (1-self.alpha_r)*r_target_filtered[l][t-1]
        else:
            r_target_filtered = r_target

        # Add plot for debugging r filtering.
        if writer is not None:
            self.add_filter_to_writer(writer, step, r_target, r_target_filtered,
                                      name='r_target')

        # postprocess what I need to learn the feedforward weights
        r_target_ss = [r[-1] for r in r_target_filtered]
        u_ss = u[-1]
        va_ss = [v[-1] for v in va]
        vb_ss = [v[-1] for v in vb]
        vs_ss = [v[-1] for v in vs]
        delta_v_ss = [vs_ss[i] - vb_ss[i] for i in range(len(vs_ss))]

        if self.save_eigenvalues:
            A, max_eig, keys = self.compute_A_matrices(vs_ss, linear=True)
            self.max_eig = max_eig

        if self.save_norm_r:
            self.compute_norm_r(r_feedforward)
        
        if self.save_correlations:
            # compute internally the dfc angles
            dfc_updates_list =[torch.zeros_like(self.layers[i].get_forward_gradients()[0]) for i in range(self.depth)]
            current_updates_list = [torch.zeros_like(self.layers[i].get_forward_gradients()[0]) for i in range(self.depth)]
            dfc_diff_hebbian_updates_list = [torch.zeros_like(self.layers[i].get_forward_gradients()[0]) for i in range(self.depth)]

        for i in range(self.depth):
            delta_v_i = delta_v_ss[i]

            if i == 0:
                r_previous = self.input
            else:
                if self.use_initial_activations:
                    r_previous = r_feedforward[i - 1]
                else:
                    r_previous = r_target_ss[i - 1]

            # learn the feedforward weights
            if self.grad_deltav_cont:
                if self.efficient_controller:
                    raise ValueError('Argument "efficient_controller" must be False for dfc method.')
                    pass 
                else:
                    # if i == 0:
                    #     r_previous_time = self.input.unsqueeze(0).expand(
                    #         int(self.tmax_di), self.input.shape[0], self.input.shape[1])
                    # else:
                    #     r_previous_time = r_target_filtered[i - 1]
                    
                    central_activities_i = self.layers[i].forward_activationfunction(vs[i])
                    # test without filtering of pre activities (targets)
                    if i == 0:
                        r_previous_time = self.input.unsqueeze(0).expand(
                            int(self.tmax_di), self.input.shape[0], self.input.shape[1])
                    else:
                        r_previous_time = self.layers[i-1].forward_activationfunction(vs[i-1])
                    
                    weights_updates = self.layers[i].compute_forward_gradients_deltav_continuous(
                        vs[i], vb[i], central_activities_i, r_previous_time,
                        learning_rule=self.learning_rule,
                        use_diff_hebbian_updates=self.use_diff_hebbian_updates,
                        use_stdp_updates=self.use_stdp_updates,
                        save_stdp_measures=self.save_stdp_measures,
                        save_correlations=self.save_correlations,
                        decay_rate=self.decay_rate,
                        stdp_samples=self.stdp_samples)
        
            else:
                weights_updates = self.layers[i].compute_forward_gradients(delta_v_i, r_previous,
                                                                learning_rule=self.learning_rule)

            self.layers[i].target = vs_ss[i]
            self.layers[i].linearactivations_ss = vs_ss[i]
            self.layers[i].activations_ss = r_target_ss[i]

            # # As we are standard using noisy dynamics, the last value of u will be noisy, and we should
            # # average over u to cancel out the noise. I assume that in the last quarter of the simulation,
            # # u has converged, so we can average over that interval.
            # interval_length=int(self.tmax_di/4)
            # self.u = torch.sum(u[-interval_length:-1,:,:], dim=0)/float(interval_length)

            if self.save_NDI_updates:
                u_ndi, v_targets_ndi, r_targets_ndi, target_ndi, delta_v_ndi = \
                    self.non_dynamical_inversion(output_target, self.alpha_di)
                delta_v_i_ndi = delta_v_ndi[i]
                self.layers[i].compute_forward_gradients(delta_v_i_ndi, r_previous,
                                                                learning_rule=self.learning_rule,
                                                                saving_ndi_updates=True)

            if not self.freeze_fb_weights and not self.use_jacobian_as_fb: 
                # if we use the ideal feedback weights, we don't need to do these updates.
                # compute highpass filter of u
                u = u[1:,:,:]

            if not self.not_high_pass_filter_u_fb:
                u_aux = torch.zeros_like(u)
                u_aux[0] = u[0]
                # exponential smoothing
                for t in range(1, len(u)):
                    # in the first few steps, u is changing very fast due to the proportional controller, hence
                    # it is best to let the low-passed u move directly with u in the beginning, to avoid transients.
                    if self.stability_tricks and t < len(u)/20:
                        u_aux[t] = u[t]
                    else:
                        u_aux[t] = self.alpha_u*u[t] + (1-self.alpha_u)*u_aux[t-1]

                # subtract the exponential smoothing (high pass filter: average out the target nudging phase)
                u_filtered = u-u_aux
            else:
                u_filtered = u

            # Add plot for debugging u filtering.
            if writer is not None:
                self.add_filter_to_writer(writer, step, u, u_filtered, name='u')

            if not self.freeze_fb_weights:
                # learn the feedback weights
                self.compute_feedback_gradients(loss, targets, self.target_stepsize,
                                            va=va, u_filtered=u_filtered)
        
            if self.grad_deltav_cont and self.save_stdp_measures and (self.epoch==self.save_epoch or (self.save_epoch==None and self.epoch==self.num_epochs-1)):
                # save the feedforward activities (nonlinear activations) 
                self.pre_activities[i] = r_previous_time
                # save difference between central and basal compartments (linear activations)
                self.delta_post_activities[i] = vs[i] - vs[i]
                # save weights values 
                self.weights[i] = self.layers[i].weights.detach().cpu()
                # save stdp weights updates values 
                self.weights_updates[i] = weights_updates.detach().cpu()
                # save central compartment activties (after nonlinearity is applied)
                self.central_activities[i] = self.layers[i].forward_activationfunction(vs[i])

            # compute layerwise angles for dfc upfdates
            out_dfc = self.compute_dfc_angles(i, vs[i], vb[i], central_activities_i, r_previous_time)
            dfc_stdp_gradients = out_dfc[0]
            dfc_gradients = out_dfc[1]
            dfc_diff_hebbian_gradients = out_dfc[2]
            dfc_angles = out_dfc[3]
            dfc_diff_hebbian_angles = out_dfc[4]

            if self.save_stdp_measures and self.epoch==self.num_epochs-1:
                self.weights_updates_bp[i] = self.layers[i].compute_bp_update(loss, retain_graph=True)[0]
                self.weights_updates_stdp[i] = dfc_stdp_gradients.detach()
                self.weights_updates_diff_hebbian[i] = dfc_diff_hebbian_gradients.detach()
                self.weights_updates_dfc[i] = dfc_gradients.detach()
            
            if self.save_correlations:
                current_updates_list[i] = dfc_stdp_gradients.detach()
                dfc_updates_list[i] = dfc_gradients.detach()
                dfc_diff_hebbian_updates_list[i] = dfc_diff_hebbian_gradients.detach()
                self.dfc_angles_list[i] = dfc_angles
                self.dfc_diff_hebbian_angles_list[i] = dfc_diff_hebbian_angles

        if self.save_correlations:
            # compute total angles for dfc udpates    
            current_updates_concat = utils.vectorize_tensor_list(current_updates_list)
            dfc_updates_concat = utils.vectorize_tensor_list(dfc_updates_list)
            dfc_total_angle = utils.compute_angle(current_updates_concat, dfc_updates_concat)
            self.dfc_total_angle = dfc_total_angle
            # compute total angles for dfc differential hebbian angles
            dfc_diff_hebbian_updates_concat = utils.vectorize_tensor_list(dfc_diff_hebbian_updates_list)
            dfc_diff_hebbian_total_angle = utils.compute_angle(current_updates_concat, dfc_diff_hebbian_updates_concat)
            self.dfc_diff_hebbian_total_angle = dfc_diff_hebbian_total_angle
        
        if self.grad_deltav_cont and self.save_correlations and self.epoch==self.num_epochs-1:
            # save bp agles
            self.correlations_bp = self.bp_angles_network.at[self.step, 0]
            # save dfc agles
            self.correlations_dfc = self.dfc_angles_network.at[self.step, 0]
            # save dfc_diff_hebbian agles
            self.correlations_dfc_diff_hebbian = self.dfc_diff_hebbian_angles_network.at[self.step, 0]
        
        self.makeplots = False 
        

    @torch.no_grad()    
    def dynamical_inversion(self, output_target, alpha=0.001, dt=0.3, tmax=100,
                            epsilon=0.5, makeplots=False,
                            compare_with_ndi=True, savedir=r'.\logs',
                            writer=None, step=None):
        """
        Performs DFC in real time, that is, controlling all hidden layers simultaneously.
        """
        
        tmax = np.round(tmax).astype(int)
        batch_size = self.layers[0].activations.shape[0]
        plot_title = '' 
        error_str = None
        error = False

        r_feedforward = [l.activations for l in self.layers]
        v_feedforward = [l.linearactivations for l in self.layers]

        r_target, u, (va, vb, vs), sample_error = \
            self.controller(output_target, alpha, dt, tmax,
                            mode=self.simulation_mode,
                            inst_system_dynamics=self.inst_system_dynamics,
                            k_p=self.k_p,
                            noisy_dynamics=self.noisy_dynamics,
                            inst_transmission=self.inst_transmission,
                            time_constant_ratio=self.time_constant_ratio,
                            proactive_controller=self.proactive_controller,
                            sigma=self.sigma,
                            sigma_output=self.sigma_output,
                            writer=writer, step=step)

        if makeplots:
            plot_title = plot_title + '_Epoch{epoch}_'.format(epoch=self.epoch)
            if error: print(error_str)

            if compare_with_ndi:
                u_ndi, v_targets_ndi, r_targets_ndi, target_ndi, delta_v_ndi = \
                    self.non_dynamical_inversion(output_target, alpha)
                self.create_plots_w_ndi(r_target, va, vb, vs, r_feedforward, r_target[-1], output_target,
                                        u_ndi, delta_v_ndi, savedir=savedir, title=plot_title)
            else:
                self.create_plots_di(r_target, va, vb, vs, r_feedforward, r_target[-1], output_target,
                                        savedir=savedir, title=plot_title)

        converged, diverged = self.check_convergence(r_target, r_feedforward, output_target,
                                                            u, sample_error, epsilon, batch_size)

        if self.save_eigenvalues_bcn:
            vs_bcn = [v[-1] for v in vs]
            A, max_eig, keys = self.compute_A_matrices(vs_bcn, linear=True)
            self.max_eig_bcn = max_eig

        if not self.include_non_converged_samples:
            indices = converged == 0
            indices = utils.bool_to_indices(indices)

            for i in range(self.depth):
                vs[i][:, indices, :] = v_feedforward[i][indices, :]
                vb[i][:, indices, :] = v_feedforward[i][indices, :]
                va[i][:, indices, :] = 0.
                r_target[i][:, indices, :] = r_feedforward[i][indices]
            u[:, indices, :] = 0.

        return u, va, vb, vs, r_target
    

    def controller(self, output_target, alpha, dt, tmax, mode='blockwise',
                   inst_system_dynamics=False, k_p=0., noisy_dynamics=False,
                   inst_transmission=False, time_constant_ratio=1.,
                   proactive_controller=False,
                   sigma=0.01, sigma_output=0.01, writer=None, step=None):
        r"""
        Simulate the feedback control loop for tmax timesteps. The following
        continuous time ODEs are simulated
        with time interval ``dt``:

        ..math::
            \frac{\tau_v}{\tau_u}\frac{d v_i(t)}{dt} = \
                -v_i(t) + W_i r_{i-1}(t) + b_i + Q_i u(t) \\
            \frac{d u(t)}{dt} = e(t) + k_p\frac{d e(t)}{dt} - \alpha u(t) \\
            e(t) = r_L^* - r_L(t)

        Note that we use a ratio :math:`\frac{\tau_v}{\tau_u}` instead of two
        separate time constants for :math:`v` and :math`u`, as a scaling of
        both timeconstants can be absorbed in the simulation timestep ``dt``.
        IMPORTANT: ``time_constant_ratio`` should never be taken smaller than
        ``dt``, as the the forward Euler method will become unstable by
        default (the simulation steps will start to 'overshoot').

        If ``inst_transmission=False``, the forward Euler method is used to
        simulate the differential equation. If ``inst_transmission=True``, a
        slight modification is made to the forward Euler method, assuming that
        we have instant transmission from one layer to the next: the basal
        voltage of layer i at timestep ``t`` will already be based on the
        forward propagation of the somatic voltage of layer i-1 at timestep ``t``,
        hence including the feedback of layer i-1 at timestep ``t``.
        It is recommended to put ``inst_transmission=True`` when the
        ``time_constant_ratio`` is approaching ``dt``, as then we are
        approaching the limit of instantaneous system dynamics in the simulation
        where inst_transmission is always used (See below).

        If ``inst_system_dynamics=True``, we assume that the time constant of the
        system (i.e. the network) is much smaller than that of the controller
        and we approximate this by replacing the dymical equations for v_i by
        their instantaneous equivalents:

        ..math::
            v_i(t) = W_i r_{i-1}(t) + b_i + Q_i u(t)

        Note that ``inst_transmission`` will always be put on True (overridden)
        in combination with inst_system_dynamics.

        If ``proactive_controller=True``, the control input u[k+1] will be used
        to compute the apical voltages v^A[k+1], instead of the control
        input u[k]. This is a slight variation on the forward Euler method and
        and corresponds to the conventional discretized control schemes.

        If ``noisy_dynamics=True``, noise is added to the apical compartment of
        the neurons. We now simulate the apical compartment with its own dynamics,
        as the ``normal_controller`` feedback learning rule needs access to the
        noisy apical compartment. We use the following stochastic differential
        equation for the apical compartment:
        ..math::
            \tau_A d v_i^A = (-v_i^A + Q_i u)dt + \sigma dW
        with W the Wiener process (Brownian motion) with covariance matrix I.
        This is simulated with the Euler-Maruyama method:
        ..math::
            v_i^A[k+1] = v_i^A[k] + \Delta t / \tau_A (-v_i^A[k] + Q_i u[k]) + \
                \sigma / sqrt(\Delta t / \tau_A) \Delta W

        with :math:`\Delta W` drawn from the zero-mean Gaussian distribution with
        covariance I. The other dynamical
        equations in the system remain the same, except that :math:`Q_i u` is
        replaced by :math:`v_i^A`

        ..math::
            \tau_v \frac{d v_i(t)}{dt} = -v_i(t) + W_i r_{i-1}(t) + b_i + v_i^A

        One can opt for instantaneous apical compartment dynamics by putting
        its timeconstant :math:`tau_A` (``apical_time_constant``) equal to
        ``dt``. This is not encouraged for training the feedback weights with
        ``normal_controller`` fb learning rule, but can be used for simulating
        noisy system dynamics for training the forward weights, resulting in:
        ..math::
            \tau_v d v_i(t)} = (-v_i(t) + W_i r_{i-1}(t) + b_i + Q_i u(t) )dt +\
                \sigma dW

        which can again be similarly discretized with the Euler-Maruyama method.

        Note that for training the feedback weights with the ``normal_controller``
        learning rule, it is recommended to put ``inst_transmission=True``, such
        that the noise of all layers can influence the output at the current
        timestep, instead of having to wait for a couple of timesteps, depending
        on the layer depth.

        Note that in
        the current implementation, we interpret that the noise is added in
        the apical compartment, and that the basal and somatic compartments
        are not noisy. At some point we might want to also add noise in the
        somatic and basal compartments for physical realism.

        Args:
            output_target (torch.Tensor): The output target :math:`r_L^*` that is used by
                the controller to compute the control error :math:`e(t)`.
            alpha (float): The leakage term of the controller
            dt (float): the time interval used in the forward Euler method
            tmax (int): the maximum number of timesteps
            mode ['blockwise', 'layerwise']: String indicating whether the
                dynamics should be simulated in a layerwise manner where we
                loop through the layers, or in a blockwise manner where we use
                block matrices and concatenated vectors to compute the updates
                in one go. Both options produce the same results, it's only a
                matter of efficiency. Use 'blockwise' preferably. If
                ``inst_system_dynamics`` and/or ``inst_transmission` is used,
                ``mode=layerwise`` will be used (overwritten if necessary).
            inst_system_dynamics (bool): Flag indicating whether we should
                replace the system dynamics by their instantaneous counterpart.
                If True, `inst_transmission`` will be overwritten to ``True``.
            k_p (float): The positive gain parameter for the proportional part
                of the controller. If it is equal to zero (which is the default),
                no proportional control will be used, only integral control.
            noisy_dynamics (bool): Flag indicating whether noise should be
                added to the dynamcis.
            inst_transmission (bool): Flag indicating whether the modified
                version of the forward Euler method should be used, where it is
                assumed that there is instant transmission between layers (but
                not necessarily instant voltage dynamics). See the docstring
                above for more information.
            time_constant_ratio (float): ratio of the time constant of the
                voltage dynamics w.r.t. the controller dynamics.
            writer: The tensorboard writer.
            step: The training step.

        Returns (tuple): Ordered tuple containing
            r_target (list): A list with at index ``i`` a torch.Tensor of
                dimension :math:`t_{max}\times B \times n_i` containing the
                firing rates of layer i for each timestep.
            u (torch.Tensor): A tensor of dimension
                :math:`t_{max}\times B \times n_L` containing the control input
                for each timestep
            (v_a, v_b, v_s) (tuple): A tuple with 3 elements, each containing
                a list with at index ``i`` a torch.Tensor of
                dimension :math:`t_{max}\times B \times n_i` containing the
                voltage levels of the apical, basal or somatic compartments
                respectively.
            sample_error (torch.Tensor): A tensor of dimension
                :math:`t_{max} \times B` containing the L2 norm of the error
                e(t) at each timestep.
        """

        if k_p < 0:
            raise ValueError('Only positive values for k_p are allowed')

        if inst_system_dynamics:
            inst_transmission = True

        if inst_transmission:
            mode = 'layerwise'

        batch_size = output_target.shape[0]
        tmax = int(tmax)

        L = len(self.layers) 
        layer_input_dims = [l.weights.shape[1] for l in self.layers]
        layer_output_dims = [l.weights.shape[0] for l in self.layers]
        
        # If hidden activations are linear, then J doen't depend on the samples
        if self.use_jacobian_as_fb and self.activation == 'linear':
            # J = self.compute_full_jacobian(linear=True)
            J = self.compute_full_jacobian(linear=True, steady_state=self.at_steady_state, retain_graph=False, r_targets=self.r_target, average_ss=self.average_ss)

        if mode == 'blockwise':
            raise NotImplementedError("blockwise controller is not implemented anymore for single-phase DFC")
        
        elif mode == 'layerwise':
            size_output = output_target.shape[1]
            u = torch.zeros((tmax, batch_size, size_output))
            if self.low_pass_filter_u:
                u_lp = torch.zeros_like(u)
            va = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # apical voltage = Ki u
            vb = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # basal voltage = Wi h_target_i-1
            vs = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # somatic voltage
            noise_filtered = [torch.zeros((batch_size, l)) for l in layer_output_dims] # exponentially filtered white noise
            if writer is not None:
                noise_pos_filtered = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims] # exponentially filtered white noise
                noise_pre_filtered = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims] # white noise
            if k_p > 0:
                u_int = torch.zeros((tmax, batch_size, size_output))
            r_target = [torch.zeros((tmax, batch_size, l)) for l in layer_output_dims]  # just v_soma after non-linearity
            sample_error = torch.ones((tmax, batch_size)) * 10
            
            # # reset activations in every layer to dummy feedforward activations 
            # if self.use_diff_hebbian_updates or self.use_stdp_updates:
            #     y = self.forward(self.input, dummy_reset=True)
            
            for i in range(L):
                va[i][0, :] = self.layers[i].linearactivations
                vb[i][0, :] = self.layers[i].linearactivations
                vs[i][0, :] = self.layers[i].linearactivations
                r_target[i][0, :] = self.layers[i].activations
            sample_error[0] = self.compute_loss(output_target, r_target[-1][0, :])

            for t in range(tmax - 1):

                e = self.compute_error(output_target, r_target[-1][t])

                # save time to convergence measures
                if self.save_stdp_measures:
                    # self.error_time_to_convergence[self.epoch,t] += torch.abs(torch.mean(e)*torch.pow(torch.tensor(10), torch.tensor(16)))
                    self.error_time_to_convergence[self.epoch,t] += torch.abs(torch.mean(e))

                # If hidden activations are nonlinear, then J does depend on the
                # samples (derivative of their activations).
                if self.use_jacobian_as_fb and self.activation != 'linear':
                    # For efficiency, we only compute J every few timesteps.
                    # The smaller dt, the less often we want to log.
                    log_freq = round(1 / (dt * 100))
                    if log_freq < 1:
                        log_freq = 1
                    if t == 0 or t % log_freq == 0:
                        # J = self.compute_full_jacobian(linear=True)
                        J = self.compute_full_jacobian(linear=True, steady_state=self.at_steady_state, retain_graph=False, r_targets=r_target, average_ss=self.average_ss)

                if k_p > 0.: 
                    u_int[t + 1] = u_int[t] + dt * (e - alpha * u[t])
                    u[t + 1] = u_int[t + 1] + k_p * e
                else:
                    u[t + 1] = u[t] + dt * (e - alpha * u[t])

                if self.low_pass_filter_u:
                    # We need to keep track both of the unfiltered u and the low-pass filtered u,
                    # as we need the high-frequency parts of u for the feedback weight updates
                    if t<100:
                    # if t == 0:
                        # start the low-pass filtering at the same value of u,
                        # as otherwise it takes a long time to recover from zero
                        # For the first couple of timesteps, the proportional control
                        # is changing very fast, so let u_lp keep track like this
                        u_lp[t+1] = u[t+1]
                    else:
                        u_lp[t + 1] = self.alpha_u*u[t + 1] + (1-self.alpha_u)*u_lp[t]

                def layer_iteration(i):
                    if i == 0:
                        r_previous = self.input
                    else:
                        if inst_transmission:
                            r_previous = r_target[i - 1][t + 1]
                        else:
                            r_previous = r_target[i - 1][t]

                    a = r_previous.mm(self.layers[i].weights.t())
                    if self.layers[i].bias is not None:
                        a += self.layers[i].bias.unsqueeze(0).expand_as(a)
                    vb[i][t + 1, :] = a

                    def get_control_signal(t, u_aux):
                        """Get the control signal Qu for the given timestep.

                        By default, this computes :math:`Qu` but in case the option
                        `use_jacobian_as_fb``is active, this computes :math:`Ju`.

                        Args:
                            t (int): The timestep.
                            u_aux (torch.Tensor): The control u to use. Can be
                                low-pass filtered or not, depending on
                                `low_pass_filter_u`.

                        Returns:
                            (torch.Tensor): The control signal.
                        """
                        
                        if self.use_jacobian_as_fb:
                            batch_size = u_aux.shape[1]
                            n_out = u_aux.shape[2]

                            # Select the correct Jacobian block.
                            J_sq = J.view(batch_size * n_out, J.shape[-1])
                            Ji = utils.split_in_layers(self, J_sq)[i]
                            Ji = Ji.view(batch_size, n_out, Ji.shape[-1])

                            return torch.matmul(u_aux[t].unsqueeze(1), Ji).squeeze()
                        else:
                            return torch.mm(u_aux[t], \
                                            self.layers[i].feedbackweights.t())

                    # Get the control signal.
                    control_signal = get_control_signal(\
                                        t + 1 if proactive_controller else t,
                                        u_lp if self.low_pass_filter_u else u)
                    va[i][t + 1, :] = control_signal

                    if self.use_stdp_updates or self.use_diff_hebbian_updates:
                        pass
                    
                    elif noisy_dynamics:
                        if i == self.depth-1:
                            sigma_copy = sigma_output
                        else:
                            sigma_copy = sigma
                        
                        new_noise = torch.randn_like(va[i][t+1,:])
                        # Warning: for very small dt, we might need to change the implementation for
                        # numerical stability and work with tau_noise*sqrt(dt) instead of
                        # alpha_noise/sqrt(dt).
                        alpha_noise = self.alpha_noise # TODO should be dt / self.tau_noise
                        noise_filtered[i] = (alpha_noise/np.sqrt(dt)) * new_noise + \
                                            (1-alpha_noise) * noise_filtered[i]

                        # Store values for plotting.
                        if writer is not None:
                            noise_pre_filtered[i][t+1, :, :] = new_noise
                            noise_pos_filtered[i][t+1, :, :] = noise_filtered[i]

                        va[i][t + 1, :] += sigma_copy * noise_filtered[i]

                    if inst_system_dynamics: 
                        vs[i][t + 1, :] = va[i][t + 1, :] + vb[i][t + 1, :]
                    else: 
                        vs[i][t + 1, :] = vs[i][t, :] + dt/time_constant_ratio \
                                          * (va[i][t + 1, :] + vb[i][t + 1, :] -
                                             vs[i][t, :])

                    r_target[i][t + 1, :] = self.layers[i].forward_activationfunction(vs[i][t + 1, :])
                    # save r_tartgets for computation of condition 2
                    self.r_target = r_target

                    # We need to store these to compute correctly the Jacobian.
                    if self.use_jacobian_as_fb:
                        self.layers[i].linearactivations = vs[i][t + 1, :]
                        self.layers[i].activations = r_target[i][t + 1, :]

                if not inst_transmission:
                    for i in range(L - 1, 0 - 1, -1):
                        layer_iteration(i)

                else:
                    for i in range(L):
                        layer_iteration(i)

                sample_error[t + 1] = self.compute_loss(output_target, r_target[-1][t + 1, :])
            
            # As we are standard using noisy dynamics, the last value of u will be noisy, and we should
            # average over u to cancel out the noise. I assume that in the last quarter of the simulation,
            # u has converged, so we can average over that interval.
            interval_length=int(tmax/4)
            if self.low_pass_filter_u:
                u = u_lp
            self.u = torch.sum(u[-interval_length:-1,:,:], dim=0)/float(interval_length)
            
            # Add plot for debugging noise filtering.
            if writer is not None and noisy_dynamics:
                self.add_filter_to_writer(writer, step, noise_pre_filtered,
                                          noise_pos_filtered, name='noise')

        else:
            raise NotImplementedError("Mode {} not recognized for the controller".format(mode))

        return r_target, u, (va, vb, vs), sample_error


    def add_filter_to_writer(self, writer, step, original, filtered, sample=0,
                             neuron_idx=0, name=''):
        """Function to add filtered signals to the writer.

        Args:
            writer: The tensorboard writer.
            step: The training step.
            original (list or torch.Tensor): The original signal.
            filtered (list or torch.Tensor): The filtered signal.
            sample (int): The index of the element in the batch to plot.
            neuron_idx (int): The index of the neuron in each layer to plot.
            name (str): The name of the signal.
        """
        if not isinstance(original, list):
            assert not isinstance(filtered, list)
            original = [original]
            filtered = [filtered]

        for l in range(len(original)):
            img = plt.figure()
            plt.plot()
            plt.plot(original[l][:, sample, neuron_idx].\
                    cpu().detach().numpy(), label=name, color='r', alpha=0.5)
            plt.plot(filtered[l][:, sample, neuron_idx].\
                    cpu().detach().numpy(), label='%s filtered' % name,
                    color='b', alpha=0.5)
            plt.xlabel('time')
            plt.ylabel(name)
            plt.legend()
            if len(original) == 1:
                tag='%s_filtering' % name
            else:
                tag='%s_filtering/layer_%i' % (name, l)
            writer.add_figure(tag=tag, figure=img, global_step=step)


    def compute_feedback_gradients(self, loss, targets, target_stepsize,
                                   va=None, u_filtered=None, init=False):
        r"""
        Compute the gradients of the feedback weights for each layer.
        This function is called in two different situations:
        1. During pre-training of the feedback weights, there has not yet been a simulation,
        so this function calls a simulation (with special values for alpha and k_p to
        ensure stability during pre-training) and uses the results of the simulation to
        update the feedback weights. In this case, the inputs ``va`` and ``u_filtered`` will
        be ``None``.
        2. During the simultaneous training of feedforward and feedback weights, the backward
        method already simulates the dynamics, and the results are passed through ``va`` and
        ``u_filtered``. In this case, we directly use these simulation results to compute the
        updates without running a new simulation.

        The feedback weight updates are computed according to the following rule:
        ..math::
            \Delta Q = -\frac{(1+\tau_v/\tau_{\epsilon})^{L-i}}{K*\sigma^2} \
                \sum_k \mathbf{v}^{fb}_i[k] \tilde{\mathbf{u}}^T[k] \

        Args:
            loss: output loss of the network
            targets: output target of the network
            va: a list with at index ``i`` a torch.Tensor of
                dimension :math:`t_{max}\times B \times n_i` containing the
                voltage levels of the apical (feedback) compartment of layer i.
                If None (by default), a new simulation will be run to calculate
                va and u_filtered
            u_filtered: a torch.Tensor of dimension t_{max}\times B \times n_L
                containing the high-pass filtered controller inputs. If None
                (by default), a new simulation will be run to calculate
                va and u_filtered
            init: if True, indicates that this is a pre-train of the weights
        """
        output_target = targets
        if init and self.pretrain_without_controller:
            output_target = self.layers[-1].activations.data

        # Determine whether only the backward weights are being trained. If the
        # rates va are provided, it means we are in the single-phase and the
        # forward weights gradients have just been computed.
        backward_only = True
        if va is not None:
            backward_only = False

        # Suffix to select the right simulation parameter depending on whether
        # we are in the common ff and fb training phase or not.
        sf = ''
        sigma_output = self.sigma
        if backward_only:
            sf = '_fb'
            sigma_output = self.sigma_output_fb

        if backward_only:
            # When only the feedback weights are being trained, we can set all
            # the simulation hyperparameters to their backward version.
            assert u_filtered == None
            if init:
                r_target, u, (va, vb, vs), sample_error = \
                    self.controller(output_target=output_target,
                                    alpha=self.alpha_fb,
                                    dt=self.dt_di_fb,
                                    tmax=self.tmax_di_fb,
                                    inst_system_dynamics=self.inst_system_dynamics,
                                    k_p=self.k_p_fb,
                                    noisy_dynamics=True,
                                    inst_transmission=self.inst_transmission_fb,
                                    time_constant_ratio=self.time_constant_ratio_fb,
                                    proactive_controller=self.proactive_controller,
                                    mode=self.simulation_mode,
                                    sigma=self.sigma_fb,
                                    sigma_output=self.sigma_output_fb)
            else:
                r_target, u, (va, vb, vs), sample_error = \
                    self.controller(output_target=output_target, 
                                    alpha=self.alpha_di,
                                    dt=self.dt_di,
                                    tmax=self.tmax_di,
                                    inst_system_dynamics=self.inst_system_dynamics,
                                    k_p=self.k_p,
                                    noisy_dynamics=True,
                                    inst_transmission=self.inst_transmission,
                                    time_constant_ratio=self.time_constant_ratio,
                                    proactive_controller=self.proactive_controller,
                                    mode=self.simulation_mode,
                                    sigma=self.sigma,
                                    sigma_output=self.sigma_output)

            u = u[1:, :, :]

            if not self.not_high_pass_filter_u_fb:
                u_aux = torch.zeros_like(u)
                u_aux[0] = u[0]
                # exponential smoothing
                for t in range(1, len(u)):
                    u_aux[t] = self.alpha_u * u[t] + (1 - self.alpha_u) * u_aux[t-1]
                # subtract the exponential smoothing (high pass filter: average out
                # the target clamping)
                u_filtered = u - u_aux
            else:
                u_filtered = u

        # Extract some important parameters that need to be used later.
        sigma = getattr(self, 'sigma' + sf)
        dt_di = getattr(self, 'dt_di' + sf)
        time_constant_ratio = getattr(self, 'time_constant_ratio' + sf)
        
        
        for i, layer in enumerate(self.layers):
            va_i = va[i][:-1, :, :]

            # compute a layerwise scaling_fb_updates for the feedback weights
            if self.scaling_fb_updates:
                # scale the update for each layer according to the theory
                scaling = (1 + time_constant_ratio * (self.alpha_noise / dt_di)) \
                        ** (len(self.layers) - i - 1)
            else:
                scaling = 1.
            
            # get the amount of noise used
            sigma_i = sigma
            if i == len(self.layers) - 1:
                sigma_i = sigma_output

            layer.compute_feedback_gradients_continuous(va_i, u_filtered,
                        sigma=sigma_i, scaling=scaling)
