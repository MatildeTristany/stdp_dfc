from networks.abstract_layer import AbstractLayer
import torch


class DFCLayer(AbstractLayer):

    def __init__(self, in_features, out_features, size_output, bias=True,
                 forward_requires_grad=False, forward_activation='tanh',
                 initialization='orthogonal', clip_grad_norm=-1):

        super().__init__(in_features=in_features,
                         out_features=out_features,
                         bias=bias,
                         forward_requires_grad=forward_requires_grad,
                         forward_activation=forward_activation,
                         initialization=initialization)

        self._clip_grad_norm = clip_grad_norm
        self._feedback_weights = torch.zeros(out_features, size_output)

    @property
    def feedbackweights(self):
        """Getter for read-only attribute :attr:`inversion_matrix` (ie K)"""
        return self._feedback_weights

    @property
    def clip_grad_norm(self):
        """Getter for read-only attribute :attr:`clip_grad_norm`"""
        return self._clip_grad_norm

    @feedbackweights.setter
    def feedbackweights(self, tensor):
        """ Setter for feedbackweights"""
        self._feedback_weights = tensor


    def compute_forward_gradients(self, delta_v, r_previous,
                                         learning_rule="voltage_difference",
                                         scale=1., saving_ndi_updates=False):
        """
        Computes gradients using a local-in-time learning rule.
        There are three possible learning rules to be applied:
        * voltage_difference : difference between basal and somatic voltage
        * derivative_matrix : the derivative matrix (of somatic voltage wrt
        forward weights) is added to the previous rule
        * nonlinear_difference: by Taylor expansion, the difference between the
        nonlinear transformation of basal and
        somatic voltages is approximately equal to the addition of the
        derivative matrix. Indeed, second and third
        learning rules have been found to be completely equivalent in practice.

        New parameter added: "saving_ndi_updates". When False, computed updates
        are added to weights.grad and bias.grad, to be later updated. When True,
        computed updates are added to ndi_updates_weights/bias, to later compare
        with the ss/continuous updates.
        """

        if learning_rule == "voltage_difference":
            teaching_signal = 2 * (-delta_v)

        elif learning_rule == "derivative_matrix":
            vectorized_jacobians = self.compute_vectorized_jacobian(self.linearactivations)
            teaching_signal = -2 * vectorized_jacobians * delta_v

        elif learning_rule == "nonlinear_difference":
            v_basal = torch.matmul(r_previous, self.weights.t())
            if self.bias is not None:
                v_basal += self.bias.unsqueeze(0).expand_as(v_basal)
            v_somatic = delta_v + v_basal 
            teaching_signal = -2*(self.forward_activationfunction(v_somatic) - self.forward_activationfunction(v_basal))

        else:
            raise ValueError("The specified learning rule %s is not valid" % learning_rule)

        batch_size = r_previous.shape[0]
        bias_grad = teaching_signal.mean(0)

        weights_grad = 1./batch_size * teaching_signal.t().mm(r_previous)

        if saving_ndi_updates:

            if self.bias is not None:
                self.ndi_updates_bias = scale * bias_grad.detach()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.ndi_updates_bias, max_norm=self.clip_grad_norm)
            self.ndi_updates_weights = scale * weights_grad.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.ndi_updates_weights, max_norm=self.clip_grad_norm)

        else:

            if self.bias is not None:
                self._bias.grad += scale * bias_grad.detach()
                if self.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self._bias, max_norm=self.clip_grad_norm)
            self._weights.grad += scale * weights_grad.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._weights, max_norm=self.clip_grad_norm)


    def poisson_spikes(self,firing_rates, timesteps=1):
        # convert firing rates to spikes
        expanded_rates = torch.repeat_interleave(firing_rates, repeats=timesteps, dim=2)
        spikes = 1*(torch.rand(expanded_rates.shape) < expanded_rates)
        return spikes
    

    def filter_spike_train(self, spikes, decay_rate=0.8):
        # apply filtering 
        aux = 0
        filtered_spikes = torch.zeros(spikes.shape)
        for t in range(spikes.shape[0]):
            aux = decay_rate*aux + (1-decay_rate)*spikes[t]
            filtered_spikes[t] = aux
        return filtered_spikes
    
    
    def compute_forward_gradients_deltav_continuous(self, vs_time, vb_time, central_activities,
                                                    r_previous_time, t_start=None, t_end=None,
                                                    learning_rule="voltage_difference", use_diff_hebbian_updates=False,
                                                    use_stdp_updates=False, save_stdp_measures=False,
                                                    save_correlations=False,
                                                    decay_rate=0.8, stdp_samples=1):
        """
        Computes gradients using an integration (sum) of voltage differences across comparments.
        It is a local-in-time learning rule. There are three possible learning rules to be applied:
        * voltage_difference : difference between basal and somatic voltage
        * derivative_matrix : the derivative matrix (of somatic voltage wrt forward weights) is added to the previous rule
        * nonlinear_difference: by Taylor expansion, the difference between the nonlinear transformation of basal and
        somatic voltages is approximately equal to the addition of the derivative matrix. Indeed, second and third
        learning rules have been found to be completely equivalent in practice.
        """

        if t_start is None: t_start = 0
        if t_end is None: t_end = central_activities.shape[0]
        T = t_end - t_start
        batch_size = central_activities.shape[1]
        
        if use_diff_hebbian_updates:

            if learning_rule == "nonlinear_difference":
                central_activities_t0 = central_activities[t_start:t_end-1]
                central_activities_t1 = central_activities[t_start+1:t_end]
                time_diff = central_activities_t1 - central_activities_t0
                time_diff = time_diff.permute(0,2,1)
                bias_grad = -2 * 1./T * torch.sum(time_diff, axis=0).mean(1)
                weights_grad = -2 * 1./batch_size * 1./T * torch.sum(time_diff @ r_previous_time[t_start:t_end-1, :, :], axis=0)
            else:
                raise ValueError("The specified learning rule %s is not valid or proper for this learning rule." % learning_rule)
            
        elif use_stdp_updates:

            if learning_rule =="nonlinear_difference":

                tmax = central_activities.shape[0]
                batch_size = central_activities.shape[1]
                num_neurons_pre = r_previous_time.shape[2]
                num_neurons_post = central_activities.shape[2]

                central_activities_spikes = torch.zeros(tmax, batch_size, num_neurons_post, stdp_samples)
                r_previous_time_spikes = torch.zeros(tmax, batch_size, num_neurons_pre, stdp_samples)
                for t in range(stdp_samples):
                    central_activities_spikes[:,:,:,t] = self.poisson_spikes(central_activities)
                    r_previous_time_spikes[:,:,:,t] = self.poisson_spikes(r_previous_time)
                
                # filter the r_previous_time_spikes (for synaptic updates)
                r_previous_time_spikes_filtered_pot = self.filter_spike_train(r_previous_time_spikes, decay_rate)
                r_previous_time_spikes_filtered_dep = torch.flip(self.filter_spike_train(torch.flip(r_previous_time_spikes, dims=[0]), decay_rate), dims=[0])
                STDP_pre = r_previous_time_spikes_filtered_pot - r_previous_time_spikes_filtered_dep
                # filter the central_activities_spikes (for bias udpates)
                central_activities_spikes_filtered_pot = self.filter_spike_train(central_activities_spikes, decay_rate)
                central_activities_spikes_filtered_dep = torch.flip(self.filter_spike_train(torch.flip(central_activities_spikes, dims=[0]), decay_rate), dims=[0])
                STDP_post = central_activities_spikes_filtered_pot - central_activities_spikes_filtered_dep
                # rearrange dimensions for right computation
                STDP_post = STDP_post.permute(0, 3, 1, 2)
                central_activities_spikes = central_activities_spikes.permute(0, 3, 2, 1)
                STDP_pre = STDP_pre.permute(0, 3, 1, 2)
                # weights_grad = -2 * 1./batch_size * 1./T * torch.sum(central_activities_spikes[t_start:t_end, :, :, :].double()
                #                                                     @ STDP_pre[t_start:t_end, :, :, :].double(), axis=0).mean(2)
                weights_grad = -2 * 1./batch_size * 1./T * torch.sum(torch.matmul(central_activities_spikes[t_start:t_end, :, :, :].double(),
                                                                     STDP_pre[t_start:t_end, :, :, :].double()), axis=1).mean(0)
                bias_grad = -2 * 1./T * torch.sum(STDP_post[t_start:t_end, :, :, :].double(), axis=0).mean(1).mean(0)

            else:
                raise ValueError("The specified learning rule %s is not valid or proper for this learning rule." % learning_rule)
            
        else:

            if learning_rule == "voltage_difference":
                time_diff = vs_time[t_start:t_end] - vb_time[t_start:t_end]
                bias_grad = -2 * 1./T * torch.sum(time_diff, axis=0).mean(0)
                weights_grad = -2 * 1./batch_size * 1./T * torch.sum(time_diff.permute(0,2,1) @ r_previous_time[t_start:t_end], axis=0)

            elif learning_rule == "derivative_matrix":
                vectorized_jacobians = self.compute_vectorized_jacobian(self.linearactivations)
                time_diff = vs_time[t_start:t_end] - vb_time[t_start:t_end]
                bias_grad = -2 * 1./T * torch.sum(vectorized_jacobians * time_diff, axis=0).mean(0)
                weights_grad = -2 * 1./batch_size * 1./T * \
                            torch.sum((vectorized_jacobians * time_diff).permute(0, 2, 1) @ r_previous_time[t_start:t_end], axis=0)

            elif learning_rule == "nonlinear_difference":
                time_diff = self.forward_activationfunction(vs_time[t_start:t_end]) \
                            - self.forward_activationfunction(vb_time[t_start:t_end])
                bias_grad = -2 * 1./T * torch.sum(time_diff, axis=0).mean(0)
                time_diff = time_diff.permute(0, 2, 1)
                weights_grad = -2 * 1./batch_size * 1. / T * torch.sum(time_diff \
                                                                        @ r_previous_time[t_start:t_end, :, :], axis=0)

            else:
                raise ValueError("The specified learning rule %s is not valid" % learning_rule)

        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._bias, max_norm=self.clip_grad_norm)
        self._weights.grad = weights_grad.detach()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self._weights, max_norm=self.clip_grad_norm)

        return self._weights.grad


    def compute_forward_gradients_deltav_continuous_dfc(self, vs_time, vb_time, central_activities,
                                                        r_previous_time, t_start=None, t_end=None,
                                                        learning_rule="voltage_difference", use_diff_hebbian_updates=False,
                                                        use_stdp_updates=False, save_stdp_measures=False,
                                                        save_correlations=False,
                                                        decay_rate=0.8, stdp_samples=1):
        """
        Useful for stdp correlation measures.
        """

        if t_start is None: t_start = 0
        if t_end is None: t_end = central_activities.shape[0]
        T = t_end - t_start
        batch_size = central_activities.shape[1]
        time_diff = self.forward_activationfunction(vs_time[t_start:t_end]) \
                    - self.forward_activationfunction(vb_time[t_start:t_end])
        time_diff = time_diff.permute(0, 2, 1)
        weights_grad = -2 * 1./batch_size * 1./T * torch.sum(time_diff @ r_previous_time[t_start:t_end, :, :], axis=0)
        return weights_grad


    def compute_forward_gradients_deltav_continuous_diff_hebbian(self, vs_time, vb_time, central_activities,
                                                        r_previous_time, t_start=None, t_end=None,
                                                        learning_rule="voltage_difference", use_diff_hebbian_updates=False,
                                                        use_stdp_updates=False, save_stdp_measures=False,
                                                        save_correlations=False,
                                                        decay_rate=0.8, stdp_samples=1):
        """
        Useful for stdp correlation measures.
        """

        if t_start is None: t_start = 0
        if t_end is None: t_end = central_activities.shape[0]
        T = t_end - t_start
        batch_size = central_activities.shape[1]
        central_activities_t0 = central_activities[t_start:t_end-1]
        central_activities_t1 = central_activities[t_start+1:t_end]
        time_diff = central_activities_t1 - central_activities_t0
        time_diff = time_diff.permute(0,2,1)
        weights_grad = -2 * 1./batch_size * 1./T * torch.sum(time_diff @ r_previous_time[t_start:t_end-1, :, :], axis=0)
        return weights_grad
    

    def compute_forward_gradients_deltav_continuous_stdp(self, vs_time, vb_time, central_activities,
                                                        r_previous_time, t_start=None, t_end=None,
                                                        learning_rule="voltage_difference", use_diff_hebbian_updates=False,
                                                        use_stdp_updates=False, save_stdp_measures=False,
                                                        save_correlations=False,
                                                        decay_rate=0.8, stdp_samples=1):
        """
        Useful for stdp correlation measures.
        """
        
        if t_start is None: t_start = 0
        if t_end is None: t_end = central_activities.shape[0]
        T = t_end - t_start
        tmax = central_activities.shape[0]
        batch_size = central_activities.shape[1]
        num_neurons_pre = r_previous_time.shape[2]
        num_neurons_post = central_activities.shape[2]

        central_activities_spikes = torch.zeros(tmax, batch_size, num_neurons_post, stdp_samples)
        r_previous_time_spikes = torch.zeros(tmax, batch_size, num_neurons_pre, stdp_samples)
        for t in range(stdp_samples):
            central_activities_spikes[:,:,:,t] = self.poisson_spikes(central_activities)
            r_previous_time_spikes[:,:,:,t] = self.poisson_spikes(r_previous_time)
        
        # filter the r_previous_time_spikes (for synaptic updates)
        r_previous_time_spikes_filtered_pot = self.filter_spike_train(r_previous_time_spikes, decay_rate)
        r_previous_time_spikes_filtered_dep = torch.flip(self.filter_spike_train(torch.flip(r_previous_time_spikes, dims=[0]), decay_rate), dims=[0])
        STDP_pre = r_previous_time_spikes_filtered_pot - r_previous_time_spikes_filtered_dep
        # filter the central_activities_spikes (for bias udpates)
        central_activities_spikes_filtered_pot = self.filter_spike_train(central_activities_spikes, decay_rate)
        central_activities_spikes_filtered_dep = torch.flip(self.filter_spike_train(torch.flip(central_activities_spikes, dims=[0]), decay_rate), dims=[0])
        STDP_post = central_activities_spikes_filtered_pot - central_activities_spikes_filtered_dep
        # rearrange dimensions for right computation
        STDP_post = STDP_post.permute(0, 3, 1, 2)
        central_activities_spikes = central_activities_spikes.permute(0, 3, 2, 1)
        STDP_pre = STDP_pre.permute(0, 3, 1, 2)
        # weights_grad = -2 * 1./batch_size * 1./T * torch.sum(central_activities_spikes[t_start:t_end, :, :, :].double()
        #                                                     @ STDP_pre[t_start:t_end, :, :, :].double(), axis=0).mean(2)
        weights_grad = -2 * 1./batch_size * 1./T * torch.sum(torch.matmul(central_activities_spikes[t_start:t_end, :, :, :].double(),
                                                                STDP_pre[t_start:t_end, :, :, :].double()), axis=1).mean(0)
        bias_grad = -2 * 1./T * torch.sum(STDP_post[t_start:t_end, :, :, :].double(), axis=0).mean(1).mean(0)
        
        return weights_grad


    def compute_feedback_gradients(self, v_a, u, sigma, scale=1.):
        r"""
        Compute the gradients for the feedback parameters Q (in the code they
        are presented by ``self.feedbackweights``) and save it in
        ``self._feedbackweights.grad``, using the
        following update rule:

        ..math::
            \Delta Q_i = \frac{1}{\sigma^2} \mathbf{v}^A_i \mathbf{u}^T

        Note that pytorch saves the positive gradient, hence we should save
        :math:`-\Delta Q_i`.

        Args:
            delta_v (torch.Tensor): :math:`\Delta \mathbf{v}_i`, the apical compartment voltage
                at its initial perturbed state.
            u: :math:`\mathbf{u}`, the control input at its initial state
            sigma (float): the standard deviation of the noise used for
                obtaining an output target in the feedback training
            scale (float): scaling factor for the gradient. Is used when
                continuous feedback updates are computed in the efficient
                controller implementation, where scale=1/t_max, as we want
                to average over all timesteps.
        """

        batch_size = v_a.shape[0]

        if sigma < 0.01:
            scale = scale/0.01**2
        else:
            scale = scale/sigma**2

        weights_grad = - scale/batch_size * v_a.t().mm(u)

        self._feedback_weights.grad += weights_grad.detach()


    def compute_feedback_gradients_old(self, delta_v, u, sigma):
        r"""
        Compute the gradients for the feedback parameters Q (in the code they
        are presented by ``self.K``) and save it in
        ``self._inversion_matrix.grad``, using the
        following update rule:

        ..math::
            \Delta Q_i = -\frac{1}{\sigma^2} \Delta \mathbf{v}_i \mathbf{u}^T 

        Note that pytorch saves the positive gradient, hence we should save
        :math:`-\Delta Q_i`.

        Args:
            delta_v (torch.Tensor): :math:`\Delta \mathbf{v}_i`, the apical compartment voltage
                level at steady-state.
            u: :math:`\mathbf{u}`, the control input at steady state
            sigma (float): the standard deviation of the noise used for
                obtaining an output target in the feedback training
        """

        batch_size = delta_v.shape[0]

        if sigma < 0.01:
            scale = 1/0.01**2
        else:
            scale = 1/sigma**2

        weights_grad = scale/batch_size * delta_v.t().mm(u)

        if self._feedback_weights.grad is None:
            self._feedback_weights.grad = weights_grad.detach()
        else:
            self._feedback_weights.grad += weights_grad.detach()


    def compute_feedback_gradients_continuous(self, va_time, u_time,
                                              t_start=None, t_end=None,
                                              sigma=1., scaling=1.):
        r"""
        Computes the feedback gradients using an integration (sum) of the
        continuous time learning rule over the specified time interval.
        ..math::
            \frac{dQ_i}{dt} = -v_i^A u^T 

        The gradients are stored in ``self.feedbackweights.grad``. Important,
        the ``.grad`` attribute represents the positive gradient, hence, we
        need to save :math:`-\Delta Q_i` in it.
        Args:
            va_time (torch.Tensor): The apical compartment voltages over
                a certain time period
            u_time (torch.Tensor): The control inputs over  certain time period
            t_start (torch.Tensor): The start index from which the summation
                over time should start
            t_end (torch.Tensor): The stop index at which the summation over
                time should stop
            sigma (float): the standard deviation of the noise in the network
                dynamics. This is used to scale the fb weight update, such that
                its magnitude is independent of the noise variance.
            scaling (float): In the theory for the feedback weight updates, the
                update for each layer should be scaled with
                :math:`(1+\tau_{v}/\tau_{\epsilon})^{L-i}`, with L the amount of
                layers and i the layer index. ``scaling`` should be the factor
                :math:`(1+\tau_{v}/\tau_{\epsilon})^{L-i}` for this layer.
        """

        if t_start is None: t_start = 0 
        if t_end is None: t_end = u_time.shape[0]
        T = t_end - t_start
        batch_size = u_time.shape[1]

        if sigma < 0.01:
            scale = 1 / 0.01 ** 2
        else:
            scale = 1 / sigma ** 2

        scale *= scaling
        aux = va_time[t_start:t_end].permute(0,2,1) @ u_time[t_start:t_end]
        feedbackweights_grad = scale/(T * batch_size) * torch.sum(aux, axis=0)

        self._feedback_weights.grad = feedbackweights_grad.detach()


    def noisy_dummy_forward(self, x, noise):
        """
        Propagates the layer input ``x`` to the next layer, while adding
        ``noise`` to the linear activations of this layer. This method does
        not save the resulting activations and linear activations in the layer
        object.
        Args:
            x (torch.Tensor): layer input
            noise (torch.Tensor): Noise to be added to the linear activations
                of this layer.

        Returns (torch.Tensor): The resulting post-nonlinearity activation of
            this layer.
        """

        a = x.mm(self.weights.t())
        if self.bias is not None:
            a += self.bias.unsqueeze(0).expand_as(a)
        a += noise
        h = self.forward_activationfunction(a)
        return h

    def save_feedback_batch_logs(self, writer, step, name, no_gradient=False,
                                 init=False):
        """
        Save logs for one minibatch.
        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
            init (bool): flag indicating that the training is in the
                initialization phase (only training the feedback weights).
        """

        if init:
            prefix = 'feedback_pretraining/{}/'.format(name)
        else:
            prefix = 'feedback_training/{}/'.format(name)
        feedback_weights_norm = torch.norm(self.feedbackweights)
        writer.add_scalar(tag=prefix + 'feedback_weights_norm',
                          scalar_value=feedback_weights_norm,
                          global_step=step)
        if self.feedbackweights.grad is not None:
            feedback_weights_gradients_norm = torch.norm(self.feedbackweights.grad)
            writer.add_scalar(
                tag=prefix + 'feedback_weights_gradient_norm',
                scalar_value=feedback_weights_gradients_norm,
                global_step=step)

    def save_forward_batch_logs(self, writer, step, name, no_gradient=False):
        """Save forward weight logs for one minibatch.

        Args:
            writer (SummaryWriter): summary writer from tensorboardX
            step (int): the global step used for the x-axis of the plots
            name (str): The name of the layer
            no_gradient (bool): flag indicating whether we should skip saving
                the gradients of the feedback weights.
        """
        prefix = 'forward_training/{}/'.format(name)
        weights_norm = torch.norm(self.weights)
        writer.add_scalar(tag=prefix + 'forward_weights_norm',
                          scalar_value=weights_norm,
                          global_step=step)
        if self.weights.grad is not None:
            weights_gradients_norm = torch.norm(self.weights.grad)
            writer.add_scalar(
                tag=prefix + 'forward_weights_gradient_norm',
                scalar_value=weights_gradients_norm,
                global_step=step)


class TPDILayer(DFCLayer):

    def compute_forward_gradients(self, h_target, h_previous, norm_ratio=1.):
        """
        Compute the gradient of the forward weights and bias, based on the
        local mse loss between the layer activation and layer target.
        The gradients are saved in the .grad attribute of the forward weights
        and forward bias.
        Args:
            h_target (torch.Tensor): the DTP target of the current layer
            h_previous (torch.Tensor): the rate activation of the previous
                layer
            norm_ratio (float): Depreciated.
        """
        
        if self.forward_activation == 'linear':
            teaching_signal = 2 * (self.activations - h_target)
        else:
            vectorized_jacobians = self.compute_vectorized_jacobian(self.linearactivations)
            teaching_signal = 2 * vectorized_jacobians * (self.activations - h_target)
        batch_size = h_target.shape[0]
        bias_grad = teaching_signal.mean(0)
        weights_grad = 1./batch_size * teaching_signal.t().mm(h_previous)

        if self.bias is not None:
            self._bias.grad = bias_grad.detach()
            torch.nn.utils.clip_grad_norm_(self._bias, max_norm=1)
        self._weights.grad = weights_grad.detach()
        torch.nn.utils.clip_grad_norm_(self._weights, max_norm=1)


class GNLayer(DFCLayer):

    def compute_forward_gradients(self, weight_update, bias_update=None):
        self._weights.grad += weight_update
        if self._bias is not None:
            self._bias.grad += bias_update
