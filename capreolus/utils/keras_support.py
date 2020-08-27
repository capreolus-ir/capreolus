# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Adam optimizer implementation."""
# pylint: disable=g-classes-have-attributes
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule


class AdamMultilr(optimizer_v2.OptimizerV2):
    r"""Optimizer that implements the Adam algorithm.

    Adam optimization is a stochastic gradient descent method that is based on
    adaptive estimation of first-order and second-order moments.

    According to
    [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
    the method is "*computationally
    efficient, has little memory requirement, invariant to diagonal rescaling of
    gradients, and is well suited for problems that are large in terms of
    data/parameters*".

    Args:
      learning_rate: A `Tensor`, floating point value, or a schedule that is a
        `tf.keras.optimizers.schedules.LearningRateSchedule`, or a callable
        that takes no arguments and returns the actual value to use, The
        learning rate. Defaults to 0.001.
      beta_1: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use. The
        exponential decay rate for the 1st moment estimates. Defaults to 0.9.
      beta_2: A float value or a constant float tensor, or a callable
        that takes no arguments and returns the actual value to use, The
        exponential decay rate for the 2nd moment estimates. Defaults to 0.999.
      epsilon: A small constant for numerical stability. This epsilon is
        "epsilon hat" in the Kingma and Ba paper (in the formula just before
        Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to
        1e-7.
      amsgrad: Boolean. Whether to apply AMSGrad variant of this algorithm from
        the paper "On the Convergence of Adam and beyond". Defaults to `False`.
      name: Optional name for the operations created when applying gradients.
        Defaults to `"Adam"`.
      **kwargs: Keyword arguments. Allowed to be one of
        `"clipnorm"` or `"clipvalue"`.
        `"clipnorm"` (float) clips gradients by norm; `"clipvalue"` (float) clips
        gradients by value.

    Usage:

    >>> opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    >>> var1 = tf.Variable(10.0)
    >>> loss = lambda: (var1 ** 2)/2.0       # d(loss)/d(var1) == var1
    >>> step_count = opt.minimize(loss, [var1]).numpy()
    >>> # The first step is `-learning_rate*sign(grad)`
    >>> var1.numpy()
    9.9

    Reference:
      - [Kingma et al., 2014](http://arxiv.org/abs/1412.6980)
      - [Reddi et al., 2018](
          https://openreview.net/pdf?id=ryQu7f-RZ) for `amsgrad`.

    Notes:

    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since Adam uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

    The sparse implementation of this algorithm (used when the gradient is an
    IndexedSlices object, typically because of `tf.gather` or an embedding
    lookup in the forward pass) does apply momentum to variable slices even if
    they were not used in the forward pass (meaning they have a gradient equal
    to zero). Momentum decay (beta1) is also applied to the entire momentum
    accumulator. This means that the sparse behavior is equivalent to the dense
    behavior (in contrast to some momentum implementations which ignore momentum
    unless a variable slice was actually used).
    """

    _HAS_AGGREGATE_GRAD = True

    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        amsgrad=False,
        name="AdamMultilr",
        pattern_lrs=None,
        **kwargs,
    ):
        super(AdamMultilr, self).__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or backend_config.epsilon()
        self.amsgrad = amsgrad
        self.pattern_lrs = pattern_lrs  # {["pattern": [pattern1, pattern2], "lr": lr]}

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _decayed_multi_lr(self, lr, var_dtype):
        """Get decayed learning rate as a Tensor with dtype=var_dtype."""
        # lr_t = self._get_hyper("learning_rate", var_dtype)
        lr_t = lr
        if isinstance(lr_t, learning_rate_schedule.LearningRateSchedule):
            local_step = math_ops.cast(self.iterations, var_dtype)
            lr_t = math_ops.cast(lr_t(local_step), var_dtype)
        if self._initial_decay > 0.0:
            local_step = math_ops.cast(self.iterations, var_dtype)
            decay_t = self._get_hyper("decay", var_dtype)
            lr_t = lr_t / (1.0 + decay_t * local_step)
        return lr_t

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(AdamMultilr, self)._prepare_local(var_device, var_dtype, apply_state)
        if self.pattern_lrs:
            for i, pair in enumerate(self.pattern_lrs):
                lr_t = array_ops.identity(self._decayed_multi_lr(pair["lr"], var_dtype))
                apply_state[(var_device, var_dtype)][f"lr-{i}_t"] = lr_t

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_1_t = array_ops.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = array_ops.identity(self._get_hyper("beta_2", var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)

        updated_lrs = {
            lr_name.replace("_t", ""): apply_state[(var_device, var_dtype)][lr_name]
            * (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power))
            for lr_name in apply_state[(var_device, var_dtype)]
            if "lr" in lr_name
        }
        # lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
        #       (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                # lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
                **updated_lrs,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super(AdamMultilr, self).set_weights(weights)

    def _find_lr(self, var_name, coefficients):
        lr_idx = -1
        for i, pattern_lr in enumerate(self.pattern_lrs):
            for pattern in pattern_lr["patterns"]:
                if re.search(pattern, var_name):
                    lr_idx = i
                    break
            if lr_idx != -1:
                break

        if lr_idx == -1:  # unfound pattern
            lr = coefficients["lr_t"]
            # print(">>>>>> DEFAULT LR: ", lr, var.name)
        else:
            lr = coefficients[f"lr-{lr_idx}_t"]
            # print("bert LR: ", lr, var.name)
        return lr

    def _resource_apply_dense(self, grad, var, apply_state=None):
        # print("grad: ", grad.name, grad.shape, "var: ", var.name, var.shape)
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)
        lr = self._find_lr(var.name, coefficients)
        # lr_idx = -1
        # for i, pattern_lr in enumerate(self.pattern_lrs):
        #     for pattern in pattern_lr["patterns"]:
        #         print("pattern: ", pattern, re.search(pattern, var.name))
        #         if re.search(pattern, var.name):
        #             lr_idx = i
        #             break
        #     if lr_idx != -1:
        #         break
        #
        # if lr_idx == -1:  # unfound pattern
        #     lr = coefficients["lr_t"]
        #     # print(">>>>>> DEFAULT LR: ", lr, var.name)
        # else:
        #     lr = coefficients[f"lr-{lr_idx}_t"]
        #     # print("bert LR: ", lr, var.name)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        if not self.amsgrad:
            return training_ops.resource_apply_adam(
                var.handle,
                m.handle,
                v.handle,
                coefficients["beta_1_power"],
                coefficients["beta_2_power"],
                lr,  # coefficients['lr_t'],
                coefficients["beta_1_t"],
                coefficients["beta_2_t"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )
        else:
            vhat = self.get_slot(var, "vhat")
            return training_ops.resource_apply_adam_with_amsgrad(
                var.handle,
                m.handle,
                v.handle,
                vhat.handle,
                coefficients["beta_1_power"],
                coefficients["beta_2_power"],
                lr,  # coefficients['lr_t'],
                coefficients["beta_1_t"],
                coefficients["beta_2_t"],
                coefficients["epsilon"],
                grad,
                use_locking=self._use_locking,
            )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get((var_device, var_dtype)) or self._fallback_apply_state(var_device, var_dtype)

        lr = self._find_lr(var.name, coefficients)
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = state_ops.assign(m, m * coefficients["beta_1_t"], use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * coefficients["one_minus_beta_2_t"]
        v_t = state_ops.assign(v, v * coefficients["beta_2_t"], use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if not self.amsgrad:
            v_sqrt = math_ops.sqrt(v_t)
            var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + coefficients["epsilon"]), use_locking=self._use_locking)
            return control_flow_ops.group(*[var_update, m_t, v_t])
        else:
            v_hat = self.get_slot(var, "vhat")
            v_hat_t = math_ops.maximum(v_hat, v_t)
            with ops.control_dependencies([v_hat_t]):
                v_hat_t = state_ops.assign(v_hat, v_hat_t, use_locking=self._use_locking)
            v_hat_sqrt = math_ops.sqrt(v_hat_t)
            var_update = state_ops.assign_sub(
                var, lr * m_t / (v_hat_sqrt + coefficients["epsilon"]), use_locking=self._use_locking
            )
            return control_flow_ops.group(*[var_update, m_t, v_t, v_hat_t])

    def get_config(self):
        config = super(AdamMultilr, self).get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "pattern_lrs": self.pattern_lrs,
            }
        )
        return config
