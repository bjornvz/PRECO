import numpy as np
import torch
from PRECO.utils import *

class Optimizer(object):
    def __init__(self, params, optim_type, learning_rate, batch_scale=False, grad_clip=None, weight_decay=None):
        self._optim_type = optim_type
        self._params = params
        self.n_params = len(params)
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.batch_scale = batch_scale
        self.weight_decay = weight_decay
        if isinstance(params["w"], list):  
            self.is_list = True  # PCN
        else:  
            self.is_list = False # PCG

        self._hparams = f"{optim_type}_lr={self.learning_rate}_gclip={self.grad_clip}_bscale={self.batch_scale}_wd={self.weight_decay}"

    @property
    def hparams(self):
        return self._hparams 
    
    @property
    def hparams_dict(self):
        return {"lr": self.learning_rate, "gradclip": self.grad_clip, "batchscale": self.batch_scale, "wd": self.weight_decay}

    def clip_grads(self, grad):
        if self.grad_clip is not None:
            grad = torch.clamp(grad, -self.grad_clip, self.grad_clip)

    def scale_batch(self, grad, batch_size):
        if self.batch_scale:
            grad /= batch_size

    # def decay_weights(self, param):
    #     if self.weight_decay is not None:
    #         param.grad["weights"] = param.grad["weights"] - self.weight_decay * param.weights

    def step(self, *args, **kwargs):
        raise NotImplementedError
    

class SGD(Optimizer):
    def __init__(self, params, learning_rate, batch_scale=False, grad_clip=None, weight_decay=None, model=None):
        super().__init__(params, optim_type="SGD", learning_rate=learning_rate, batch_scale=batch_scale, grad_clip=grad_clip,
                         weight_decay=weight_decay)

    def step(self, params, grads, batch_size):
        if self.is_list:
            for i in range(len(params["w"])):
                self._update_single_param(params["w"], grads["w"], i, batch_size, self.learning_rate)
                if params["use_bias"]:
                    self._update_single_param(params["b"], grads["b"], i, batch_size, self.learning_rate)
        else:
            self._update_single_param(params["w"], grads["w"], None, batch_size, self.learning_rate)
            if params["use_bias"]:
                self._update_single_param(params["b"], grads["b"], None, batch_size, self.learning_rate)

    def _update_single_param(self, param_group, grad_group, i, batch_size, learning_rate):
        if i is not None:
            param = param_group[i]
            grad = grad_group[i]
        else:
            param = param_group
            grad = grad_group

        self.scale_batch(grad, batch_size)
        self.clip_grads(grad)
        param -= learning_rate * grad

        if i is not None:
            param_group[i] = param
        else:
            param_group.copy_(param)


class Adam(Optimizer):
    def __init__(self, params, learning_rate, batch_scale=False, grad_clip=None, 
                 beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0, AdamW=False):
        super().__init__(params, optim_type="Adam", learning_rate=learning_rate, batch_scale=batch_scale,
                         grad_clip=grad_clip, weight_decay=weight_decay)

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # Time step
        self.weight_decay = weight_decay
        self.AdamW = AdamW

        if self.is_list: # PCN
            self.m_w = [torch.zeros_like(param, device=DEVICE) for param in params["w"]]
            self.v_w = [torch.zeros_like(param, device=DEVICE) for param in params["w"]]
            self.m_b = [torch.zeros_like(param, device=DEVICE) for param in params["b"]]
            self.v_b = [torch.zeros_like(param, device=DEVICE) for param in params["b"]]
        else: # PCG
            self.m_w = torch.zeros_like(params["w"], device=DEVICE)
            self.v_w = torch.zeros_like(params["w"], device=DEVICE)
            self.m_b = torch.zeros_like(params["b"], device=DEVICE)
            self.v_b = torch.zeros_like(params["b"], device=DEVICE)

    def step(self, params, grads, batch_size):
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1. - self.beta2 ** self.t) / (1. - self.beta1 ** self.t)

        if self.is_list:
            for i in range(len(params["w"])):
                self._update_single_param(params["w"], grads["w"], self.m_w, self.v_w, i, batch_size, lr_t, self.AdamW)
                if params["use_bias"]:
                    self._update_single_param(params["b"], grads["b"], self.m_b, self.v_b, i, batch_size, lr_t, self.AdamW)
        else:
            self._update_single_param(params["w"], grads["w"], self.m_w, self.v_w, None, batch_size, lr_t, self.AdamW)
            if params["use_bias"]:
                self._update_single_param(params["b"], grads["b"], self.m_b, self.v_b, None, batch_size, lr_t, self.AdamW)

    def _update_single_param(self, param_group, grad_group, m_group, v_group, i, batch_size, lr_t, AdamW):
        if i is not None:
            param = param_group[i]
            grad = grad_group[i]
            m = m_group[i]
            v = v_group[i]
        else:
            param = param_group
            grad = grad_group
            m = m_group
            v = v_group

        self.scale_batch(grad, batch_size)
        self.clip_grads(grad)

        grad += self.weight_decay * param
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)
        step = lr_t * m / (torch.sqrt(v) + self.epsilon)

        if AdamW:
            param *= (1. - self.weight_decay * self.learning_rate)
        param -= step

        if i is not None:
            param_group[i] = param
            m_group[i] = m
            v_group[i] = v
        else:
            param_group.copy_(param)
            m_group.copy_(m)
            v_group.copy_(v)



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0, verbose=True, relative=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_obj = float('inf')
        self.verbose = verbose
        self.relative = relative

    def early_stop(self, validation_obj):
        if np.isnan(validation_obj):
            print("Validation objective is NaN. Stopping early.")
            return True
        difference = validation_obj - self.min_validation_obj
        if self.relative:
            difference /= self.min_validation_obj
        if validation_obj < self.min_validation_obj:
            if self.verbose:
                print(f"Validation objective decreased ({self.min_validation_obj:.6f} --> {validation_obj:.6f}).")
            self.min_validation_obj = validation_obj
            self.counter = 0
        elif difference >= self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"Validation objective increased ({self.min_validation_obj:.6f} --> {validation_obj:.6f}).")
                print(f"Early stopping counter: {self.counter} out of {self.patience}\n")
            if self.counter >= self.patience:
                return True
        return False