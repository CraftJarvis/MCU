import random
import logging
from typing import Any, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange
import gym
import gymnasium
from gym3.types import DictType, Discrete, Real, TensorType, ValType

LOG0 = -100

def fan_in_linear(module: nn.Module, scale=1.0, bias=True):
    """Fan-in init"""
    module.weight.data *= scale / module.weight.norm(dim=1, p=2, keepdim=True)

    if bias:
        module.bias.data *= 0

class ActionHead(nn.Module):
    """Abstract base class for action heads compatible with forc"""

    def forward(self, input_data, **kwargs) -> Any:
        """
        Just a forward pass through this head
        :returns pd_params - parameters describing the probability distribution
        """
        raise NotImplementedError

    def logprob(self, action_sample, pd_params, **kwargs):
        """Logartithm of probability of sampling `action_sample` from a probability described by `pd_params`"""
        raise NotImplementedError

    def entropy(self, pd_params):
        """Entropy of this distribution"""
        raise NotImplementedError

    def sample(self, pd_params, deterministic: bool = False) -> Any:
        """
        Draw a sample from probability distribution given by those params

        :param pd_params Parameters of a probability distribution
        :param deterministic Whether to return a stochastic sample or deterministic mode of a distribution
        """
        raise NotImplementedError

    def kl_divergence(self, params_q, params_p):
        """KL divergence between two distribution described by these two params"""
        raise NotImplementedError


class DiagGaussianActionHead(ActionHead):
    """
    Action head where actions are normally distributed uncorrelated variables with specific means and variances.

    Means are calculated directly from the network while standard deviations are a parameter of this module
    """

    LOG2PI = np.log(2.0 * np.pi)

    def __init__(self, input_dim: int, num_dimensions: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions

        self.linear_layer = nn.Linear(input_dim, num_dimensions)
        self.log_std = nn.Parameter(torch.zeros(num_dimensions), requires_grad=True)

    def reset_parameters(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        assert not mask, "Can not use a mask in a gaussian action head"
        means = self.linear_layer(input_data)
        # Unsqueeze many times to get to the same shape
        logstd = self.log_std[(None,) * (len(means.shape) - 1)]

        mean_view, logstd = torch.broadcast_tensors(means, logstd)

        return torch.stack([mean_view, logstd], dim=-1)

    def logprob(self, action_sample: torch.Tensor, pd_params: torch.Tensor) -> torch.Tensor:
        """Log-likelihood"""
        means = pd_params[..., 0]
        log_std = pd_params[..., 1]

        std = torch.exp(log_std)

        z_score = (action_sample - means) / std

        return -(0.5 * ((z_score ** 2 + self.LOG2PI).sum(dim=-1)) + log_std.sum(dim=-1))

    def entropy(self, pd_params: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution entropy calculation - sum probs * log(probs).
        In case of diagonal gaussian distribution - 1/2 log(2 pi e sigma^2)
        """
        log_std = pd_params[..., 1]
        return (log_std + 0.5 * (self.LOG2PI + 1)).sum(dim=-1)

    def sample(self, pd_params: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        means = pd_params[..., 0]
        log_std = pd_params[..., 1]

        if deterministic:
            return means
        else:
            return torch.randn_like(means) * torch.exp(log_std) + means

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)

        Formula is:
        log(sigma_p) - log(sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2))/(2 * sigma_p^2)
        """
        means_q = params_q[..., 0]
        log_std_q = params_q[..., 1]

        means_p = params_p[..., 0]
        log_std_p = params_p[..., 1]

        std_q = torch.exp(log_std_q)
        std_p = torch.exp(log_std_p)

        kl_div = log_std_p - log_std_q + (std_q ** 2 + (means_q - means_p) ** 2) / (2.0 * std_p ** 2) - 0.5

        return kl_div.sum(dim=-1, keepdim=True)


class CategoricalActionHead(ActionHead):
    """Action head with categorical actions"""

    def __init__(
        self, input_dim: int, shape: Tuple[int], num_actions: int, builtin_linear_layer: bool = True, temperature: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.output_shape = shape + (num_actions,)
        self.temperature = temperature

        if builtin_linear_layer:
            self.linear_layer = nn.Linear(input_dim, np.prod(self.output_shape))
        else:
            assert (
                input_dim == num_actions
            ), f"If input_dim ({input_dim}) != num_actions ({num_actions}), you need a linear layer to convert them."
            self.linear_layer = None

    def reset_parameters(self):
        if self.linear_layer is not None:
            init.orthogonal_(self.linear_layer.weight, gain=0.01)
            init.constant_(self.linear_layer.bias, 0.0)
            finit.fan_in_linear(self.linear_layer, scale=0.01)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> Any:
        if self.linear_layer is not None:
            flat_out = self.linear_layer(input_data)
        else:
            flat_out = input_data
        shaped_out = flat_out.reshape(flat_out.shape[:-1] + self.output_shape)
        shaped_out /= self.temperature
        if mask is not None:
            shaped_out[~mask] = LOG0

        # Convert to float32 to avoid RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Half'
        return F.log_softmax(shaped_out.float(), dim=-1)

    def logprob(self, actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        value = actions.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        result = log_pmf.gather(-1, value).squeeze(-1)
        # result is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            result = result.sum(dim=-1)
        return result

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Categorical distribution entropy calculation - sum probs * log(probs)"""
        probs = torch.exp(logits)
        entropy = -torch.sum(probs * logits, dim=-1)
        # entropy is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            entropy = entropy.sum(dim=-1)
        return entropy

    # minecraft domain should directly use this sample function
    def sample(self, logits: torch.Tensor, deterministic: bool = False, **kwargs) -> Any:
        """The original sample function from VPT library. """
        if deterministic:
            return torch.argmax(logits, dim=-1)
        else:
            logits = torch.nn.functional.log_softmax(logits, dim=-1)
            # Gumbel-Softmax trick.
            u = torch.rand_like(logits)
            # In float16, if you have around 2^{float_mantissa_bits} logits, sometimes you'll sample 1.0
            # Then the log(-log(1.0)) will give -inf when it should give +inf
            # This is a silly hack to get around that.
            # This hack does not skew the probability distribution, because this event can't possibly win the argmax.
            u[u == 1.0] = 0.999
            
            return torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)
    
    # def sample(self, logits: torch.Tensor, deterministic: bool = False, p: float = 0.85, **kwargs) -> Any:
    #     """The nucleus sample function. """
    #     # assert not deterministic, "Not deterministic"
    #     if random.randint(0, 99) < 5 or deterministic:
    #         # this is to introduce some randomness to avoid deterministic behavior
    #         return self.vanilla_sample(logits, deterministic)
    #     probs = torch.exp(logits)
    #     sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    #     cum_sum_probs = torch.cumsum(sorted_probs, dim=-1) 
    #     # print(f"{p = }, {cum_sum_probs = }")
    #     nucleus = cum_sum_probs < p 
    #     nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    #     sorted_log_probs = torch.log(sorted_probs)
    #     sorted_log_probs[~nucleus] = float('-inf')
    #     sampled_sorted_indexes = self.vanilla_sample(sorted_log_probs, deterministic=False)
    #     res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))
    #     return res.squeeze(-1)

    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)
        When talking about logits this is:
        sum exp(Q_i) * (Q_i - P_i)
        """
        kl = (torch.exp(logits_q) * (logits_q - logits_p)).sum(-1, keepdim=True)
        # kl is per-entry, still of size self.output_shape; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            kl = kl.sum(dim=-2)  # dim=-2 because we use keepdim=True above.
        return kl

class HLGaussActionHead(ActionHead):
    
    def __init__(
        self, 
        input_dim: int, 
        num_dimensions: int, 
        min_value: float, 
        max_value: float, 
        num_bins: int, 
        sigma: float = 1.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_dimensions = num_dimensions
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = nn.Parameter(
            torch.linspace(
                min_value, max_value, num_bins + 1, dtype=torch.float32
            ), requires_grad=False
        )
        self.linear_layer = nn.Linear(input_dim, num_dimensions * num_bins)

    def reset_parameters(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)
    
    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        assert not mask, "Can not use a mask in a gaussian action head"
        logits = self.linear_layer(input_data).reshape(input_data.shape[:-1] + (self.num_dimensions, self.num_bins))
        return logits 

    def logprob(self, action_sample: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        '''
        :param action_sample: B, T, D
        :param logits: B, T, D, X
        :return: B, T
        '''
        B, T = action_sample.shape[:2]
        target = self.transform_to_probs(action_sample) # B, T, D, X
        # remove nan elements (because of large action labels)
        target[torch.isnan(target)] = 0.
        lp = (target * F.log_softmax(logits, dim=-1)).sum([-1, -2]) # (B, T)
        return lp
    
    def sample(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        '''
        :param logits: B, T, D, X
        :return: B, T, D
        '''
        return self.transform_from_probs(F.softmax(logits, dim=-1))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1))
            / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)
    
    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim=-1)

class MSEActionHead(ActionHead):

    def __init__(self, input_dim: int, num_dimensions: int):
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions

        self.linear_layer = nn.Linear(input_dim, num_dimensions)

    def reset_parameters(self):
        init.orthogonal_(self.linear_layer.weight, gain=0.01)
        init.constant_(self.linear_layer.bias, 0.0)

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> torch.Tensor:
        assert not mask, "Can not use a mask in a mse action head"
        means = self.linear_layer(input_data)

        return means

    def logprob(self, action_sample: torch.Tensor, pd_params: torch.Tensor) -> torch.Tensor:
        return - ((action_sample - pd_params).pow(2)).sum(dim=-1)

    def entropy(self, pd_params: torch.Tensor) -> torch.Tensor:
        # raise NotImplementedError("Entropy is not defined for MSE action head")
        return torch.zeros_like(pd_params).sum(dim=-1)

    def sample(self, pd_params: torch.Tensor, deterministic: bool = False, **kwargs) -> torch.Tensor:
        return pd_params

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("KL divergence is not defined for MSE action head")

class TupleActionHead(nn.ModuleList, ActionHead):
    """Action head with multiple sub-actions"""

    def reset_parameters(self):
        for subhead in self:
            subhead.reset_parameters()
    
    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        return tuple([ subhead(input_data) for subhead in self ])

    def logprob(self, actions: Tuple[torch.Tensor], logits: Tuple[torch.Tensor]) -> torch.Tensor:
        return tuple([ subhead.logprob(actions[k], logits[k]) for k, subhead in enumerate(self) ])

    def sample(self, logits: Tuple[torch.Tensor], deterministic: bool = False) -> Any:
        return tuple([ subhead.sample(logits[k], deterministic) for k, subhead in enumerate(self) ])

    def entropy(self, logits: Tuple[torch.Tensor]) -> torch.Tensor:
        return tuple([ subhead.entropy(logits[k]) for k, subhead in enumerate(self) ])

    def kl_divergence(self, logits_q: Tuple[torch.Tensor], logits_p: Tuple[torch.Tensor]) -> torch.Tensor:
        return sum( subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in enumerate(self) )

class DictActionHead(nn.ModuleDict, ActionHead):
    """Action head with multiple sub-actions"""

    def reset_parameters(self):
        for subhead in self.values():
            subhead.reset_parameters()

    def forward(self, input_data: torch.Tensor, **kwargs) -> Any:
        """
        :param kwargs: each kwarg should be a dict with keys corresponding to self.keys()
                e.g. if this ModuleDict has submodules keyed by 'A', 'B', and 'C', we could call:
                    forward(input_data, foo={'A': True, 'C': False}, bar={'A': 7}}
                Then children will be called with:
                    A: forward(input_data, foo=True, bar=7)
                    B: forward(input_data)
                    C: forward(input_Data, foo=False)
        """
        result = {}
        for head_name, subhead in self.items():
            head_kwargs = {
                kwarg_name: kwarg[head_name]
                for kwarg_name, kwarg in kwargs.items()
                if kwarg is not None and head_name in kwarg
            }
            result[head_name] = subhead(input_data, **head_kwargs)
        return result

    def logprob(self, actions: Dict[str, torch.Tensor], logits: Dict[str, torch.Tensor], return_dict=False) -> torch.Tensor:
        if return_dict:
            return {k: subhead.logprob(actions[k], logits[k]) for k, subhead in self.items()}
        else:
            return sum(subhead.logprob(actions[k], logits[k]) for k, subhead in self.items())

    def sample(self, logits: Dict[str, torch.Tensor], deterministic: bool = False) -> Any:
        return {k: subhead.sample(logits[k], deterministic) for k, subhead in self.items()}

    def entropy(self, logits: Dict[str, torch.Tensor], return_dict=False) -> torch.Tensor:
        if return_dict:
            return {k: subhead.entropy(logits[k]) for k, subhead in self.items()}
        else:
            return sum(subhead.entropy(logits[k]) for k, subhead in self.items())

    def kl_divergence(self, logits_q: Dict[str, torch.Tensor], logits_p: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(subhead.kl_divergence(logits_q[k], logits_p[k]) for k, subhead in self.items())

class DLMLActionHead(ActionHead):

    def __init__(self, 
        input_dim: int, 
        num_dimensions: int, 
        num_mixtures: int, 
        num_output_bins: int, 
        output_min: float, 
        output_max: float,
        log_scale_min: float
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_dimensions = num_dimensions
        self.num_mixtures = num_mixtures
        self.num_output_bins = num_output_bins
        self.output_min = output_min
        self.output_max = output_max
        self.log_scale_min = log_scale_min

        self.fc_mixture_logits = nn.Linear(input_dim, num_mixtures)
        self.fc_means = nn.Linear(input_dim, num_dimensions * num_mixtures)
        self.fc_log_scales = nn.Linear(input_dim, num_dimensions * num_mixtures)

    def reset_parameters(self):
        pass

    def forward(self, input_data: torch.Tensor, mask=None, **kwargs) -> Dict[str, torch.Tensor]:
        assert not mask, "Can not use a mask in a mse action head"
        mixture_logits = self.fc_mixture_logits(input_data)
        means = self.fc_means(input_data)
        log_scales = self.fc_log_scales(input_data)
        log_scales = torch.clamp(log_scales, min=self.log_scale_min)
        return {"mixture_logits": mixture_logits, "means": means, "log_scales": log_scales}
    
    def _normalize(self, input_data: torch.Tensor) -> torch.Tensor:
        #  [output_min, output_max] -> [-1, 1]
        return 2 * (input_data - self.output_min) / (self.output_max - self.output_min) - 1
    
    def _denormalize(self, input_data: torch.Tensor) -> torch.Tensor:
        #  [-1, 1] -> [output_min, output_max]
        return (input_data + 1) * (self.output_max - self.output_min) / 2 + self.output_min

    def logprob(self, action_sample, pd_params, **kwargs):
        assert len (action_sample.shape) == 3 # B, T, D

        action_sample = self._normalize(action_sample)

        if action_sample.max() > 1.0 or action_sample.min() < -1.0:
            rich.print("[bold red]Action sample out of range, clipping to [-1, 1][/bold red]")
            action_sample = torch.clamp(action_sample, -1.0, 1.0)
            
        mixture_logits = pd_params["mixture_logits"] # B, T, K
        means = pd_params["means"].reshape(action_sample.shape[0], action_sample.shape[1], self.num_dimensions, self.num_mixtures) # B, T, D, K
        log_scales = pd_params["log_scales"].reshape(action_sample.shape[0], action_sample.shape[1], self.num_dimensions, self.num_mixtures) # B, T, D, K

        centered_action = action_sample.unsqueeze(-1) - means # B, T, D, K
        inv_stdv = torch.exp(-log_scales) # B, T, D, K

        plus_in = inv_stdv * (centered_action + 1. / (self.num_output_bins - 1))
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered_action - 1. / (self.num_output_bins - 1))
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - torch.nn.functional.softplus(plus_in) # log probability for edge case of 0 (before scaling)
        log_one_minus_cdf_min = - torch.nn.functional.softplus(min_in) # log probability for edge case of 255 (before scaling)
        cdf_delta = cdf_plus - cdf_min # probability for all other cases
        mid_in = inv_stdv * centered_action
        log_pdf_mid = mid_in - log_scales - 2. * torch.nn.functional.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)
        action_sample_expanded = action_sample.unsqueeze(-1).expand_as(means)
        log_probs = torch.where(
            action_sample_expanded < -0.999, log_cdf_plus, 
            torch.where(
                action_sample_expanded > 0.999, log_one_minus_cdf_min, 
                torch.where(cdf_delta > 1e-5, torch.log(torch.clamp_min(cdf_delta, 1e-12)), log_pdf_mid - np.log((self.num_output_bins - 1) / 2.0)
            )
        )) # type: ignore

        log_probs = log_probs.sum(-2) + torch.log_softmax(mixture_logits, dim=-1) # B, T, K
        return torch.logsumexp(log_probs, dim=-1) # B, T

    def entropy(self, pd_params) -> torch.Tensor:
        return torch.full_like(pd_params['mixture_logits'], torch.nan).sum(dim=-1)

    def sample(self, pd_params, deterministic: bool = False, **kwargs) -> torch.Tensor:
        if len (pd_params["means"].shape) < 3:
            cnt = 0
            while len (pd_params["means"].shape) < 3:
                pd_params = {k: v.unsqueeze(0) for k, v in pd_params.items()}
                cnt += 1
            ret = self.sample(pd_params, deterministic)
            for _ in range(cnt):
                ret = ret.squeeze(0)
            return ret

        assert len (pd_params["means"].shape) == 3 # B, T, D * K

        means = pd_params["means"].reshape(pd_params["means"].shape[0], pd_params["means"].shape[1], self.num_dimensions, self.num_mixtures) # B, T, D, K
        mixture_logits = pd_params["mixture_logits"] # B, T, K
        log_scales = pd_params["log_scales"].reshape(pd_params["log_scales"].shape[0], pd_params["log_scales"].shape[1], self.num_dimensions, self.num_mixtures) # B, T, D, K

        if deterministic:
            mixture_argmax = torch.argmax(mixture_logits, dim=-1) # B, T
            sampled_output = torch.gather(means, -1, mixture_argmax.unsqueeze(-1).expand(-1, -1, self.num_dimensions).unsqueeze(-1)).squeeze(-1) # B, T, D
        else :
            mixture_sample = torch.distributions.Categorical(logits=mixture_logits).sample()
            means_sample = torch.gather(means, -1, mixture_sample.unsqueeze(-1).expand(-1, -1, self.num_dimensions).unsqueeze(-1)).squeeze(-1) # B, T, D
            log_scales_sample = torch.gather(log_scales, -1, mixture_sample.unsqueeze(-1).expand(-1, -1, self.num_dimensions).unsqueeze(-1)).squeeze(-1) # B, T, D

            base_distribution = torch.distributions.Uniform(torch.zeros_like(means_sample), torch.ones_like(means_sample))
            transforms = [torch.distributions.SigmoidTransform().inv, torch.distributions.AffineTransform(loc=means_sample, scale=torch.exp(log_scales_sample))]
            logistic = torch.distributions.TransformedDistribution(base_distribution, transforms)

            sampled_output = logistic.rsample() # B, T, D
    
        return self._denormalize(sampled_output)

    def kl_divergence(self, params_q: torch.Tensor, params_p: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("KL divergence is not defined for MSE action head")

class JointActionHead(ActionHead):
    """Action head with joint action space"""

    def __init__(
        self, input_dim: int, shape: Tuple[int], num_actions: int, temperature: float = 1.0
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.output_shape = shape + (num_actions,)
        self.temperature = temperature

        assert len(shape) == 1, "Only 1D action spaces are supported when using JointActionSpace"
        self.embedding_layer = nn.Embedding(num_actions, input_dim)
        self.recurrent_layer = nn.GRU(input_dim, input_dim, num_layers=2, batch_first=True, dropout=0.5)
        self.linear_layer = nn.Linear(input_dim, num_actions)
        Console().log("[NOTICE] Using JointActionHead...")

    def reset_parameters(self):
        ...

    def forward(self, input_data: torch.Tensor, mask=None, actions=None, **kwargs) -> Any:
        
        # input_data: shape BxTxD
        # actions: shape BxTx7
        cod_feats = rearrange(input_data, 'b t d -> (b t) 1 d')
        act_feats = self.embedding_layer(actions) # BxTx7xD
        act_feats = rearrange(act_feats, 'b t n d -> (b t) n d') 
        inp_feats = torch.cat([cod_feats, act_feats[:, :-1, :]], dim=1)
        opt_feats, _ = self.recurrent_layer(inp_feats)
        opt_feats = self.linear_layer(opt_feats) # generate action logits
        
        shaped_opt = rearrange(opt_feats, '(b t) n d -> b t n d', b=input_data.shape[0], t=input_data.shape[1])
        shaped_opt = shaped_opt / self.temperature # remove the first timestep
        if mask is not None:
            shaped_opt[~mask] = LOG0
        # Convert to float32 to avoid RuntimeError: "log_softmax_lastdim_kernel_impl" not implemented for 'Half'
        return F.log_softmax(shaped_opt.float(), dim=-1)

    def logprob(self, actions: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        value = actions.long().unsqueeze(-1)
        value, log_pmf = torch.broadcast_tensors(value, logits)
        value = value[..., :1]
        result = log_pmf.gather(-1, value).squeeze(-1)
        # result is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            result = result.sum(dim=-1)
        return result

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Categorical distribution entropy calculation - sum probs * log(probs)"""
        probs = torch.exp(logits)
        entropy = -torch.sum(probs * logits, dim=-1)
        # entropy is per-entry, still of size self.output_shape[:-1]; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            entropy = entropy.sum(dim=-1)
        return entropy

    def sample(self, logits: torch.Tensor, pi_latent: torch.Tensor, deterministic: bool = False, **kwargs) -> Any:
        logits = None # we do not use the passed logits outside
        B, T = pi_latent.shape[:2]
        result = []
        input_feats = rearrange(pi_latent, "b t d -> (b t) 1 d")
        memory = None
        for i in range(self.output_shape[0]):
            logit_feats, memory = self.recurrent_layer(input_feats, memory)
            logits = self.linear_layer(logit_feats)
            if deterministic:
                next_token = torch.argmax(logits, dim=-1)
            else:
                logits = torch.nn.functional.log_softmax(logits, dim=-1)
                u = torch.rand_like(logits)
                u[u == 1.0] = 0.999
                next_token = torch.argmax(logits - torch.log(-torch.log(u)), dim=-1)
            result += [next_token]
            input_feats = self.embedding_layer(next_token)
        actions = rearrange(torch.cat(result, dim=1), "(b t) n -> b t n", b=B, t=T)
        return actions
    
    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        """
        Categorical distribution KL divergence calculation
        KL(Q || P) = sum Q_i log (Q_i / P_i)
        When talking about logits this is:
        sum exp(Q_i) * (Q_i - P_i)
        """
        kl = (torch.exp(logits_q) * (logits_q - logits_p)).sum(-1, keepdim=True)
        # kl is per-entry, still of size self.output_shape; we need to reduce of the rest of it.
        for _ in self.output_shape[:-1]:
            kl = kl.sum(dim=-2)  # dim=-2 because we use keepdim=True above.
        return kl


def make_action_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0, **kwargs):
    """Helper function to create an action head corresponding to the environment action space"""
    if isinstance(ac_space, gymnasium.spaces.MultiDiscrete):
        head_type = kwargs.get('type', 'independent').lower()
        if head_type == 'independent' or head_type == 'mse': #! mse is debug only
            return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.nvec[0].item(), temperature=temperature)
        elif head_type == 'joint':
            return JointActionHead(pi_out_size, ac_space.shape, ac_space.nvec[0].item(), temperature=temperature)
        else:
            raise NotImplementedError(f"Action head type {head_type} is not supported")
    elif isinstance(ac_space, gymnasium.spaces.Dict):
        return DictActionHead({k: make_action_head(v, pi_out_size, temperature) for k, v in ac_space.items()})
    elif isinstance(ac_space, gymnasium.spaces.Tuple):
        return TupleActionHead([make_action_head(v, pi_out_size, temperature) for v in ac_space])
    elif isinstance(ac_space, gym.spaces.Discrete):
        return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.n, temperature=temperature)
    elif isinstance(ac_space, gym.spaces.Box) or isinstance(ac_space, gymnasium.spaces.Box):
        assert len(ac_space.shape) == 1, "Nontrivial shapes not yet implemented."
        head_type = kwargs.get('type', 'mse').lower()
        if head_type == 'mse':
            return MSEActionHead(pi_out_size, ac_space.shape[0])
        elif head_type == 'dlml':
            return DLMLActionHead(pi_out_size, ac_space.shape[0], **kwargs['dlml_kwargs'])
        else:
            raise NotImplementedError(f"Action head type {head_type} is not supported")
        # return DLMLActionHead(pi_out_size, ac_space.shape[0], **kwargs['dlml_kwargs'])
        # return MSEActionHead(pi_out_size, ac_space.shape[0])
        # return DiagGaussianActionHead(pi_out_size, ac_space.shape[0])
        # return HLGaussActionHead(pi_out_size, ac_space.shape[0], **kwargs['hl_gauss_kwargs'])

    raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")

# def make_action_head(ac_space: ValType, pi_out_size: int, temperature: float = 1.0):
#     """Helper function to create an action head corresponding to the environment action space"""
#     if isinstance(ac_space, TensorType):
#         if isinstance(ac_space.eltype, Discrete):
#             return CategoricalActionHead(pi_out_size, ac_space.shape, ac_space.eltype.n, temperature=temperature)
#         elif isinstance(ac_space.eltype, Real):
#             if temperature != 1.0:
#                 logging.warning("Non-1 temperature not implemented for DiagGaussianActionHead.")
#             assert len(ac_space.shape) == 1, "Nontrivial shapes not yet implemented."
#             return DiagGaussianActionHead(pi_out_size, ac_space.shape[0])
#     elif isinstance(ac_space, DictType):
#         return DictActionHead({k: make_action_head(v, pi_out_size, temperature) for k, v in ac_space.items()})
        
#     raise NotImplementedError(f"Action space of type {type(ac_space)} is not supported")

if __name__ == '__main__':
    import ipdb
    import d4rl_atari
    
    env = gym.make("breakout-expert-v0")
    action_space = env.action_space
    
    action_head = make_action_head(action_space, 1024)
    
    g = torch.tensor([action_space.sample(), action_space.sample()])
    x = torch.rand((2, 1024))
    y = action_head(x)
    log_prob = action_head.logprob(g, y)
    ipdb.set_trace()
    
    
    