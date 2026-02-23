"""
Self-contained PPO implementation extracted from Stable Baselines 3.

This file contains a complete PPO implementation that only depends on:
- numpy
- torch
- gymnasium

It assumes the environment is already a VecEnv-like interface with:
- env.reset() -> obs (shape: [n_envs, *obs_shape])
- env.step(actions) -> obs, rewards, dones, infos
- env.num_envs -> int
- env.observation_space, env.action_space

Features:
- MLP and CNN policies
- Model saving/loading
- TensorBoard logging
"""

import io
import pathlib
import pickle
import time
import warnings
import zipfile
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch.nn import functional as F

# Optional TensorBoard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


# =============================================================================
# Type Aliases
# =============================================================================

Schedule = Callable[[float], float]
TensorDict = dict[str, th.Tensor]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


# =============================================================================
# Utility Functions
# =============================================================================


def get_device(device: th.device | str = "auto") -> th.device:
    """Retrieve PyTorch device."""
    if device == "auto":
        device = "cuda" if th.cuda.is_available() else "cpu"
    return th.device(device)


def set_random_seed(seed: int, using_cuda: bool = False) -> None:
    """Seed the random generators."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if using_cuda:
        th.backends.cudnn.deterministic = True
        th.backends.cudnn.benchmark = False


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Computes fraction of variance that ypred explains about y."""
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def obs_as_tensor(obs: np.ndarray | dict[str, np.ndarray], device: th.device) -> th.Tensor | TensorDict:
    """Moves the observation to the given device as a tensor."""
    if isinstance(obs, np.ndarray):
        return th.as_tensor(obs, device=device)
    elif isinstance(obs, dict):
        return {key: th.as_tensor(_obs, device=device) for (key, _obs) in obs.items()}
    else:
        raise TypeError(f"Unrecognized type of observation {type(obs)}")


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """Update the learning rate for a given optimizer."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


class ConstantSchedule:
    """Constant schedule that always returns the same value."""
    def __init__(self, val: float):
        self.val = val

    def __call__(self, _: float) -> float:
        return self.val


class FloatSchedule:
    """Wrapper that ensures the output of a Schedule is cast to float."""
    def __init__(self, value_schedule: Schedule | float):
        if isinstance(value_schedule, FloatSchedule):
            self.value_schedule: Schedule = value_schedule.value_schedule
        elif isinstance(value_schedule, (float, int)):
            self.value_schedule = ConstantSchedule(float(value_schedule))
        else:
            assert callable(value_schedule)
            self.value_schedule = value_schedule

    def __call__(self, progress_remaining: float) -> float:
        return float(self.value_schedule(progress_remaining))


# =============================================================================
# Preprocessing Functions
# =============================================================================


def is_image_space(
    observation_space: spaces.Space,
    check_channels: bool = False,
    normalized_image: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype of a valid image.
    
    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    
    :param observation_space: Observation space
    :param check_channels: Whether to check for the number of channels
    :param normalized_image: Whether to assume image is already normalized
    :return: True if observation space is an image space
    """
    check_dtype = check_bounds = not normalized_image
    if isinstance(observation_space, spaces.Box) and len(observation_space.shape) == 3:
        if check_dtype and observation_space.dtype != np.uint8:
            return False
        if check_bounds and (np.any(observation_space.low != 0) or np.any(observation_space.high != 255)):
            return False
        if not check_channels:
            return True
        # Check channels (first or last dimension)
        n_channels = min(observation_space.shape[0], observation_space.shape[-1])
        return n_channels in [1, 3, 4]
    return False


def is_image_space_channels_first(observation_space: spaces.Box) -> bool:
    """Check if an image observation space is channels-first (CxHxW)."""
    smallest_dimension = np.argmin(observation_space.shape).item()
    return smallest_dimension == 0


def get_obs_shape(observation_space: spaces.Space) -> tuple[int, ...]:
    """Get the shape of the observation (useful for the buffers)."""
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        return (len(observation_space.nvec),)
    elif isinstance(observation_space, spaces.MultiBinary):
        return observation_space.shape
    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    """Get the dimension of the observation space when flattened."""
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        return spaces.utils.flatdim(observation_space)


def get_action_dim(action_space: spaces.Space) -> int:
    """Get the dimension of the action space."""
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(action_space.n, int)
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def preprocess_obs(
    obs: th.Tensor,
    observation_space: spaces.Space,
    normalize_images: bool = True,
) -> th.Tensor:
    """Preprocess observation to be fed to a neural network."""
    if isinstance(observation_space, spaces.Box):
        if normalize_images and is_image_space(observation_space):
            return obs.float() / 255.0
        return obs.float()
    elif isinstance(observation_space, spaces.Discrete):
        return F.one_hot(obs.long(), num_classes=int(observation_space.n)).float()
    elif isinstance(observation_space, spaces.MultiDiscrete):
        return th.cat(
            [
                F.one_hot(obs_.long(), num_classes=int(observation_space.nvec[idx])).float()
                for idx, obs_ in enumerate(th.split(obs.long(), 1, dim=1))
            ],
            dim=-1,
        ).view(obs.shape[0], sum(observation_space.nvec))
    elif isinstance(observation_space, spaces.MultiBinary):
        return obs.float()
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {observation_space}")


# =============================================================================
# Distributions
# =============================================================================


def sum_independent_dims(tensor: th.Tensor) -> th.Tensor:
    """Sum components of the log_prob or entropy for continuous actions."""
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor


class Distribution(ABC):
    """Abstract base class for distributions."""

    @abstractmethod
    def proba_distribution_net(self, *args, **kwargs) -> nn.Module | tuple[nn.Module, nn.Parameter]:
        pass

    @abstractmethod
    def proba_distribution(self, *args, **kwargs) -> "Distribution":
        pass

    @abstractmethod
    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        pass

    @abstractmethod
    def entropy(self) -> th.Tensor | None:
        pass

    @abstractmethod
    def sample(self) -> th.Tensor:
        pass

    @abstractmethod
    def mode(self) -> th.Tensor:
        pass

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()


class DiagGaussianDistribution(Distribution):
    """Gaussian distribution with diagonal covariance matrix, for continuous actions."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Normal | None = None

    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        mean_actions = nn.Linear(latent_dim, self.action_dim)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std

    def proba_distribution(self, mean_actions: th.Tensor, log_std: th.Tensor) -> "DiagGaussianDistribution":
        action_std = th.ones_like(mean_actions) * log_std.exp()
        self.distribution = Normal(mean_actions, action_std)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None
        return sum_independent_dims(self.distribution.entropy())

    def sample(self) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.rsample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.mean


class CategoricalDistribution(Distribution):
    """Categorical distribution for discrete actions."""

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.distribution: Categorical | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy()

    def sample(self) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None
        return th.argmax(self.distribution.probs, dim=1)


class MultiCategoricalDistribution(Distribution):
    """MultiCategorical distribution for multi discrete actions."""

    def __init__(self, action_dims: list[int]):
        super().__init__()
        self.action_dims = action_dims
        self.distributions: list[Categorical] = []

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        action_logits = nn.Linear(latent_dim, sum(self.action_dims))
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "MultiCategoricalDistribution":
        self.distributions = [
            Categorical(logits=split) for split in th.split(action_logits, list(self.action_dims), dim=1)
        ]
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return th.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, th.unbind(actions, dim=1), strict=True)],
            dim=1,
        ).sum(dim=1)

    def entropy(self) -> th.Tensor:
        return th.stack([dist.entropy() for dist in self.distributions], dim=1).sum(dim=1)

    def sample(self) -> th.Tensor:
        return th.stack([dist.sample() for dist in self.distributions], dim=1)

    def mode(self) -> th.Tensor:
        return th.stack([th.argmax(dist.probs, dim=1) for dist in self.distributions], dim=1)


class BernoulliDistribution(Distribution):
    """Bernoulli distribution for MultiBinary action spaces."""

    def __init__(self, action_dims: int):
        super().__init__()
        self.action_dims = action_dims
        self.distribution: Bernoulli | None = None

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        action_logits = nn.Linear(latent_dim, self.action_dims)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "BernoulliDistribution":
        self.distribution = Bernoulli(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=1)

    def entropy(self) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=1)

    def sample(self) -> th.Tensor:
        assert self.distribution is not None
        return self.distribution.sample()

    def mode(self) -> th.Tensor:
        assert self.distribution is not None
        return th.round(self.distribution.probs)


def make_proba_distribution(action_space: spaces.Space) -> Distribution:
    """Return an instance of Distribution for the correct type of action space."""
    if isinstance(action_space, spaces.Box):
        return DiagGaussianDistribution(get_action_dim(action_space))
    elif isinstance(action_space, spaces.Discrete):
        return CategoricalDistribution(int(action_space.n))
    elif isinstance(action_space, spaces.MultiDiscrete):
        return MultiCategoricalDistribution(list(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        assert isinstance(action_space.n, int)
        return BernoulliDistribution(action_space.n)
    else:
        raise NotImplementedError(f"Action space {type(action_space)} not supported")


# =============================================================================
# Neural Network Layers
# =============================================================================


class BaseFeaturesExtractor(nn.Module):
    """Base class for features extractors."""

    def __init__(self, observation_space: spaces.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim


class FlattenExtractor(BaseFeaturesExtractor):
    """Feature extractor that flattens the input."""

    def __init__(self, observation_space: spaces.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.flatten(observations)


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted (default: 512)
    :param normalized_image: Whether to assume image is already normalized
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            f"NatureCNN must be used with a gym.spaces.Box observation space, not {observation_space}"
        )
        super().__init__(observation_space, features_dim)
        
        # We assume CxHxW images (channels first)
        assert is_image_space(observation_space, normalized_image=normalized_image), (
            f"NatureCNN should only be used with images, not with {observation_space}"
        )
        
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class MlpExtractor(nn.Module):
    """
    Constructs an MLP for the policy and value networks.
    
    :param feature_dim: Dimension of the feature vector
    :param net_arch: Network architecture specification (dict with 'pi' and 'vf' keys or list)
    :param activation_fn: Activation function
    :param device: PyTorch device
    """

    def __init__(
        self,
        feature_dim: int,
        net_arch: list[int] | dict[str, list[int]],
        activation_fn: type[nn.Module],
        device: th.device | str = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        policy_net: list[nn.Module] = []
        value_net: list[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", [])
            vf_layers_dims = net_arch.get("vf", [])
        else:
            pi_layers_dims = vf_layers_dims = net_arch

        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim

        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


# =============================================================================
# Actor-Critic Policy
# =============================================================================


class ActorCriticPolicy(nn.Module):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule
    :param net_arch: Network architecture
    :param activation_fn: Activation function
    :param ortho_init: Whether to use orthogonal initialization
    :param log_std_init: Initial value for log standard deviation
    :param features_extractor_class: Features extractor class (FlattenExtractor or NatureCNN)
    :param features_extractor_kwargs: Keyword arguments for features extractor
    :param normalize_images: Whether to normalize images (divide by 255)
    :param optimizer_class: Optimizer class
    :param optimizer_kwargs: Optimizer keyword arguments
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        log_std_init: float = 0.0,
        features_extractor_class: type[BaseFeaturesExtractor] | None = None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.ortho_init = ortho_init
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        self.normalize_images = normalize_images

        # Determine features extractor
        if features_extractor_class is None:
            # Auto-select based on observation space
            if is_image_space(observation_space):
                features_extractor_class = NatureCNN
            else:
                features_extractor_class = FlattenExtractor
        
        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs

        # Default network architecture
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []  # No additional MLP layers for CNN
            else:
                net_arch = dict(pi=[64, 64], vf=[64, 64])
        self.net_arch = net_arch

        # Features extractor
        self.features_extractor = features_extractor_class(observation_space, **features_extractor_kwargs)
        self.features_dim = self.features_extractor.features_dim

        # Action distribution
        self.action_dist = make_proba_distribution(action_space)

        self._build(lr_schedule)

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        """Orthogonal initialization."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def _build(self, lr_schedule: Schedule) -> None:
        """Create the networks and the optimizer."""
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
            self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        if self.ortho_init:
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def device(self) -> th.device:
        """Infer which device this policy lives on."""
        for param in self.parameters():
            return param.device
        return get_device("cpu")

    def set_training_mode(self, mode: bool) -> None:
        """Put the policy in training or evaluation mode."""
        self.train(mode)

    def extract_features(self, obs: th.Tensor) -> th.Tensor:
        """Preprocess the observation and extract features."""
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return self.features_extractor(preprocessed_obs)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """Retrieve action distribution given the latent codes."""
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        else:
            raise ValueError("Invalid action distribution")

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass in all the networks (actor and critic)."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        """Evaluate actions according to the current policy."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Get the estimated values according to the current policy."""
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get the policy action from an observation."""
        self.set_training_mode(False)
        obs_tensor = th.as_tensor(observation, device=self.device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with th.no_grad():
            actions, _, _ = self.forward(obs_tensor, deterministic=deterministic)
        actions = actions.cpu().numpy()
        if isinstance(self.action_space, spaces.Box):
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
        return actions


# =============================================================================
# Rollout Buffer
# =============================================================================


class RolloutBuffer:
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    
    :param buffer_size: Max number of elements in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for GAE
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)
        self.action_dim = get_action_dim(action_space)
        self.device = get_device(device)
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.pos = 0
        self.full = False
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """Compute the lambda-return (TD(lambda) estimate) and GAE(lambda) advantage."""
        last_values = last_values.clone().cpu().numpy().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """Swap and flatten axes 0 and 1."""
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """Convert a numpy array to a PyTorch tensor."""
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "Buffer must be full before sampling"
        indices = np.random.permutation(self.buffer_size * self.n_envs)

        if not self.generator_ready:
            _tensor_names = ["observations", "actions", "values", "log_probs", "advantages", "returns"]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds: np.ndarray) -> RolloutBufferSamples:
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(*tuple(map(self.to_torch, data)))


# =============================================================================
# PPO Algorithm
# =============================================================================


class PPO:
    """
    Proximal Policy Optimization algorithm (PPO) (clip version).
    
    Paper: https://arxiv.org/abs/1707.06347
    
    :param env: The environment to learn from (VecEnv-like interface)
    :param learning_rate: The learning rate
    :param n_steps: Number of steps to run for each environment per update
    :param batch_size: Minibatch size
    :param n_epochs: Number of epochs when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for GAE
    :param clip_range: Clipping parameter
    :param clip_range_vf: Clipping parameter for the value function
    :param normalize_advantage: Whether to normalize the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for gradient clipping
    :param target_kl: Limit the KL divergence between updates
    :param tensorboard_log: Path for TensorBoard logs (None to disable)
    :param seed: Seed for the pseudo random generators
    :param device: Device on which the code should run
    :param policy_kwargs: Additional arguments for the policy
    :param verbose: Verbosity level
    """

    def __init__(
        self,
        env,  # VecEnv-like
        learning_rate: float | Schedule = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: float | Schedule | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        tensorboard_log: str | None = None,
        seed: int | None = None,
        device: th.device | str = "auto",
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        self.env = env
        self.n_envs = env.num_envs
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.device = get_device(device)
        self.verbose = verbose
        self.tensorboard_log = tensorboard_log

        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.seed = seed
        self.policy_kwargs = policy_kwargs or {}

        self.num_timesteps = 0
        self._n_updates = 0
        self._current_progress_remaining = 1.0
        self._start_time: float = 0.0
        self._tb_writer: "SummaryWriter | None" = None

        # Sanity check
        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1"

        buffer_size = self.n_envs * self.n_steps
        assert buffer_size > 1 or not normalize_advantage

        self._setup_model()

    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""
        self.lr_schedule = FloatSchedule(self.learning_rate)

        if self.seed is not None:
            set_random_seed(self.seed, using_cuda=self.device.type == "cuda")

        self.rollout_buffer = RolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.policy = ActorCriticPolicy(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def _update_learning_rate(self) -> None:
        """Update the optimizer learning rate."""
        lr = self.lr_schedule(self._current_progress_remaining)
        update_learning_rate(self.policy.optimizer, lr)

    def collect_rollouts(self) -> bool:
        """
        Collect experiences using the current policy and fill the rollout buffer.
        
        :return: True if collection was successful
        """
        assert self._last_obs is not None

        self.policy.set_training_mode(False)
        n_steps = 0
        self.rollout_buffer.reset()

        while n_steps < self.n_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            # Clip actions for continuous action spaces
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions

            new_obs, rewards, dones, infos = self.env.step(clipped_actions)

            self.num_timesteps += self.n_envs
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstrapping with value function
            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = th.as_tensor(infos[idx]["terminal_observation"], device=self.device).unsqueeze(0)
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value.cpu().numpy()

            self.rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        self.rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def train(self) -> dict[str, float]:
        """Update policy using the currently gathered rollout buffer."""
        self.policy.set_training_mode(True)
        self._update_learning_rate()

        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        approx_kl_divs = []

        continue_training = True

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at epoch {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        return {
            "entropy_loss": np.mean(entropy_losses),
            "policy_gradient_loss": np.mean(pg_losses),
            "value_loss": np.mean(value_losses),
            "approx_kl": np.mean(approx_kl_divs),
            "clip_fraction": np.mean(clip_fractions),
            "loss": loss.item(),
            "explained_variance": explained_var,
            "n_updates": self._n_updates,
            "clip_range": clip_range,
            "learning_rate": self.lr_schedule(self._current_progress_remaining),
        }

    def _setup_tensorboard(self, tb_log_name: str = "PPO") -> None:
        """Setup TensorBoard logging."""
        if self.tensorboard_log is not None and TENSORBOARD_AVAILABLE:
            # Find latest run number
            log_path = pathlib.Path(self.tensorboard_log)
            log_path.mkdir(parents=True, exist_ok=True)
            
            run_num = 1
            for existing in log_path.iterdir():
                if existing.is_dir() and existing.name.startswith(f"{tb_log_name}_"):
                    try:
                        num = int(existing.name.split("_")[-1])
                        run_num = max(run_num, num + 1)
                    except ValueError:
                        pass
            
            self._tb_writer = SummaryWriter(str(log_path / f"{tb_log_name}_{run_num}"))
            if self.verbose >= 1:
                print(f"TensorBoard logging to: {log_path / f'{tb_log_name}_{run_num}'}")
        elif self.tensorboard_log is not None and not TENSORBOARD_AVAILABLE:
            warnings.warn("TensorBoard not available. Install with: pip install tensorboard")

    def _log_to_tensorboard(self, train_info: dict[str, float], iteration: int) -> None:
        """Log training metrics to TensorBoard."""
        if self._tb_writer is not None:
            for key, value in train_info.items():
                if isinstance(value, (int, float)):
                    self._tb_writer.add_scalar(f"train/{key}", value, self.num_timesteps)
            
            # Log FPS
            elapsed = time.time() - self._start_time
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            self._tb_writer.add_scalar("time/fps", fps, self.num_timesteps)
            self._tb_writer.add_scalar("time/iterations", iteration, self.num_timesteps)

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "PPO",
    ) -> "PPO":
        """
        Train the PPO agent.
        
        :param total_timesteps: Total number of timesteps to train
        :param log_interval: Number of iterations between logs
        :param reset_num_timesteps: Whether to reset the timestep counter
        :param tb_log_name: Name for the TensorBoard run
        :return: The trained PPO agent
        """
        iteration = 0
        self._start_time = time.time()

        if reset_num_timesteps:
            self.num_timesteps = 0

        # Setup TensorBoard
        self._setup_tensorboard(tb_log_name)

        # Handle both gym (obs, info) and vec_env (obs) reset returns
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            self._last_obs = reset_result[0]
        else:
            self._last_obs = reset_result
        self._last_episode_starts = np.ones((self.n_envs,), dtype=bool)

        while self.num_timesteps < total_timesteps:
            self.collect_rollouts()

            iteration += 1
            self._current_progress_remaining = 1.0 - float(self.num_timesteps) / float(total_timesteps)

            train_info = self.train()

            # Log to TensorBoard
            self._log_to_tensorboard(train_info, iteration)

            if log_interval is not None and log_interval > 0 and iteration % log_interval == 0:
                elapsed = time.time() - self._start_time
                fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
                print(f"Iteration {iteration}, Timesteps: {self.num_timesteps}, FPS: {fps}")
                for key, value in train_info.items():
                    if key != "n_updates":
                        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        # Close TensorBoard writer
        if self._tb_writer is not None:
            self._tb_writer.close()

        return self

    def save(self, path: str | pathlib.Path) -> None:
        """
        Save model to a zip file.
        
        :param path: Path to save the model
        """
        path = pathlib.Path(path)
        if not path.suffix:
            path = path.with_suffix(".zip")
        
        # Data to save (excluding non-serializable objects)
        data = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "n_envs": self.n_envs,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "normalize_advantage": self.normalize_advantage,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "target_kl": self.target_kl,
            "policy_kwargs": self.policy_kwargs,
            "verbose": self.verbose,
            "num_timesteps": self.num_timesteps,
            "_n_updates": self._n_updates,
        }
        
        # Save policy state dict
        policy_state = self.policy.state_dict()
        optimizer_state = self.policy.optimizer.state_dict()
        
        # Write to zip file
        with zipfile.ZipFile(path, "w") as archive:
            # Save data as pickle
            archive.writestr("data.pkl", pickle.dumps(data))
            
            # Save policy weights
            policy_buffer = io.BytesIO()
            th.save(policy_state, policy_buffer)
            archive.writestr("policy.pth", policy_buffer.getvalue())
            
            # Save optimizer state
            optimizer_buffer = io.BytesIO()
            th.save(optimizer_state, optimizer_buffer)
            archive.writestr("optimizer.pth", optimizer_buffer.getvalue())
        
        if self.verbose >= 1:
            print(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        env=None,
        device: th.device | str = "auto",
        **kwargs,
    ) -> "PPO":
        """
        Load model from a zip file.
        
        :param path: Path to the saved model
        :param env: Environment to use (required for loading)
        :param device: Device to load the model on
        :param kwargs: Additional arguments to override saved parameters
        :return: Loaded PPO model
        """
        path = pathlib.Path(path)
        if not path.suffix:
            path = path.with_suffix(".zip")
        
        # Resolve device
        device = get_device(device)
        
        with zipfile.ZipFile(path, "r") as archive:
            # Load data
            data = pickle.loads(archive.read("data.pkl"))
            
            # Load policy weights
            policy_buffer = io.BytesIO(archive.read("policy.pth"))
            policy_state = th.load(policy_buffer, map_location=device, weights_only=True)
            
            # Load optimizer state
            optimizer_buffer = io.BytesIO(archive.read("optimizer.pth"))
            optimizer_state = th.load(optimizer_buffer, map_location=device, weights_only=True)
        
        # Check environment
        if env is None:
            raise ValueError("Environment must be provided to load the model")
        
        # Verify spaces match
        if env.observation_space != data["observation_space"]:
            warnings.warn(f"Observation spaces don't match: {env.observation_space} vs {data['observation_space']}")
        if env.action_space != data["action_space"]:
            warnings.warn(f"Action spaces don't match: {env.action_space} vs {data['action_space']}")
        
        # Create model with loaded parameters
        model = cls(
            env=env,
            learning_rate=data["learning_rate"],
            n_steps=data["n_steps"],
            batch_size=data["batch_size"],
            n_epochs=data["n_epochs"],
            gamma=data["gamma"],
            gae_lambda=data["gae_lambda"],
            normalize_advantage=data["normalize_advantage"],
            ent_coef=data["ent_coef"],
            vf_coef=data["vf_coef"],
            max_grad_norm=data["max_grad_norm"],
            target_kl=data["target_kl"],
            policy_kwargs=data["policy_kwargs"],
            verbose=data["verbose"],
            device=device,
            **kwargs,
        )
        
        # Load policy and optimizer state
        model.policy.load_state_dict(policy_state)
        model.policy.optimizer.load_state_dict(optimizer_state)
        
        # Restore training state
        model.num_timesteps = data["num_timesteps"]
        model._n_updates = data["_n_updates"]
        
        return model

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Get the policy action from an observation."""
        return self.policy.predict(observation, deterministic=deterministic)


# =============================================================================
# VecEnv Wrapper for Gymnasium
# =============================================================================


class GymVecEnvWrapper:
    """
    Wrapper to adapt gymnasium's SyncVectorEnv to SB3-like VecEnv interface.
    
    SB3's VecEnv expects:
    - observation_space/action_space: Single env spaces (not batched)
    - num_envs: Number of parallel environments
    - reset() -> obs (np.ndarray)
    - step(actions) -> (obs, rewards, dones, infos)
    
    Gymnasium's SyncVectorEnv provides:
    - observation_space/action_space: Batched spaces
    - single_observation_space/single_action_space: Single env spaces
    - reset() -> (obs, info)
    - step(actions) -> (obs, rewards, terminated, truncated, infos)
    """
    
    def __init__(self, vec_env):
        self._vec_env = vec_env
        # Use single env spaces (not batched)
        self.observation_space = vec_env.single_observation_space
        self.action_space = vec_env.single_action_space
        self.num_envs = vec_env.num_envs

    def reset(self):
        obs, info = self._vec_env.reset()
        return obs

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = self._vec_env.step(actions)
        # Combine terminated and truncated into dones
        dones = np.logical_or(terminated, truncated)
        # Convert infos dict format to list of dicts and add TimeLimit.truncated
        processed_infos = []
        for i in range(self.num_envs):
            info = {k: v[i] if isinstance(v, np.ndarray) else v for k, v in infos.items()}
            if truncated[i]:
                info["TimeLimit.truncated"] = True
                # Gymnasium stores terminal observation in final_observation
                if "final_observation" in infos and infos["final_observation"][i] is not None:
                    info["terminal_observation"] = infos["final_observation"][i]
            processed_infos.append(info)
        return obs, rewards, dones, processed_infos

    def close(self):
        self._vec_env.close()


# =============================================================================
# Example Usage
# =============================================================================


if __name__ == "__main__":
    import gymnasium as gym
    from gymnasium.vector import SyncVectorEnv

    # Create a vectorized environment
    def make_env():
        return gym.make("CartPole-v1")

    n_envs = 4
    gym_vec_env = SyncVectorEnv([make_env for _ in range(n_envs)])
    env = GymVecEnvWrapper(gym_vec_env)

    # Create and train PPO
    ppo = PPO(
        env=env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
    )

    ppo.learn(total_timesteps=10000, log_interval=1)

    # Test the trained policy
    test_env = gym.make("CartPole-v1")
    obs, _ = test_env.reset()
    total_reward = 0
    done = False
    while not done:
        action = ppo.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = test_env.step(action[0])
        done = terminated or truncated
        total_reward += reward

    print(f"\nTest episode reward: {total_reward}")
    test_env.close()
    env.close()
