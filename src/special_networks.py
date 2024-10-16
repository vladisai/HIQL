from jaxrl_m.dataset import Dataset
from jaxrl_m.typing import *
from jaxrl_m.networks import *
import jax
from flax.core.frozen_dict import FrozenDict


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    activate_final: int = False
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                x = nn.LayerNorm()(x)
        return x


class LayerNormRepresentation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = LayerNormMLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(self.hidden_dims, activate_final=self.activate_final)(
            observations
        )


class Representation(nn.Module):
    hidden_dims: tuple = (256, 256)
    activate_final: bool = True
    ensemble: bool = True

    @nn.compact
    def __call__(self, observations):
        module = MLP
        if self.ensemble:
            module = ensemblize(module, 2)
        return module(
            self.hidden_dims, activate_final=self.activate_final, activations=nn.gelu
        )(observations)


class RelativeRepresentation(nn.Module):
    rep_dim: int = 256
    hidden_dims: tuple = (256, 256)
    module: nn.Module = None
    rep_encoder: nn.Module = None
    visual: bool = False
    layer_norm: bool = False
    rep_type: str = "state"
    bottleneck: bool = (
        True  # Meaning that we're using this representation for high-level actions
    )

    @nn.compact
    def __call__(self, targets, bases=None):
        if self.rep_encoder is not None:
            targets = self.rep_encoder()(targets)
            if bases is not None:
                bases = self.rep_encoder()(bases)
            else:
                return targets

        if bases is None:
            inputs = targets
        else:
            if self.rep_type == "state":
                inputs = targets
            elif self.rep_type == "diff":
                inputs = jax.tree_map(
                    lambda t, b: t - b + jnp.ones_like(t) * 1e-6, targets, bases
                )
            elif self.rep_type == "concat":
                inputs = jax.tree_map(
                    lambda t, b: jnp.concatenate([t, b], axis=-1), targets, bases
                )
            else:
                raise NotImplementedError

        if self.visual:
            inputs = self.module()(inputs)
        if self.layer_norm:
            rep = LayerNormMLP(
                self.hidden_dims,
                activate_final=not self.bottleneck,
                activations=nn.gelu,
            )(inputs)
        else:
            rep = MLP(
                self.hidden_dims,
                activate_final=not self.bottleneck,
                activations=nn.gelu,
            )(inputs)

        if self.bottleneck:
            rep = (
                rep
                / jnp.linalg.norm(rep, axis=-1, keepdims=True)
                * jnp.sqrt(self.rep_dim)
            )

        return rep


class MonolithicVF(nn.Module):
    hidden_dims: tuple = (256, 256)
    readout_size: tuple = (256,)
    use_layer_norm: bool = True
    rep_dim: int = None
    obs_rep: int = 0

    def setup(self) -> None:
        repr_class = LayerNormRepresentation if self.use_layer_norm else Representation
        self.value_net = repr_class((*self.hidden_dims, 1), activate_final=False)

    def __call__(self, observations, goals=None, info=False):
        phi = observations
        psi = goals

        v1, v2 = self.value_net(jnp.concatenate([phi, psi], axis=-1)).squeeze(-1)

        if info:
            return {
                "v": (v1 + v2) / 2,
            }
        return v1, v2


def get_rep(
    encoder: nn.Module,
    targets: jnp.ndarray,
    bases: jnp.ndarray = None,
):
    if encoder is None:
        return targets
    else:
        if bases is None:
            return encoder(targets)
        else:
            return encoder(targets, bases)


class OneLayer(nn.Module):
    def setup(self):
        self.layer1 = nn.Dense(512)
        self.activations = nn.gelu
        self.ln = nn.LayerNorm()
        self.layer2 = nn.Dense(10)

    @nn.compact
    def __call__(self, x):
        x1 = self.layer1(x)
        x2 = self.activations(x1)
        x3 = self.ln(x2)
        x4 = self.layer2(x3)
        return x1, x2, x3, x4


class HierarchicalActorCritic(nn.Module):
    encoders: Dict[str, nn.Module]
    networks: Dict[str, nn.Module]
    use_waypoints: int
    # one_layer_net: nn.Module

    def value(self, observations, goals, **kwargs):
        if self.encoders["rep"] is not None:
            observations = self.encoders["rep"](observations)
            # goals = self.encoders["rep"](goals)

        state_reps = get_rep(self.encoders["value_state"], targets=observations)
        goal_reps = get_rep(
            self.encoders["value_goal"], targets=goals, bases=observations
        )
        return self.networks["value"](state_reps, goal_reps, **kwargs)

    def one_layer(self, observations):
        state_reps = get_rep(self.encoders["value_state"], targets=observations)
        return self.one_layer_net(state_reps)

    def value_state(self, observations, **kwargs):
        return get_rep(self.encoders["value_state"], targets=observations)

    def target_value(self, observations, goals, **kwargs):
        if self.encoders["rep"] is not None:
            observations = self.encoders["rep"](observations)
            # goals = self.encoders["rep"](goals)

        state_reps = get_rep(self.encoders["value_state"], targets=observations)
        goal_reps = get_rep(
            self.encoders["value_goal"], targets=goals, bases=observations
        )
        return self.networks["target_value"](state_reps, goal_reps, **kwargs)

    def actor(
        self,
        observations,
        goals,
        low_dim_goals=False,
        state_rep_grad=True,
        goal_rep_grad=True,
        **kwargs,
    ):
        if self.encoders["rep"] is not None:
            observations = self.encoders["rep"](observations)
            # goals = self.encoders["rep"](goals)

        state_reps = get_rep(self.encoders["policy_state"], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        if low_dim_goals:
            goal_reps = goals
        else:
            if self.use_waypoints:
                # Use the value_goal representation
                goal_reps = get_rep(
                    self.encoders["value_goal"], targets=goals, bases=observations
                )
            else:
                goal_reps = get_rep(
                    self.encoders["policy_goal"], targets=goals, bases=observations
                )
            if not goal_rep_grad:
                goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks["actor"](
            jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs
        )

    def high_actor(
        self, observations, goals, state_rep_grad=True, goal_rep_grad=True, **kwargs
    ):
        if self.encoders["rep"] is not None:
            observations = self.encoders["rep"](observations)
            # goals = self.encoders["rep"](goals)

        state_reps = get_rep(self.encoders["high_policy_state"], targets=observations)
        if not state_rep_grad:
            state_reps = jax.lax.stop_gradient(state_reps)

        goal_reps = get_rep(
            self.encoders["high_policy_goal"], targets=goals, bases=observations
        )
        if not goal_rep_grad:
            goal_reps = jax.lax.stop_gradient(goal_reps)

        return self.networks["high_actor"](
            jnp.concatenate([state_reps, goal_reps], axis=-1), **kwargs
        )

    def value_goal_encoder(self, targets, bases, **kwargs):
        # if self.encoders["rep"] is not None:
        #     targets = self.encoders["rep"](targets)
        # bases = self.encoders["rep"](bases)

        return get_rep(self.encoders["value_goal"], targets=targets, bases=bases)

    def policy_goal_encoder(self, targets, bases, **kwargs):
        if self.encoders["rep"] is not None:
            targets = self.encoders["rep"](targets)
            bases = self.encoders["rep"](bases)

        assert not self.use_waypoints
        return get_rep(self.encoders["policy_goal"], targets=targets, bases=bases)

    def __call__(self, observations, goals):
        # Only for initialization
        if self.encoders["rep"] is not None:
            observations_encs = self.encoders["rep"](observations)
            goals_encs = self.encoders["rep"](goals)
        else:
            observations_encs = observations
            goals_encs = goals

        rets = {
            "value": self.value(observations, goals),
            "target_value": self.target_value(observations, goals),
            "actor": self.actor(observations, goals),
            "high_actor": self.high_actor(observations, goals),
            "value_state": self.value_state(observations),
            # "one_layer": self.one_layer(observations),
        }
        return rets


@jax.jit
def simnorm(x: jnp.ndarray, dim: int = 8) -> jnp.ndarray:
    """
    Simplicial normalization function.

    Adapted from https://arxiv.org/abs/2204.00616.

    Reshapes the input to split the last dimension and applies softmax
    along the last axis (simplicial normalization).

    Args:
        x: Input array.
        dim: The dimensionality for the softmax normalization.

    Returns:
        The normalized array.
    """
    # Get the shape of the input
    shp = x.shape
    # Reshape the input array so that the last dimension is split
    x = x.reshape(*shp[:-1], -1, dim)
    # Apply softmax normalization along the last axis
    x = jax.nn.softmax(x, axis=-1)
    # Reshape it back to the original shape
    return x.reshape(*shp)


@jax.jit
def softplus(x: jnp.ndarray) -> jnp.ndarray:
    r"""Softplus activation function.

    Computes the element-wise function

    .. math::
      \mathrm{softplus}(x) = \log(1 + e^x)

    Args:
      x : input array
    """
    return jnp.logaddexp(x, 0)


@jax.jit
def mish(x: jnp.ndarray) -> jnp.ndarray:
    r"""Mish activation function.

    Computes the element-wise function:

    .. math::
      \mathrm{mish}(x) = x \cdot \mathrm{tanh}(\mathrm{softplus}(x))

    For more information, see
    `Mish: A Self Regularized Non-Monotonic Activation Function
    <https://arxiv.org/abs/1908.08681>`_.

    Args:
      x : input array

    Returns:
      An array.
    """
    # numpy_util.check_arraylike("mish", x)
    x_arr = jnp.asarray(x)
    return x_arr * jnp.tanh(softplus(x_arr))


class FrozenTDMPCEncoder(nn.Module):
    frozen_params: FrozenDict  # Add frozen parameters here
    activations: Callable[[jnp.ndarray], jnp.ndarray] = mish
    final_act: Callable[[jnp.ndarray], jnp.ndarray] = simnorm

    def setup(self):
        # Hardcoding the MLP layers for 39 -> 256 -> 512
        self.dense1 = nn.Dense(
            256, name="frozen_dense1"
        )  # First hidden layer: 39 -> 256
        self.layer_norm1 = nn.LayerNorm(
            epsilon=1e-5, name="frozen_layer_norm1"
        )  # Layer norm after the first layer

        self.dense2 = nn.Dense(
            512, name="frozen_dense2"
        )  # Second hidden layer: 256 -> 512
        self.layer_norm2 = nn.LayerNorm(
            epsilon=1e-5, name="frozen_layer_norm2"
        )  # Layer norm after the second layer

        self.frozen_dense1_params = self.frozen_params["frozen_dense1"]
        self.frozen_dense2_params = self.frozen_params["frozen_dense2"]
        self.frozen_layer_norm1_params = self.frozen_params["frozen_layer_norm1"]
        self.frozen_layer_norm2_params = self.frozen_params["frozen_layer_norm2"]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Apply the frozen weights in the first layer: 39 -> 256
        x = self.dense1.apply({"params": self.frozen_dense1_params}, x)
        x = self.layer_norm1.apply({"params": self.frozen_layer_norm1_params}, x)
        x = self.activations(x)  # Apply activation

        # Apply the frozen weights in the second layer: 256 -> 512
        x = self.dense2.apply({"params": self.frozen_dense2_params}, x)
        x = self.layer_norm2.apply({"params": self.frozen_layer_norm2_params}, x)
        x = self.final_act(x)  # Apply the final activation

        return x

    @classmethod
    def load_frozen_weights(cls, path: str) -> FrozenDict:
        """
        Class method to load frozen weights from a file.

        Args:
            path (str): Path to the file with the pre-trained weights.

        Returns:
            FrozenDict: The frozen parameters to be used in the model.
        """
        # Load the pre-trained weights from a file (assuming they are saved in .npy format)

        pre_trained_params = np.load(path, allow_pickle=True).item()
        # pre_trained_params is a dict with:
        # {0: {"weights": ..., "bias": ...}, 1: {"weights": ..., "bias": ...}}

        # Convert this dict into the format expected by the `FrozenDict`
        # We'll need to map these params into Flax's expected parameter format for the two layers.

        frozen_params = {
            "frozen_dense1": {
                "kernel": pre_trained_params["linear_0"][
                    "weight"
                ].T,  # Map weights to kernel
                "bias": pre_trained_params["linear_0"]["bias"],  # Map bias to bias
            },
            "frozen_dense2": {
                "kernel": pre_trained_params["linear_1"][
                    "weight"
                ].T,  # Map weights to kernel
                "bias": pre_trained_params["linear_1"]["bias"],  # Map bias to bias
            },
            "frozen_layer_norm1": {
                "scale": pre_trained_params["layer_norm_0"][
                    "weight"
                ],  # Map scale to weight
                "bias": pre_trained_params["layer_norm_0"]["bias"],  # Map bias to bias
            },
            "frozen_layer_norm2": {
                "scale": pre_trained_params["layer_norm_1"][
                    "weight"
                ],  # Map scale to weight
                "bias": pre_trained_params["layer_norm_1"]["bias"],  # Map bias to bias
            },
        }

        # Freeze the parameters so that they are immutable
        return flax.core.freeze(frozen_params)
