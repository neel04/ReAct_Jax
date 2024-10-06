import os
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Annotated, Any, Tuple

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree
from jmp import Policy

from ReAct.utils.helpers import get_spec_on_larger_dim, viz_obj


def get_strategy(strategy: str, *args):
    strategy = strategy.strip().lower()

    match strategy:
        case "ddp":
            strat = DDPSharding(*args)

        case "simple mp":
            strat = SimpleMPSharding(*args)

        case "megatron":
            strat = MegatronSharding(*args)

        case _:
            raise NotImplementedError(f"Strategy {strategy} does not exist.")

    return strat


class Sharding(ABC):
    def __init__(self, model_axis: int = 1) -> None:
        self.model_axis: int = model_axis
        self.policy: Policy | None = None

    @abstractmethod
    def get_mesh(self) -> Mesh: ...

    @abstractmethod
    def shard_data(self, tree: PyTree | Array) -> PyTree | Array: ...

    @abstractmethod
    def shard_model(self, tree: PyTree) -> PyTree: ...

    @abstractmethod
    def shard_one_hot(self, tree: PyTree) -> PyTree: ...

    def shard_cast(self, tree: PyTree) -> PyTree:
        """
        Return the casted & sharded version of the PyTree. Uses `policy.cast_to_compute`
        """
        assert (
            self.policy is not None
        ), "No policy registered for sharding. Use `filter_shard` instead of `shard_cast`"

        # return self.shard_data(self.policy.cast_to_compute(tree))
        return self.policy.cast_to_compute(tree)

    def set_policy(self, policy: Policy) -> None:
        self.policy = policy

    def get_devices(self):
        return mesh_utils.create_device_mesh(
            (jax.device_count() // self.model_axis, self.model_axis),
            allow_split_physical_axes=True,
        )

    @staticmethod
    def unwrap_struct(tgt: PyTree[Annotated[str, "DataclassInstance"]]) -> Tuple[Any]:
        get_val = lambda x: getattr(x, fields(x)[0].name)  # noqa: E731
        return tuple(map(get_val, tgt))

    def __call__(self, policy: Policy):
        self.policy = policy
        return self


class DDPSharding(Sharding):
    def __init__(self, model_axis: int = 1) -> None:
        super().__init__(model_axis)
        self.mesh = self.get_mesh()

    def get_mesh(self):
        return Mesh(self.get_devices(), axis_names=("data", "model"))

    def shard_data(self, tree: PyTree | Array) -> PyTree | Array:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P("data")))

    def shard_model(self, tree: PyTree) -> PyTree:
        return jtu.tree_map_with_path(self.ddp_sharding, tree)

    def shard_one_hot(self, tree: PyTree) -> PyTree:
        return tree

    def ddp_sharding(
        self, kp: Annotated[str, "DataclassInstance"], leaf: PyTree
    ) -> PyTree:
        if not eqx.is_array(leaf):
            return leaf

        sharding_ = NamedSharding(self.mesh, P())

        return eqx.filter_shard(leaf, sharding_)


class SimpleMPSharding(Sharding):
    def __init__(self, model_axis: int = 2) -> None:
        super().__init__(model_axis)
        self.mesh = self.get_mesh()

    def get_mesh(self) -> Mesh:
        return Mesh(self.get_devices(), axis_names=("data", "model"))

    def shard_data(self, tree: PyTree | Array) -> PyTree | Array:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P("data")))

    def shard_model(self, tree: PyTree) -> PyTree:
        return jtu.tree_map_with_path(self.simple_sharding, tree)

    def shard_one_hot(self, tree: PyTree) -> PyTree:
        return tree

    def simple_sharding(
        self, kp: Annotated[str, "DataclassInstance"], leaf: PyTree
    ) -> PyTree:
        if not eqx.is_array(leaf):
            return leaf

        sharding_ = NamedSharding(self.mesh, P())

        if leaf.ndim == 1:
            sharding_ = NamedSharding(self.mesh, P("model"))

        if leaf.ndim == 2:
            sharding_ = NamedSharding(self.mesh, P(None, "model"))

        return eqx.filter_shard(leaf, sharding_)


class MegatronSharding(Sharding):
    def __init__(self, model_axis: int = 2) -> None:
        super().__init__(model_axis)
        self.mesh = self.get_mesh()

    def get_mesh(self) -> Mesh:
        return Mesh(self.get_devices(), axis_names=("data", "model"))

    def shard_data(self, tree: PyTree | Array) -> PyTree | Array:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P("data")))

    def shard_model(self, tree: PyTree) -> PyTree:
        return jtu.tree_map_with_path(self.megatron_sharding, tree)

    def shard_one_hot(self, tree: PyTree) -> PyTree:
        return tree

    def megatron_sharding(
        self, kp: Annotated[str, "DataclassInstance"], leaf: PyTree
    ) -> PyTree:
        if not eqx.is_array(leaf):
            return leaf

        sharding_ = NamedSharding(self.mesh, P())

        if leaf.ndim == 2:
            p_spec = get_spec_on_larger_dim(leaf)
            sharding_ = NamedSharding(self.mesh, P(*p_spec))

        return eqx.filter_shard(leaf, sharding_)


if __name__ == "__main__":
    # import sys
    # sys.path.append('.')

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    assert jax.device_count() == 8, "Hosts not correctly spoofed"

    key = jax.random.PRNGKey(0)
    BSZ, SEQLEN, WIDTH = 32, 256, 64

    model = GPT(4, SEQLEN, 2, WIDTH, 0.01, 50304, key=key)
    strategy = get_strategy("megatron", 1)

    data = jax.numpy.ones((BSZ, SEQLEN))
    data = strategy.shard_data(tree=data)
    sharded_model = strategy.shard_model(model)

    viz_obj(sharded_model)

    print("\n ++++++++ Sharded data: +++++++++++\n")
    jax.debug.visualize_array_sharding(data)

    print(model)
