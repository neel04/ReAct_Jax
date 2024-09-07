import os
from abc import ABC, abstractmethod
from typing import List, Tuple

import equinox as eqx
import jax
import jax.tree_util as jtu
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, PyTree

from ReAct.model.baseline import GPT
from ReAct.utils.helpers import get_spec_on_larger_dim, viz_obj


def get_strategy(strategy: str, *args):
    strategy = strategy.strip().lower()
    match strategy:
        case 'ddp':
            strat = DDPSharding(*args)
    
        case 'simple mp':
            strat = SimpleMPSharding(*args)

        case 'megatron':
            strat = MegatronSharding(*args)

        case _:
            raise NotImplementedError(f'Strategy {strategy} does not exist.')

    return strat

        
class Sharding(ABC):
    def __init__(self, model_axis: int = 1) -> None:
        self.model_axis = model_axis

    @abstractmethod
    def get_mesh(self) -> Mesh:
        ...

    @abstractmethod
    def shard_data(self, tree: PyTree) -> PyTree:
        ...

    @abstractmethod
    def shard_model(self, tree: PyTree) -> PyTree:
        ...

    @abstractmethod
    def shard_one_hot(self, tree: PyTree) -> PyTree:
        ...

    def get_devices(self):
        return mesh_utils.create_device_mesh(
            (jax.device_count() // self.model_axis, self.model_axis),
            allow_split_physical_axes=True,
        )

    def add_indices_to_tree(self, tree: PyTree, start_index: int = 0, dims_to_count = 3):
        '''
        dims_to_count: leaves of what `.ndim` would be counted.
        '''
        def add_index(leaf, index):
            return [leaf, index[0]]

        def index_incrementer(leaf: PyTree) -> List[int]:
            if not eqx.is_array(leaf):
                return [-999]

            nonlocal start_index
            start_index += 1 if leaf.ndim >= 3 else 0
            return [start_index - 1]

        indexed_tree = jtu.tree_map(add_index, tree, jtu.tree_map(index_incrementer, tree))
        return indexed_tree

class DDPSharding(Sharding):
    def __init__(self, model_axis: int = 1) -> None:
        super().__init__(model_axis)
        self.mesh = self.get_mesh()

    def get_mesh(self) -> Mesh:
        return Mesh(self.get_devices(), axis_names=('data', 'model'))

    def shard_data(self, tree: PyTree | Array) -> PyTree | Array:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P('data')))

    def shard_model(self, tree: PyTree) -> PyTree:
        # return jtu.tree_map(self.ddp_sharding, tree)
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P()))

    def shard_one_hot(self, tree: PyTree) -> PyTree:
        return tree
        
    def ddp_sharding(self, leaf: PyTree) -> PyTree:
        if not eqx.is_array(leaf):
            return leaf

        sharding_ = NamedSharding(self.mesh, P())

        if leaf.ndim >= 2:
            sharding_ = NamedSharding(self.mesh, P('model'))

        return eqx.filter_shard(leaf, sharding_)

class SimpleMPSharding(Sharding):
    def __init__(self, strategy: str, model_axis: int = 2) -> None:
        super().__init__(model_axis)
        self.mesh = self.get_mesh()
    
    def get_mesh(self) -> Mesh:
        return Mesh(self.get_devices(), axis_names=('data', 'model'))

    def shard_data(self, tree: PyTree | Array) -> PyTree | Array:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P('data')))

    def shard_model(self, tree: PyTree) -> PyTree:
        return jtu.tree_map(self.simple_sharding, tree)

    def shard_one_hot(self, tree: PyTree) -> PyTree:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P('data', None, 'model')))

    def simple_sharding(self, leaf: PyTree) -> PyTree:
        if not eqx.is_array(leaf):
            return leaf

        sharding_ = NamedSharding(self.mesh, P())

        if leaf.ndim == 1:
            sharding_ = NamedSharding(self.mesh, P("model"))

        if leaf.ndim >= 2:
            sharding_ = NamedSharding(self.mesh, P(None, "model"))

        return eqx.filter_shard(leaf, sharding_)


class MegatronSharding(Sharding):
    def __init__(self, strategy: str, model_axis: int = 2) -> None:
        super().__init__(model_axis)
        self.mesh = self.get_mesh()

    def get_mesh(self) -> Mesh:
        return Mesh(self.get_devices(), axis_names=('data', 'model'))

    def shard_data(self, tree: PyTree | Array) -> PyTree | Array:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P('data')))
    
    def shard_model(self, tree: PyTree) -> PyTree:
        is_leaf = lambda x: isinstance(x, list)  # noqa: E731
        tree = self.add_indices_to_tree(tree, dims_to_count = 3)
        return jtu.tree_map(self.megatron_sharding, tree, is_leaf=is_leaf)

    def shard_one_hot(self, tree: PyTree) -> PyTree:
        return eqx.filter_shard(tree, NamedSharding(self.mesh, P("data", None, "model")))

    def megatron_sharding(self, leaf_and_index: Tuple[PyTree, int]) -> PyTree:
        leaf, idx = leaf_and_index

        if not eqx.is_array(leaf):
            return leaf

        sharding_ = NamedSharding(self.mesh, P())

        # LN params and embedding 1Ds
        if leaf.ndim == 1:
            if max(leaf.shape) >= 2**14:
                sharding_ = NamedSharding(self.mesh, P('model'))

        # embedding and unembedding
        if leaf.ndim == 2:
            p_spec = get_spec_on_larger_dim(leaf)
            sharding_ = NamedSharding(self.mesh, P(*p_spec))

        if leaf.ndim == 3:
            if idx % 2 == 0:
                sharding_ = NamedSharding(self.mesh, P(None, None, "model"))
            else:
                sharding_ = NamedSharding(self.mesh, P(None, "model", None))

        return eqx.filter_shard(leaf, sharding_)


if __name__ == "__main__":
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    assert len(jax.devices()) == 8, "Hosts not correctly spoofed"

    key = jax.random.PRNGKey(0)
    BSZ, SEQLEN, WIDTH = 32, 256, 64

    model = GPT(4, SEQLEN, 2, WIDTH, 0.01, 50304, key=key)
    strategy= get_strategy('megatron', 1)

    data = jax.numpy.ones((BSZ, SEQLEN))
    data = strategy.shard_data(tree=data)
    sharded_model = strategy.shard_model(model)

    viz_obj(sharded_model)

    print('\n ++++++++ Sharded data: +++++++++++\n')
    jax.debug.visualize_array_sharding(data)
