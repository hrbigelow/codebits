import optax
from dataclasses import dataclass
from typing import Tuple, Any, List
import jax, jax.numpy as jnp
import functools
from flax import nnx
from flax.nnx.filterlib import Filter
from flax.nnx.training import optimizer

class PartialOptimizer(nnx.Optimizer):
  def __init__(self, model, tx, wrt=nnx.Param):
    super().__init__(model=model, tx=tx, wrt=wrt)
    state = nnx.state(model, wrt)
    state_list, self.treedef = jax.tree.flatten(state)
    self.param_shapes = [p.shape for p in state_list]

  def num_steps(self):
    """
    Return the number of sub-steps required for a complete SGD step
    """
    raise NotImplementedError

  def slices(self, i):
    """
    i:  index for the substep, in [0, num_steps())
    Returns:
    active: bool tensor.  active[i] means i'th parameter tensor is updated
    slices: List.  slices[i] provides a slice expression for the i'th parameter tensor
    """
    raise NotImplementedError

  def _update_one(self, active, slice_expr, grad, param, opt_state):
    """
    Optionally updates one parameter tensor if it is active
    """
    # optionally compute updated grad, param and opt_state if `active`
    # will be applied to each param tensor
    def do_update(grad, param, opt_state):
      grad_s = grad.at[slice_expr].get()
      param_s = param.at[slice_expr].get()
      # opt_state = opt_state[slice_expr].get() # TODO: generalize this for stateful updates
      opt_state_s = opt_state
      update_s, opt_state_s = self.tx.update(grad_s, opt_state_s)
      param_s = optax.apply_updates(param_s, update_s)
      param = param.at[slice_expr].set(param_s)
      # opt_state = opt_state.at[slice_expr].set(opt_state_s)
      # jax.debug.print('in do_update: mean squared: update_s: {}', jnp.mean(update_s ** 2))
      return param, opt_state
    def noop(grad, param, opt_state):
      # jax.debug.print('in noop')
      return param, opt_state 
    return jax.lax.cond(active, do_update, noop, grad, param, opt_state)

  def update(self, grads, i):
    """
    Update all parameter tensors
    """
    # jax.debug.print('in update, sub-step {}', i)
    active, slice_exprs = self.slices(i)
    params = nnx.state(self.model, self.wrt)
    params_l = jax.tree.leaves(params)
    grads_l = jax.tree.leaves(grads)
    opt_state = optimizer._opt_state_variables_to_state(self.opt_state)
    opt_state_l = [opt_state] * len(slice_exprs)

    new_params_l, new_opt_state_l = [], []
    for a, s, g, p, o in zip(active, slice_exprs, grads_l, params_l, opt_state_l):
      p, o = self._update_one(a, s, g, p, o)
      new_params_l.append(p)
      new_opt_state_l.append(o)

    params = jax.tree.unflatten(self.treedef, new_params_l)
    nnx.update(self.model, params)
    new_opt_state_l = opt_state # TODO: generalize this to stateful SGD
    optimizer._update_opt_state(self.opt_state, new_opt_state_l)

class SequentialOptimizer(PartialOptimizer):
  # update one coordinate at a time
  def __init__(self, model, tx, wrt=nnx.Param):
    super().__init__(model, tx, wrt)
    sizes = [jnp.prod(jnp.array(shape)) for shape in self.param_shapes]
    self.cumul_sizes = nnx.Variable(jnp.cumsum(jnp.array([0] + sizes)))

  def num_steps(self):
    return sum((jnp.prod(jnp.array(shape)) for shape in self.param_shapes))

  def slices(self, i):
    active = (i >= self.cumul_sizes[:-1]) & (i < self.cumul_sizes[1:])
    slice_exprs = i - self.cumul_sizes[:-1]
    return active, slice_exprs

class LayerOptimizer(PartialOptimizer):
  # update one layer's worth of parameters at a time
  def num_steps(self):
    return len(self.param_shapes)

  def slices(self, i):
    active = (i == jnp.arange(self.num_steps()))
    slice_exprs = [Ellipsis] * self.num_steps()
    return active, slice_exprs

class AllParamOptimizer(PartialOptimizer):
  """
  Update all trainable parameters together
  """
  def num_steps(self):
    return 1

  def slices(self, i):
    return jnp.array([True]), [Ellipsis]

