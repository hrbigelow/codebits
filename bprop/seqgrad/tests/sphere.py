import sys
import itertools
from collections import deque
import numpy as np
import jax.numpy as jnp, jax
import fire
from flax import nnx
from flax.nnx import filterlib
from typing import List
import optax
from seqgrad.module import SGModule
from seqgrad import opt
from seqgrad.tools import get_function_args
from pprint import pprint

def sphere_dataset(total_size, ndims, rng_key):
  # random points on an `ndims`-dimensional unit hypersphere 
  z = jax.random.normal(rng_key, (total_size, ndims))
  z = z / jnp.sqrt(jnp.sum(z ** 2, axis=1, keepdims=True))
  return z

def so_dataset(ndims, rng_key):
  from scipy.stats import special_ortho_group
  key_data = jax.random.key_data(rng_key)
  mat = special_ortho_group.rvs(ndims, 1, np.random.RandomState(key_data))
  assert jnp.allclose((mat**2).sum(axis=0), 1.0)
  assert jnp.allclose((mat**2).sum(axis=1), 1.0)
  return mat

def get_dataset(dataset_type, ndims, total_size, rng_key):
  if dataset_type == 'sphere':
    return sphere_dataset(total_size, ndims, rng_key)
  elif dataset_type == 'so':
    return so_dataset(ndims, rng_key)
  else:
    raise RuntimeError(f'{dataset_type=} must be one of (sphere, so)')

def check_eigenvalues(data):
    cov = data.T @ data / data.shape[0]
    return np.linalg.eigvals(cov)

def shuffle_dataset(dataset, rng_key):
  perm = jax.random.permutation(rng_key, dataset.shape[0]) 
  return dataset[perm,:]

class Lin2(nnx.Module):
  def __init__(self, 
               widths: List[int], 
               rngs: nnx.Rngs,
               init_mode: str,
               use_bias: bool=False):

    if init_mode == 'ortho':
      init = nnx.initializers.orthogonal()
    else:
      init = nnx.initializers.variance_scaling()

    self.layers = [
        # SGModule(nnx.Linear, tx, do_seqgrad, in_dim, out_dim, rngs=rngs, use_bias=use_bias)
        nnx.Linear(in_dim, out_dim, rngs=rngs, use_bias=use_bias, kernel_init=init)
        for in_dim, out_dim in zip(widths[:-1], widths[1:])
        ]

  def __call__(self, x: jax.Array):
    for layer in self.layers:
      x = layer(x)
    return x


def sphere_mapping_loss_fn(model, batch):
  y = model(batch)
  norms2 = jnp.sum(y ** 2, axis=1)
  loss = jnp.mean((1.0 - norms2) ** 2)
  return loss

def zero_mapping_loss_fn(model, batch):
  y = model(batch) # [batch, out_dim]
  norms2 = jnp.sum(y ** 2, axis=1)
  loss = 0.5 * jnp.mean(norms2)
  return loss

def zero_mapping_sqrt_loss_fn(model, batch):
  y = model(batch) # [batch, out_dim]
  norms2 = jnp.sum(y ** 2, axis=1)
  loss = jnp.mean(jnp.sqrt(norms2))
  return loss

def get_loss_fn(loss_mode):
  if loss_mode == 'origin':
    return zero_mapping_loss_fn
  elif loss_mode == 'origin_sqrt':
    return zero_mapping_sqrt_loss_fn
  elif loss_mode == 'sphere':
    return sphere_mapping_loss_fn
  else:
    raise RuntimeError(f'loss_mode must be either `origin`, `origin_sqrt` or `sphere`, got {loss_mode}')

@nnx.jit(static_argnames=('loss_mode',))
def evaluate(optimizer, points, loss_mode):
  # points is f4[dataset_size, ndim]
  dataset_size, ndim = points.shape
  assert dataset_size % 100 == 0, f'{dataset_size=} must be divisible by 100'
  points = points.reshape(dataset_size // 100, 100, ndim)
  metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
  loss_fn = get_loss_fn(loss_mode)
  for batch in points:
    loss = loss_fn(optimizer.model, batch)
    metrics.update(loss=loss)
  return metrics.compute()['loss']


@nnx.jit(static_argnames=('loss_mode'))
def train_step(optimizer, batch, coord_index, loss_mode):
  loss_fn = get_loss_fn(loss_mode)
    # jax.debug.print('step: {}', step)
  diff_state = nnx.DiffState(0, optimizer.wrt)
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=False, argnums=diff_state)
  loss, grads = grad_fn(optimizer.model, batch)
  optimizer.update(grads, coord_index)
  return loss

def make_test_fn(dataset_size, batch_size, max_iterations, widths, do_seqgrad,
                 opt_mode, loss_mode):

  def diverges(lr, target_loss, data_key, rngs):
    tx = optax.sgd(lr)
    model = Lin2(widths, rngs, init_mode)
    optimizer = get_optimizer(model, tx, opt_mode)
    assert dataset_size % batch_size == 0, f'{dataset_size=} must be multiple of {batch_size=}'
    ds_shape = dataset_size, batch_size, widths[0]
    test_ds_shape = dataset_size // 100, 100, widths[0]
    iteration = 0

    ds = sphere_dataset(dataset_size, widths[0], data_key)
    ds = ds.reshape(*ds_shape)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    initial_loss = None
    while iteration < max_iterations:
      for coord_index in range(optimizer.num_coord_blocks()):
        ds = shuffle_dataset(ds, rngs())
        for step, batch in enumerate(ds):
          _ = train_step(optimizer, metrics, batch, coord_index, loss_mode)
        test_ds = ds.reshape(*test_ds_shape)
        metrics.reset()
        for batch in test_ds:
          evaluate(optimizer, metrics, batch, loss_mode) 
        stats = metrics.compute()
        loss = stats['loss']
        if jnp.isnan(loss) or jnp.isinf(loss): 
            # print(f'diverged at step {step}, loss {loss}')
          return True
        if loss < target_loss:
          return False
        print(f'{iteration=}, {loss=:8.5f}')
      iteration += 1
    return True
  return diverges
  
def get_optimizer(model, tx, opt_mode, rng):
  if opt_mode == 'per_layer':
    return opt.LayerOptimizer(model, tx)
  elif opt_mode == 'single_coord':
    return opt.SequentialOptimizer(model, tx)
  elif opt_mode == 'all_param':
    return opt.AllParamOptimizer(model, tx)
  elif opt_mode == 'odd_even':
    return opt.PartialOptimizer(model, tx)
  elif opt_mode.startswith('partition'):
    try:
      npart = int(opt_mode[9:])
      assert npart > 0
    except Exception as e:
      raise RuntimeError(
          f'For opt_mode starting with `partition`, must end in positive integer.'
          f'Got {opt_mode=}')
    return opt.PartitionOptimizer(model, tx, npart, rng)
  else:
    raise RuntimeError(
        f'opt_mode must be one of '
        f'(per_layer, single_coord, all_param, odd_even). '
        f'got "{opt_mode}"')


def stochastic_lr_search(
    seed=12345,
    starting_lr=0.1,
    window_size=100,
    warmup=60,
    factor=1.1,
    step_budget=100000,
    target_loss=1e-5,
    batch_size=1,
    widths=None,
    do_seqgrad=False,
    opt_mode='single',
    loss_mode='origin'):
  if widths is None:
    widths = [2, 2]

  print('\n'.join(f'{k:18} = {v}' for k, v in get_function_args().items()))

  diverges_fn = make_test_fn(step_budget, batch_size, widths, do_seqgrad, opt_mode, loss_mode)

  lr = starting_lr
  lrs = deque([None] * window_size)
  rngs = nnx.Rngs(seed)

  for i in range(warmup + window_size):
    if diverges_fn(lr, target_loss, rngs):
      lr *= factor ** -1
    else:
      lr *= factor
    print(f'\r{lr:6.5f}', end='')
    lrs.popleft()
    lrs.append(lr)
  stochastic_lr = sum(lrs) / len(lrs) 
  print(f'\r{opt_mode=} {loss_mode=} {widths=} {stochastic_lr=:9.8f}')


def coordinate_descent(optimizer, points, learning_rate, batch_size, target_loss, rngs, eval_every,
                       max_train_steps, loss_mode, logger):
  train_step_num = 0
  initial_loss = evaluate(optimizer, points, loss_mode)
  dataset_size, ndims = points.shape
  ds = points.reshape(dataset_size // batch_size, batch_size, ndims)

  for coord_index in itertools.cycle(range(optimizer.num_coord_blocks())):
    if coord_index == 0:
      optimizer.on_new_step()
    ds = shuffle_dataset(ds, rngs())
    for step, batch in enumerate(ds):
      if train_step_num % eval_every == 0:
        loss = evaluate(optimizer, points, loss_mode)
        print(f'train_step: {train_step_num}, coord: {coord_index}, loss: {loss:8.7f}')
        if logger is not None:
          logger.write(f'lr-{learning_rate}', x=train_step_num, y=loss)
        if loss < target_loss:
          return 'Converged'
        if loss > 5.0 * initial_loss:
          return 'Diverged'
      _ = train_step(optimizer, batch, coord_index, loss_mode)
      train_step_num += 1
      if train_step_num >= max_train_steps:
        return 'Exceeded max train steps'

def main(learning_rate=0.001,
         widths=None,
         dataset_type='so',
         dataset_size=100,
         max_train_steps=100000, 
         batch_size=1, 
         eval_every=100, 
         # do_seqgrad=False,
         opt_mode='single',
         loss_mode='origin',
         init_mode='ortho',
         target_loss=1e-5,
         seed=12345,
         log_run_name=None,
         log_path=None):

  if widths is None:
    widths = [2, 2]

  print('\n'.join(f'{k:18} = {v}' for k, v in get_function_args().items()))
  rngs = nnx.Rngs(seed)
  data_key = rngs()
  tx = optax.sgd(learning_rate)
  model = Lin2(widths, rngs, init_mode)
  optimizer = get_optimizer(model, tx, opt_mode, rngs)
  assert dataset_size % batch_size == 0, f'{dataset_size=} must be multiple of {batch_size=}'
  ndims = widths[0]
  points = get_dataset(dataset_type, ndims, dataset_size, data_key)
  print(f'mean example sum: {jnp.mean(points.sum(axis=1))}')
  print(f'mean example norm: {jnp.mean(jnp.sum(points ** 2, axis=1))}')
  # print(f'eiginvalues of dataset: {check_eigenvalues(points)}')

  train_step_num = 0

  if log_run_name is not None:
    assert log_path is not None, f'log_path not provided but log_run_name provided'
    buffer_items = 100
    from streamvis.logger import DataLogger
    logger = DataLogger(log_run_name)
    logger.init(log_path, buffer_items)
  else:
    logger = None

  status = coordinate_descent(optimizer, points, learning_rate, batch_size,
                              target_loss, rngs, eval_every, max_train_steps,
                              loss_mode, logger)
  if logger is not None:
    logger.flush_buffer()
  print(status)

if __name__ == '__main__':
  cmds = dict(main=main, stoc=stochastic_lr_search)
  fire.Fire(cmds)

