import sys
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

def sphere_dataset(total_size, ndims, data_key):
  # Generate random points on a `ndims` unit sphere 
  z = jax.random.normal(data_key, (total_size, ndims))
  z = z / jnp.sqrt(jnp.sum(z ** 2, axis=1, keepdims=True))
  return z

def shuffle_dataset(dataset, shuf_key):
  perm = jax.random.permutation(shuf_key, dataset.shape[0]) 
  return dataset[perm,:]

class Lin2(nnx.Module):
  def __init__(self, 
               tx: optax.GradientTransformation,
               widths: List[int], 
               do_seqgrad: bool,
               rngs: nnx.Rngs,
               use_bias: bool=False):

    self.layers = [
        SGModule(nnx.Linear, tx, do_seqgrad, in_dim, out_dim, rngs=rngs, use_bias=use_bias)
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

@nnx.jit(static_argnames=('loss_mode',))
def evaluate(optimizer, metrics, batch, loss_mode):
  # accumulate metrics
  loss_fn = get_loss_fn(loss_mode)
  loss = loss_fn(optimizer.model, batch)
  metrics.update(loss=loss)


@nnx.jit(static_argnames=('loss_mode'))
def train_step(optimizer, metrics: nnx.MultiMetric, batch, coord_index, loss_mode):
  loss_fn = get_loss_fn(loss_mode)
    # jax.debug.print('step: {}', step)
  diff_state = nnx.DiffState(0, optimizer.wrt)
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=False, argnums=diff_state)
  loss, grads = grad_fn(optimizer.model, batch)
  optimizer.update(grads, coord_index)
  metrics.update(loss=loss)

def make_test_fn(dataset_size, batch_size, max_iterations, widths, do_seqgrad,
                 opt_mode, loss_mode):

  def diverges(lr, target_loss, data_key, rngs):
    tx = optax.sgd(lr)
    model = Lin2(tx, widths, do_seqgrad, rngs)
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
          train_step(optimizer, metrics, batch, coord_index, loss_mode)
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

def get_loss_fn(loss_mode):
  if loss_mode == 'origin':
    return zero_mapping_loss_fn
  elif loss_mode == 'origin_sqrt':
    return zero_mapping_sqrt_loss_fn
  elif loss_mode == 'sphere':
    return sphere_mapping_loss_fn
  else:
    raise RuntimeError(f'loss_mode must be either `origin`, `origin_sqrt` or `sphere`, got {loss_mode}')
  
def get_optimizer(model, tx, opt_mode):
  if opt_mode == 'per_layer':
    return opt.LayerOptimizer(model, tx)
  elif opt_mode == 'single_coord':
    return opt.SequentialOptimizer(model, tx)
  elif opt_mode == 'all_param':
    return opt.AllParamOptimizer(model, tx)
  elif opt_mode == 'odd_even':
    return opt.PartialOptimizer(model, tx)
  else:
    raise RuntimeError(f'opt_mode must be one of `layer`, `single`, `odd_even` or `coord`.'
                       f'  got {opt_mode}')


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

  print(
      f'{seed=}\n'
      f'{starting_lr=}\n'
      f'{warmup=}\n'
      f'{window_size=}\n'
      f'{factor=}\n'
      f'{step_budget=}\n'
      f'{target_loss=}\n'
      f'{widths=}\n'
      f'{do_seqgrad=}\n'
      f'{opt_mode=}\n'
      f'{loss_mode=}\n'
      )

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


def main(learning_rate=0.001,
         widths=None,
         dataset_size=100,
         max_epochs=10000,
         batch_size=1, 
         eval_every=100, 
         do_seqgrad=False,
         opt_mode='single',
         loss_mode='origin',
         target_loss=0.0,
         seed=12345):

  if widths is None:
    widths = [2, 2]

  print(
      f'{learning_rate=}\n'
      f'{seed=}\n'
      f'{dataset_size=}\n'
      f'{max_epochs=}\n'
      f'{batch_size=}\n'
      f'{do_seqgrad=}\n'
      f'{opt_mode=}\n'
      f'{loss_mode=}\n'
      f'{widths=}\n'
      f'{target_loss=}\n'
      f'{eval_every=}\n')
  rngs = nnx.Rngs(seed)
  data_key = rngs()
  tx = optax.sgd(learning_rate)
  model = Lin2(tx, widths, do_seqgrad, rngs)
  optimizer = get_optimizer(model, tx, opt_mode)

  metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
  assert dataset_size % batch_size == 0, f'{dataset_size=} must be multiple of {batch_size=}'
  ds_shape = dataset_size, batch_size, widths[0]
  test_ds_shape = dataset_size // 100, 100, widths[0]
  ds = sphere_dataset(dataset_size, widths[0], data_key)
  ds = ds.reshape(*ds_shape)
  initial_loss = None
  epoch = 0
  
  while epoch < max_epochs:
    for coord_index in range(optimizer.num_coord_blocks()):
      ds = shuffle_dataset(ds, rngs())
      for step, batch in enumerate(ds):
        train_step(optimizer, metrics, batch, coord_index, loss_mode)
      if epoch % eval_every == 0:
        test_ds = ds.reshape(*test_ds_shape)
        metrics.reset()
        for batch in test_ds:
          evaluate(optimizer, metrics, batch, loss_mode) 
        stats = metrics.compute()
        loss = stats['loss']
        print(f'epoch: {epoch}, coord: {coord_index}, step: {step}, loss: {loss:8.7f}')
        if loss < target_loss:
          break
      epoch += 1


if __name__ == '__main__':
  cmds = dict(main=main, stoc=stochastic_lr_search)
  fire.Fire(cmds)

