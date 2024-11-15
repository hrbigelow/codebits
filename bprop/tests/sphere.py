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

def sphere_dataset(num_steps, batch_size, ndims, rngs):
  # Generate random points on a `ndims` unit sphere 
  def make_chunk(chunk):
    z = jax.random.normal(rngs(), (chunk, batch_size, ndims))
    z = z / jnp.sqrt(jnp.sum(z ** 2, axis=2, keepdims=True))
    return z
  while num_steps > 0:
    z = make_chunk(min(num_steps, 1000))
    yield from z
    num_steps -= z.shape[0]


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


@nnx.jit(static_argnames=('loss_mode'))
def train_step(model, metrics: nnx.MultiMetric, batch, optimizer, loss_mode):
  loss_fn = get_loss_fn(loss_mode)
  # for i in range(optimizer.num_substeps):
  def step_fn(step, inputs):
    _, optimizer = inputs
    diff_state = nnx.DiffState(0, optimizer.wrt)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False, argnums=diff_state)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads, step)
    return loss, optimizer 

  loss, _ = nnx.fori_loop(0, optimizer.num_substeps, step_fn, (0.0, optimizer))
  metrics.update(loss=loss)

def make_test_fn(step_budget, batch_size, widths, do_seqgrad, opt_mode, loss_mode):

  def diverges(lr, target_loss, rngs):
    tx = optax.sgd(lr)
    model = Lin2(tx, widths, do_seqgrad, rngs)
    # train_step = make_train_step(model, tx, opt_mode, loss_mode)
    # loss_fn = get_loss_fn(loss_mode)
    optimizer = get_optimizer(model, tx, opt_mode)
    ds = sphere_dataset(step_budget, batch_size, widths[0], rngs)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    initial_loss = None
    for step, batch in enumerate(ds):
      train_step(model, metrics, batch, optimizer, loss_mode)
      stats = metrics.compute()
      metrics.reset()
      loss = stats['loss']
      if step == 30:
        initial_loss = loss
      if jnp.isnan(loss) or jnp.isinf(loss): 
        # print(f'diverged at step {step}, loss {loss}')
        return True
      # if initial_loss is not None and loss > 2.0 * initial_loss:
        # print(f'diverged at step {step}, loss {loss}')
        # return True
      if loss < target_loss:
        return False
      # print(step, loss, target_loss)
    print(f'exit loop at step {step}')
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
  if opt_mode == 'layer':
    prefixes = [('layers', l, 'mod') for l in range(len(model.layers))][::-1]
    filters = [filterlib.PathStartsWith(p) for p in prefixes]
    update_exprs = [ParamUpdateExpr(f, None) for f in filters]
    return opt.PartialOptimizer(update_exprs, model, tx)
  elif opt_mode == 'single':
    return opt.PartialOptimizer(opt.SequentialCoordFn, model, tx)
  elif opt_mode == 'odd_even':
    return opt.PartialOptimizer(opt.OddEvenCoordFn, model, tx)
  elif opt_mode == 'coord':
    # individual coordinates
    pass

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


def find_lowest_divergent(
    starting_lr = 0.1,
    start_step_size = 0.05,
    tolerance = 0.00001,
    widths = None,
    batch_size=1, 
    num_steps=500, 
    success_loss = 0.0001,
    do_seqgrad=True,
    opt_mode='single',
    loss_mode='origin',
    seed=12345,
    dataset_seed=12345):
  """
  Perform binary search to find lowest learning rate that diverges at least once in
  `num_tries` trainings of `num_steps`.  Here, "diverges" means that the loss takes
  on a nan or inf value before `num_steps` has occurred.
  """
  if widths is None:
    widths = [2, 2]

  print(
      f'{seed=}\n'
      f'{start_step_size=}\n'
      f'{dataset_seed=}\n'
      f'{widths=}\n'
      f'{do_seqgrad=}\n'
      f'{opt_mode=}\n'
      f'{num_steps=}\n'
      f'{tolerance=}\n'
      f'{success_loss=}\n'
      )
  # print()
  rngs = nnx.Rngs(seed)
  def target_loss(lr):
    tx = optax.sgd(lr)
    model = Lin2(tx, widths, do_seqgrad, rngs)
    train_step(model, metrics, batch, optimizer, loss_mode)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    ds = sphere_dataset(num_steps, batch_size, widths[0], dataset_seed)
    for step, batch in enumerate(ds):
      train_step(model, metrics, batch)
      stats = metrics.compute()
      loss = stats['loss']
      metrics.reset()
    return loss

  def clear_print(msg):
    print('\r' + ' ' * 120, end='', flush=True)
    print('\r' + msg, end='', flush=True)

  # starting_lr is assumed to diverge
  ceiling_x = jnp.log(starting_lr)
  step_size = start_step_size
  while step_size > tolerance:
    test_lr = jnp.exp(ceiling_x - step_size)
    ceiling_lr = jnp.exp(ceiling_x)
    diverged = False

    # clear_print(f'{step_size=:8.7f}, {ceiling_lr=:10.9f}, {test_lr=:10.9f}, try {t}')
    ref_loss = target_loss(test_lr)
    if jnp.isnan(ref_loss) or jnp.isinf(ref_loss):
      ref_loss = 1.0
    print(f'{test_lr:10.9f}\t{ref_loss:10.9f}', file=sys.stderr)
    if ref_loss > success_loss:
      diverged = True
    if diverged:
      ceiling_x = jnp.log(test_lr) 
      step_size = start_step_size
      # msg = f'diverged at try {diverged_t}'
      # step_size *= 1.01
    else:
      step_size *= 1.01 ** -1 
      # msg = ''
    # clear_print(f'\r{step_size=:8.7f}, {ceiling_lr=:10.9f} {test_lr=:10.9f}, {msg}')

  print()
  print(f'Minimum divergent learning rate: {ceiling_lr:10.9f}')


def main(learning_rate=0.001,
         widths=None,
         batch_size=1, 
         num_steps=3000, 
         eval_every=100, 
         do_seqgrad=False,
         opt_mode='single',
         loss_mode='origin',
         goal_loss=0.0,
         seed=12345):

  if widths is None:
    widths = [2, 2]

  print(
      f'{learning_rate=}\n'
      f'{seed=}\n'
      f'{do_seqgrad=}\n'
      f'{opt_mode=}\n'
      f'{loss_mode=}\n'
      f'{widths=}\n'
      f'{num_steps=}\n'
      f'{goal_loss=}\n'
      f'{eval_every=}\n')
  rngs = nnx.Rngs(seed)
  tx = optax.sgd(learning_rate)
  model = Lin2(tx, widths, do_seqgrad, rngs)
  optimizer = get_optimizer(model, tx, opt_mode)

  metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
  ds = sphere_dataset(num_steps, batch_size, widths[0], rngs)
  initial_loss = None
  
  for step, batch in enumerate(ds):
    train_step(model, metrics, batch, optimizer, loss_mode)
    stats = metrics.compute()
    metrics.reset()
    loss = stats['loss']
    if initial_loss is None:
      initial_loss = loss 
    if loss < goal_loss:
      break

    if step >= 0 and (step % eval_every == 0 or step == num_steps - 1):  # One training epoch has passed.
      print(
          f'seqgrad: {do_seqgrad} step: {step}, '
          f'loss: {stats["loss"]:8.7f}'
          )


if __name__ == '__main__':
  cmds = dict(main=main, 
              findl=find_lowest_divergent, 
              stoc=stochastic_lr_search)
  fire.Fire(cmds)

