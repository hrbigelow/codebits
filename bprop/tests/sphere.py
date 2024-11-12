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

def sphere_dataset(batch_size, num_steps, rngs):
  """
  Generate random points on a 2D unit sphere
  """
  for _ in range(num_steps):
    angle = jax.random.normal(rngs(), (batch_size,)) * jnp.pi * 2
    points = jnp.stack([jnp.sin(angle), jnp.cos(angle)], axis=1)
    yield points


class Lin2(nnx.Module):
  def __init__(self, 
               tx: optax.GradientTransformation,
               num_layers: int, 
               input_width: int,
               layer_width: int, 
               do_seqgrad: bool,
               rngs: nnx.Rngs):
    widths = [input_width] + [layer_width] * (num_layers - 2) + [input_width]

    self.layers = [
        SGModule(nnx.Linear, tx, do_seqgrad, in_dim, out_dim, rngs=rngs)
        for in_dim, out_dim in zip(widths[:-1], widths[1:])
        ]

  def __call__(self, x: jax.Array):
    for layer in self.layers:
      x = layer(x)
    return x


def loss_fn(model, batch):
  y = model(batch)
  norms2 = jnp.sum(y ** 2, axis=1)
  loss = jnp.mean((1.0 - norms2) ** 2)
  return loss

@nnx.jit
def train_step(model, optimizers: List[nnx.Optimizer], metrics: nnx.MultiMetric, batch):
  # perform one logical train step as a series of one or more block steps, each
  # defined by the optimizer.wrt filter
  for opt in optimizers:
    diff_state = nnx.DiffState(0, opt.wrt)
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=False, argnums=diff_state)
    loss, grads = grad_fn(model, batch)
    opt.update(grads)
  metrics.update(loss=loss)

def make_test_func(test_step, batch_size, num_layers, input_width, layer_width,
                   do_seqgrad, do_layerwise_opt):
  def diverges(lr, target_loss, rngs):
    tx = optax.sgd(lr)
    input_width = 2
    model = Lin2(tx, num_layers, input_width, layer_width, do_seqgrad, rngs)
    ds = sphere_dataset(batch_size, test_step, rngs)
    if do_layerwise_opt:
      optimizers = layerwise_optimizers(model, tx)
    else:
      optimizers = [nnx.Optimizer(model, tx)]
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    for step, batch in enumerate(ds):
      train_step(model, optimizers, metrics, batch)
      stats = metrics.compute()
      loss = stats['loss']
      metrics.reset()
    return jnp.isnan(loss) or jnp.isinf(loss) or loss > target_loss
  return diverges

def stochastic_lr_search(
    seed=12345,
    starting_lr=0.1,
    window_size=100,
    warmup=40,
    factor=1.1,
    test_step=500,
    batch_size=1,
    num_layers=2,
    input_width=2,
    layer_width=100,
    target_loss=1e-5,
    do_seqgrad=False,
    do_layerwise_opt=False):
  print(
      f'{seed=}\n'
      f'{starting_lr=}\n'
      f'{warmup=}\n'
      f'{window_size=}\n'
      f'{factor=}\n'
      f'{test_step=}\n'
      f'{target_loss=}\n'
      f'{num_layers=}\n'
      f'{layer_width=}\n'
      f'{do_seqgrad=}\n'
      f'{do_layerwise_opt=}\n'
      )

  diverges_fn = make_test_func(test_step, batch_size, num_layers, input_width,
                               layer_width, do_seqgrad, do_layerwise_opt)

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
  print(f'\rStochastic LR: {stochastic_lr:9.8f}')


def find_lowest_divergent(
    starting_lr = 0.1,
    start_step_size = 0.05,
    tolerance = 0.00001,
    num_layers=3,
    layer_width=2, 
    batch_size=1, 
    num_steps=500, 
    success_loss = 0.0001,
    do_seqgrad=True,
    do_layerwise_opt=False,
    seed=12345,
    dataset_seed=12345):
  """
  Perform binary search to find lowest learning rate that diverges at least once in
  `num_tries` trainings of `num_steps`.  Here, "diverges" means that the loss takes
  on a nan or inf value before `num_steps` has occurred.
  """
  print(
      f'{seed=}\n'
      f'{start_step_size=}\n'
      f'{dataset_seed=}\n'
      f'{num_layers=}\n'
      f'{layer_width=}\n'
      f'{do_seqgrad=}\n'
      f'{do_layerwise_opt=}\n'
      f'{num_steps=}\n'
      f'{tolerance=}\n'
      f'{success_loss=}\n'
      )
  # print()
  rngs = nnx.Rngs(seed)
  def target_loss(lr):
    tx = optax.sgd(lr)
    input_width = 2
    model = Lin2(tx, num_layers, input_width, layer_width, do_seqgrad, rngs)
    if do_layerwise_opt:
      optimizers = layerwise_optimizers(model, tx)
    else:
      optimizers = [nnx.Optimizer(model, tx)]
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    ds = sphere_dataset(batch_size, num_steps, dataset_seed)
    for step, batch in enumerate(ds):
      train_step(model, optimizers, metrics, batch)
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

def layerwise_optimizers(model, tx):
  prefixes = [('layers', l, 'mod') for l in range(len(model.layers))][::-1]
  return [nnx.Optimizer(model, tx, wrt=filterlib.PathStartsWith(p)) for p in prefixes]

def main(learning_rate=0.001,
         num_layers=2, 
         layer_width=2, 
         batch_size=1, 
         num_steps=3000, 
         eval_every=100, 
         do_seqgrad=True,
         do_layerwise_opt=False,
         stop_early=True,
         seed=12345):
  print(
      f'{learning_rate=}, '
      f'{seed=}, '
      f'{do_seqgrad=}, '
      f'{do_layerwise_opt=}, '
      f'{num_layers=}, '
      f'{num_steps=}, '
      f'{eval_every=}, '
      f'{layer_width=}')
  tx = optax.sgd(learning_rate)
  input_width = 2
  rngs = nnx.Rngs(seed)
  model = Lin2(tx, num_layers, input_width, layer_width, do_seqgrad, rngs)
  if do_layerwise_opt:
    optimizers = layerwise_optimizers(model, tx)
  else:
    optimizers = [nnx.Optimizer(model, tx)]

  metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
  ds = sphere_dataset(batch_size, num_steps, seed)
  initial_loss = None
  num_steps_to_goal = None
  goal_loss = None

  
  for step, batch in enumerate(ds):
    train_step(model, optimizers, metrics, batch)
    stats = metrics.compute()
    metrics.reset()
    loss = stats['loss']
    if initial_loss is None:
      initial_loss = loss 
    if num_steps_to_goal is None and initial_loss is not None:
      if loss < 0.01 * initial_loss:
        num_steps_to_goal = step
        goal_loss = loss
        if stop_early:
          break

    if step >= 0 and (step % eval_every == 0 or step == num_steps - 1):  # One training epoch has passed.
      print(
          f'seqgrad: {do_seqgrad} step: {step}, '
          f'loss: {stats["loss"]:8.7f}'
          )
  print(f'Num steps for 100x loss reduction '
        f'({initial_loss:8.7f} to {goal_loss:8.7f}): '
        f'{num_steps_to_goal}')

def scatter(file, probe_step, success_loss):
  # draw a scatter plot in log scale
  import matplotlib.pyplot as plt
  xs, ys = [], []
  with open(file, 'r') as fh:
    for line in fh:
      line.strip()
      x, y = line.split('\t')
      x = float(x)
      y = float(y)
      xs.append(x)
      ys.append(y)

  min_nonzero_y = min(y for y in ys if y != 0.0)
  max_y = max(ys)

  xs = np.array(xs)
  ys = np.array(ys)
  ys = np.maximum(min_nonzero_y * 0.1, ys)
  min_y = np.min(ys)
  plt.figure(figsize=(12,6))
  isinf = (ys == max_y)  
  iszero = (ys == min_y) 
  plt.scatter(xs[isinf], ys[isinf], color='red', alpha=0.6)
  plt.scatter(xs[iszero], ys[iszero], color='green', alpha=0.6)
  plt.scatter(xs[~(isinf | iszero)], ys[~(isinf | iszero)], alpha=0.6)
  x_min, x_max = min(xs), max(xs)
  # plt.xticks(np.linspace(x_min, x_max, 10), rotation=90, ha='center')
  plt.yscale('log')
  plt.xlabel('learning rate')
  plt.ylabel(f'loss at step {probe_step}')
  plt.title(f'Learning rate probed at step {probe_step} for threshold {success_loss}')
  plt.grid(True, which="both", ls="-", alpha=0.2)
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  cmds = dict(main=main, 
              findl=find_lowest_divergent, 
              stoc=stochastic_lr_search, 
              scatter=scatter)
  fire.Fire(cmds)

