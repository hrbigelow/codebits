import jax.numpy as jnp, jax
import fire
from flax import nnx
import optax
from seqgrad.module import SGModule

def sphere_dataset(batch_size, num_steps, seed):
  """
  Generate random points on a 2D unit sphere
  """
  rng = nnx.Rngs(seed)
  for _ in range(num_steps):
    angle = jax.random.normal(rng(), (batch_size,)) * jnp.pi * 2
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

# @nnx.jit
def train_step(model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=False)
  loss, grads = grad_fn(model, batch)
  # print(grads)
  metrics.update(loss=loss)
  optimizer.update(grads)


def find_lowest_divergent(
    min_lr=0.005,
    max_lr=0.2,
    line_search_steps=30,
    num_tries=5,
    num_layers=3,
    layer_width=2, 
    batch_size=1, 
    num_steps=300, 
    do_seqgrad=True,
    seed=12345):
  """
  Perform line search to find lowest learning rate that diverges at least once in

  """
  def diverges(lr):
    tx = optax.sgd(lr)
    input_width = 2
    rngs = nnx.Rngs(seed)
    model = Lin2(tx, num_layers, input_width, layer_width, do_seqgrad, rngs)
    optimizer = nnx.Optimizer(model, tx)
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
    ds = sphere_dataset(batch_size, num_steps, seed)
    for step, batch in enumerate(ds):
      train_step(model, optimizer, metrics, batch)
      stats = metrics.compute()
      loss = stats['loss']
      metrics.reset()
      if jnp.isnan(loss) or jnp.isinf(loss):
        return True
    return False

  for lstep in range(line_search_steps):
    lr = (min_lr + max_lr) / 2.0
    print(f'{lstep=}, [{min_lr:10.8f}, {max_lr:10.8f}]', end='')
    diverged = False
    for _ in range(num_tries):
      if diverges(lr):
        diverged = True
        break
    if diverged:
      max_lr = lr
      print(' diverged')
    else:
      min_lr = lr
      print('')
  print(f'Minimum divergent learning rate: {lr:10.9f}')

def main(learning_rate=0.001,
         num_layers=2, 
         layer_width=2, 
         batch_size=1, 
         num_steps=3000, 
         eval_every=100, 
         do_seqgrad=True,
         stop_early=True,
         seed=12345):
  print(
      f'{learning_rate=}, '
      f'{seed=}, '
      f'{do_seqgrad=}, '
      f'{num_layers=}, '
      f'{layer_width=}')
  tx = optax.sgd(learning_rate)
  input_width = 2
  rngs = nnx.Rngs(seed)
  model = Lin2(tx, num_layers, input_width, layer_width, do_seqgrad, rngs)
  optimizer = nnx.Optimizer(model, tx)
  metrics = nnx.MultiMetric(loss=nnx.metrics.Average('loss'))
  ds = sphere_dataset(batch_size, num_steps, seed)
  initial_loss = None
  num_steps_to_goal = None
  goal_loss = None

  for step, batch in enumerate(ds):
    train_step(model, optimizer, metrics, batch)
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

if __name__ == '__main__':
  cmds = dict(main=main, findl=find_lowest_divergent)
  fire.Fire(cmds)

