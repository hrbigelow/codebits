import fire
import jax.numpy as jnp, jax

# @jax.jit
def set_rand(a, b, i, rng):
  blocks = jax.random.randint(rng, a.shape, 0, 3)
  mask = (blocks == i)
  # aref = a.at[mask]
  # bslc = b.at[mask].get()
  return a.at[mask].set(b.at[mask].get())

@jax.jit
def set_rand2(a, b, i, rng):
  blocks = jax.random.randint(rng, a.shape, 0, 3)
  mask = (blocks == i)
  return jnp.where(mask, b, a).astype(a.dtype)

def main(seed, i, func):
  shape = 3, 5
  a = jnp.arange(15).reshape(*shape)
  b = jnp.ones(shape)
  rng = jax.random.PRNGKey(seed)
  if func == 'at':
    c = set_rand(a, b, i, rng)
  elif func == 'where':
    c = set_rand2(a, b, i, rng)
  print(c)

if __name__ == '__main__':
  fire.Fire(main)

