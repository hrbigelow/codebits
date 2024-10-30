import optax
import jax
from flax import nnx
from flax.nnx import extract, graph
from flax.nnx.transforms import general
from flax.nnx.training import optimizer as optim
import typing as tp

@graph.update_context('vjp')
def nnx_vjp(f, ins, out_g):
  g = general.merge_inputs(f, ctxtag='vjp')
  pure_ins = extract.to_tree(ins, ctxtag='vjp')
  h = lambda a: g(a)[1]            # h: a -> b
  _, vjp_fn = jax.vjp(h, pure_ins) # vjp_fn: b' -> (a,)
  ins_g, = vjp_fn(out_g)
  if isinstance(ins_g, extract.NodeStates):
    return ins_g.state
  return ins_g

class SGModule(nnx.Module):
  """
  Wrapper for an arbitrary nnx.Module which computes
  sequential gradient.  Uses its own optimizer instance, but
  just to compute the updated parameters.  Does not actually perform
  updates during backward.
  """
  def __init__(self, 
               module_cls: tp.Type, 
               tx: optax.GradientTransformation,
               do_seqgrad: bool,
               *args, **kwargs):
    self.mod = module_cls(*args, **kwargs)
    self.opt = MyOptimizer(self.mod, tx)
    self.sgrad_fn = make_custom_vjp(do_seqgrad)

  def __call__(self, x: jax.Array):
    return self.sgrad_fn(self.opt, x)

def make_custom_vjp(do_seqgrad=True):
  """
  Return a function which is enabled for sequential gradient 
  """
  def fn(opt, x):
    return opt.model(x)

  if not do_seqgrad:
    return fn

  @nnx.custom_vjp
  def primal(opt, x):
    return fn(opt, x)

  def fn_fwd(opt, x):
    return fn(opt, x), (opt, x)

  def fn_bwd(res, g):
    ins_g, out_g = g
    opt, x = res
    opt_g = nnx_vjp(lambda opt_arg: opt_arg.model(x), opt, out_g)
    # param_g = jax.lax.pmean(param_g, 'dev') # ? each gradient on a different device
    new_params = opt.get_updated(opt_g['model'])
    graphdef = nnx.graphdef(opt.model)
    new_model = nnx.merge(graphdef, new_params)
    data_g = nnx_vjp(lambda x_arg: new_model(x_arg), x, out_g)
    # print(f'in do_seqgrad')
    # opt.update(opt_g['model']) 
    return (opt_g, data_g) 

  primal.defvjp(fn_fwd, fn_bwd)
  return primal

class MyOptimizer(nnx.Optimizer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def get_updated(self, grads):
    """
    Get the updated params without updating any state
    """
    params = nnx.state(self.model, self.wrt)
    opt_state = optim._opt_state_variables_to_state(self.opt_state)

    updates, new_opt_state = self.tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    assert isinstance(new_params, nnx.State)
    return new_params

