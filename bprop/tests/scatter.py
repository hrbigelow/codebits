import fire

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
    fire.Fire(scatter)
