import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator



def plot_alignment(true_labels, pred_labels, info=None):
  lens = len(true_labels)
  matrix = np.zeros(shape=[lens,lens],dtype=np.int32)
  for j in range(lens):
      for i in range(lens):
          matrix[i][j] = 1 if pred_labels[i]==true_labels[j] else 0
  # plot
  plt.switch_backend('agg')
  fig = plt.figure()
  ax = fig.add_subplot(111)
  cax = ax.matshow(matrix)
  fig.colorbar(cax)
  ax.xaxis.set_major_locator(MultipleLocator(1))
  ax.yaxis.set_major_locator(MultipleLocator(1))
  for i in range(matrix.shape[0]):
      ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
  ax.set_xticklabels([''] + true_labels, rotation=90)
  ax.set_yticklabels([''] + true_labels)

  plt.savefig('temp.png', format='png')