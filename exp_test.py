import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score

from experiments import plot_confusion_matrix

cm = np.array([[21207,509,1223],
      [3361,673,1183],
      [2100,324,2141]])

class_names = ['Low','Mid','High']
plt.figure()
plot_confusion_matrix(cm,classes=class_names,
                      title='Confusion matrix, without normalization')
plt.figure()
plot_confusion_matrix(cm,classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()