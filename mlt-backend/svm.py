import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
import itertools
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from uuid import uuid4

csv_file = {}
X_database = {}
Y_database = {}
# Phase (1). data collection (file upload):
def svm_fileUpload(files):
  #  if request.method == 'POST':
  if files['file'].content_type != 'text/csv':
    return (json.dumps({'message': 'File must be a CSV'}), 400)
  
  df = pd.read_csv(files.get('file'))

  uuid = str(uuid4())

  csv_file[uuid] = df
  #print(df)
  return (json.dumps({'id': uuid}), 200)

# Phase (2). data preprocessing (removing missing values and scaling, return processed dataframe preview): 
def svm_rmMissingvalues(id):
  df = csv_file[id]
  df_new = df.dropna()
  return ((df_new.to_json()), 200)

def svm_scaling(id, scaleMode):
  df = csv_file[id]
  df_new = df.dropna()
  idx1, idx2 = 2, 3
  X_plot = df_new.iloc[:, 2:].to_numpy()

  X = [[x[idx1], x[idx2]] for x in df_new.iloc[:, 2:].to_numpy()]
  y = df_new.iloc[:, 1].to_numpy()
  feature_names = df_new.columns
  #labels = ["malignant", "benign"]  #feature_names[1]
  #cm_labels = ["malignant", "benign"]

  # Map labels from ['M', 'B'] to [0, 1] space
  y = np.where(y=='M', 0, 1)

  if scaleMode == "standardization":
    # standardization
    X = (X - np.mean(X)) / np.std(X)
    X_plot = (X_plot - np.mean(X_plot, axis=0)) / np.std(X_plot, axis=0)
  elif scaleMode == "normalization":
    # normalization
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X_plot = (X_plot - np.min(X_plot, axis=0)) / (np.max(X_plot, axis=0) - np.min(X_plot, axis=0))
  # error catching logic required here

  # remap labels
  for i in range(len(y)):
    if y[i] == 0: y[i] = 1
    else: y[i] = -1
  return (X, y, feature_names, X_plot)

def svm_get_label_mapping(i):
  labels = ["malignant", "benign"]  #feature_names[1]
  # Map between labels to help in visualizing them later
  if i == 1:
    return labels[0]
  else:
    return "Not " + labels[0]

def svm_scatter_plot(id, scaleMode):
  (_, y, feature_names, X_plot) = svm_scaling(id, scaleMode)
  x_index, y_index = 2, 3
  formatter = plt.FuncFormatter(lambda i, *args: svm_get_label_mapping(int(i)))
  plt.clf()
  plt.figure(figsize=(8, 6))
  plt.scatter(X_plot[:, x_index], X_plot[:, y_index], c=y)
  plt.colorbar(ticks=[1, -1], format=formatter)
  plt.xlabel(feature_names[x_index])
  plt.ylabel(feature_names[y_index])

  imgScatter = svm_img_to_base64(plt)
  return (json.dumps({'imgScatter': imgScatter}), 200)
  
"""# Visualize the Train and Test Data Split"""
def svm_train_test_split(id, test_size, scaleMode):
  (X, y, _, _) = svm_scaling(id, scaleMode)
  # Import train_test_split function
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
  return (X_train, X_test, y_train, y_test)

def svm_train_test_method(X_data, Y_data, id, scaleMode):
  x_index, y_index = 2, 3
  (_, _, feature_names, _) = svm_scaling(id, scaleMode)
  formatter = plt.FuncFormatter(lambda i, *args: svm_get_label_mapping(int(i)))
  plt.scatter(np.array(X_data)[:, 0], np.array(X_data)[:, 1], c=Y_data)
  plt.colorbar(ticks=[1, -1], format=formatter)
  plt.xlabel(feature_names[x_index])
  plt.ylabel(feature_names[y_index])


def svm_train_test_plot(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size

  (X_train, X_test, y_train, y_test) = svm_train_test_split(id, test_size, scaleMode)
  plt.clf()
  plt.figure(figsize=(14, 4))
  plt.subplot(121)
  svm_train_test_method(X_train, y_train, id, scaleMode)

  plt.subplot(122)
  svm_train_test_method(X_test, y_test, id, scaleMode)

  trainTestImg = svm_img_to_base64(plt)

  return (json.dumps({'trainTestImg': trainTestImg}), 200)

"""# Train the Model"""

#Import svm model
#Create a svm Classifier
# Hyperparameters can be changed/tuned based on the compile and fit function arguments defined at -
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

def svm_modelTrain(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size

  (X_train, X_test, y_train, _) = svm_train_test_split(id, test_size, scaleMode)
  clf = svm.SVC(kernel='linear', C= 10) # Linear Kernel
  #Train the model using the training sets
  clf.fit(X_train, y_train)

  """# Predict the Model on Test data"""

  #Predict the response for test dataset
  y_pred = clf.predict(X_test)
  return (y_pred, clf)

"""# Visualize the Solution"""
def svm_solution(id, test_size, scaleMode):
  x_index = 2
  y_index = 3
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size

  (_, clf) = svm_modelTrain(id, test_size, scaleMode)
  (_, y, feature_names, X_plot) = svm_scaling(id, scaleMode)
  plt.clf()
  plt.figure(figsize=(8, 6))
  formatter = plt.FuncFormatter(lambda i, *args: svm_get_label_mapping(int(i)))
  plt.scatter(X_plot[:, x_index], X_plot[:, y_index], c=y)
  plt.colorbar(ticks=[1, -1], format=formatter)
  plt.xlabel(feature_names[x_index])
  plt.ylabel(feature_names[y_index])

  ax = plt.gca()
  xlim = ax.get_xlim()
  ylim = ax.get_ylim()

  # create grid to evaluate model
  xx = np.linspace(xlim[0], xlim[1], 30)
  yy = np.linspace(ylim[0], ylim[1], 30)
  YY, XX = np.meshgrid(yy, xx)
  xy = np.vstack([XX.ravel(), YY.ravel()]).T
  Z = clf.decision_function(xy).reshape(XX.shape)

  # plot decision boundary and margins
  ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
  # plot support vectors
  ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
  solutionImg = svm_img_to_base64(plt)

  return (json.dumps({'solutionImg': solutionImg}), 200)

"""# Plot Confusion Matrix"""
def svm_confusion_matrix_method(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.clf()
  figure(figsize=(10, 8), dpi=80)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  return (svm_img_to_base64(plt))

def svm_confusion_matrix(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size

  (_, _, _, y_test) = svm_train_test_split(id, test_size, scaleMode)
  (y_pred, _) = svm_modelTrain(id, test_size, scaleMode)
  cm = confusion_matrix(y_test, y_pred)
  cm_labels = ["malignant", "benign"]
  confMatrix = svm_confusion_matrix_method(cm, cm_labels)  

  return (json.dumps({'confMatrix': confMatrix}), 200)

"""# Evaluate the Model on Test Data"""
def svm_evaluation(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  (_, _, _, y_test) = svm_train_test_split(id, test_size, scaleMode)
  (y_pred, _) = svm_modelTrain(id, test_size, scaleMode)
  # Model Accuracy: how often is the classifier correct?
  modelAccuracy = str(metrics.accuracy_score(y_test, y_pred))
  # Model Precision: what percentage of positive tuples are labeled as such?
  modelPrecision = str(metrics.precision_score(y_test, y_pred))
  # Model Recall: what percentage of positive tuples are labelled as such?
  modelRecall = str(metrics.recall_score(y_test, y_pred))
  return (json.dumps({'Model Accuracy:': modelAccuracy, 'Model Precision:': modelPrecision, 'Model Recall:': modelRecall}), 200)

def svm_img_to_base64(plt):
  chart = BytesIO()
  plt.savefig(chart, format = 'png')
  chart.seek(0)
  output_chart = base64.b64encode(chart.getvalue())
  return str(output_chart, 'utf-8')