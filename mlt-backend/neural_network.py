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
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from uuid import uuid4

csv_file = {}
X_database = {}
Y_database = {}
# Phase (1). data collection (file upload):
def nr_fileUpload(files):
  #  if request.method == 'POST':
  if files['file'].content_type != 'text/csv':
    return (json.dumps({'message': 'File must be a CSV'}), 400)
  
  df = pd.read_csv(files.get('file'))

  uuid = str(uuid4())

  csv_file[uuid] = df
  #print(df)
  return (json.dumps({'id': uuid}), 200)

# Phase (2). data preprocessing (removing missing values and scaling, return processed dataframe preview): 
def nr_rmMissingvalues(id):
  df = csv_file[id]
  df_new = df.dropna()
  return ((df_new.to_json()), 200)

def nr_scaling(id, scaleMode):
  df = csv_file[id]
  df_new = df.dropna()
  X = df_new.iloc[:, 2:]
  y = df_new.iloc[:, 1]
  feature_names = df.columns
  labels = feature_names[1]

  # Map labels from ['M', 'B'] to [0, 1] space
  y = np.where(y=='M', 0, 1)
  if scaleMode == "standardization":
    # standardization
    X = (X - np.mean(X)) / np.std(X)
  elif scaleMode == "normalization":
    # normalization
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
  # error catching logic required here
  return (X, y, feature_names, labels)
# Phase 3: data visualization (whole data visualization, training data visualization, and testing data visualization, return charts)

def nr_explore(id, scaleMode):
  (_, _, feature_names, _) = nr_scaling(id, scaleMode)
  index_as_list = feature_names.tolist()
  return (json.dumps({'Features: ': index_as_list}), 200)


  # Split the Dataset into Training and Test Set
def nr_spliting(id, test_size, scaleMode):
  (X, y, _, _) = nr_scaling(id, scaleMode)
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  X_train_split, X_test_split, y_train, y_test = train_test_split(X, y, test_size=test_size)
  return (X_train_split, X_test_split, y_train, y_test)

def nr_getShape(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  (X_train_split, X_test_split, y_train, y_test) = nr_spliting(id, test_size=test_size, scaleMode=scaleMode)
  XTrainShape = X_train_split.shape
  XTestShape = X_test_split.shape
  yTrainShape = y_train.shape
  yTestShape = y_test.shape
  return (json.dumps({'X_train shape: ': XTrainShape, 'X_test shape: ': XTestShape, 'y_train shape: ': yTrainShape, 'y_test shape: ': yTestShape}), 200)

# Phase 4: model training
def nr_model_generate(id, test_size, scaleMode):
  (X_train, _, y_train, _) = nr_spliting(id, test_size, scaleMode)
  mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)
  mlp.fit(X_train,y_train)
  return mlp

# Predict on Test Data
def nr_predict(id, test_size, scaleMode):
  (_, X_test, _, _) = nr_spliting(id, test_size, scaleMode)
  mlp = nr_model_generate(id, test_size, scaleMode)
  y_pred = mlp.predict(X_test)
  return y_pred

# Plot Confusion Matrix

def nr_confusion_matrix_method(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
  return (nr_img_to_base64(plt))

def nr_confusion_matrix_plot(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  (_, _, _, y_test) = nr_spliting(id, test_size, scaleMode)
  y_pred = nr_predict(id, test_size, scaleMode)
  cm = confusion_matrix(y_test, y_pred)
  cm_labels = ["malignant", "benign"]
  confsMatrix = nr_confusion_matrix_method(cm, cm_labels)  
  return (json.dumps({'confsMatrix': confsMatrix}), 200)


# Phase 5: Evaluate on Test data
def nr_evaluate(id, test_size, scaleMode):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  (_, _, _, y_test) = nr_spliting(id, test_size, scaleMode)
  y_pred = nr_predict(id, test_size, scaleMode)
  modelAccuracy = str(metrics.accuracy_score(y_test, y_pred))
  modelPrecision = str(metrics.precision_score(y_test, y_pred))
  modelRecall = str(metrics.recall_score(y_test, y_pred))
  # Evaluate the quality of the training (Generate Evaluation Metrics)
  return (json.dumps({'Model Accuracy:': modelAccuracy, 'Model Precision:': modelPrecision, 'Model Recall:': modelRecall}), 200)


# Overfitting and the bias-variance tradeoff

# High bias -> less complex model -> underfitting
# High variance -> more complex model -> overfitting

# Extract a subset from train and test datasets to showcase the effects of under and overfitting using models of different sizes
def nr_three_models(id, test_size, scaleMode):
  (X_train, X_test, y_train, y_test) = nr_spliting(id, test_size, scaleMode)
  #X_train = X_train[:20, :]
  #y_train = y_train[:20]

  #X_test = X_test[:20, :]
  #y_test = y_test[:20]

  # Less complex model - Less number of neurons
  # underfit model
  mlp_underfit = MLPClassifier(hidden_layer_sizes=(1), alpha=0.0001, learning_rate_init=0.0001, activation='relu', solver='adam', max_iter=1000, shuffle=True, random_state=0)
  mlp_underfit.fit(X_train, y_train)

  # More complex model - More number of neurons 
  # overfit model
  mlp_overfit = MLPClassifier(hidden_layer_sizes=(900), alpha=0.0001, learning_rate_init=0.0001, activation='relu', solver='adam', max_iter=1000, shuffle=True, random_state=0)
  mlp_overfit.fit(X_train, y_train)

  # Best fit model - Ideal number of neurons
  mlp_bestfit = MLPClassifier(hidden_layer_sizes=(8), alpha=0.0001, learning_rate_init=0.0001, activation='relu', solver='adam', max_iter=1000, shuffle=True, random_state=0)
  mlp_bestfit.fit(X_train, y_train)

  # Predict these models on Test Data

  y_pred_underfit = mlp_underfit.predict(X_test)
  y_pred_overfit = mlp_overfit.predict(X_test)
  y_pred_bestfit = mlp_bestfit.predict(X_test)
  return (y_pred_underfit, y_pred_overfit, y_pred_bestfit, mlp_underfit, mlp_overfit, mlp_bestfit)

def nr_train_data_errors(id, test_size, scaleMode):
  # Evaluate these models on Test data

  # Errors on Train data
  (X_train, _, y_train, _) = nr_spliting(id, test_size, scaleMode)
  (_, _, _, mlp_underfit, mlp_overfit, mlp_bestfit) = nr_three_models(id, test_size, scaleMode)
  train_mean_ab_err_best = str(metrics.mean_absolute_error(y_train, mlp_bestfit.predict(X_train)))
  train_mean_ab_err_underfit = str(metrics.mean_absolute_error(y_train, mlp_underfit.predict(X_train)))
  train_mean_ab_err_overfit = str(metrics.mean_absolute_error(y_train, mlp_overfit.predict(X_train)))

  train_mean_squared_err_best = str(metrics.mean_squared_error(y_train, mlp_bestfit.predict(X_train)))
  train_mean_squared_err_underfit = str(metrics.mean_squared_error(y_train, mlp_underfit.predict(X_train)))
  train_mean_squared_err_overfit = str(metrics.mean_squared_error(y_train, mlp_overfit.predict(X_train)))

  train_rt_mean_squared_err_best = str(np.sqrt(metrics.mean_squared_error(y_train, mlp_bestfit.predict(X_train))))
  train_rt_mean_squared_err_underfit = str(np.sqrt(metrics.mean_squared_error(y_train, mlp_underfit.predict(X_train))))
  train_rt_mean_squared_err_overfit = str(np.sqrt(metrics.mean_squared_error(y_train, mlp_overfit.predict(X_train))))
  return (json.dumps({
    'Mean Absolute Error - Best fit model: ': train_mean_ab_err_best, 
    'Mean Absolute Error - Underfit model: ': train_mean_ab_err_underfit, 
    'Mean Absolute Error - Overfit model:': train_mean_ab_err_overfit,
    'Mean Squared Error - Best fit model: ': train_mean_squared_err_best,
    'Mean Squared Error - Underfit model: ': train_mean_squared_err_underfit, 
    'Mean Squared Error - Overfit model: ': train_mean_squared_err_overfit,
    'Root Mean Squared Error - Best fit model: ': train_rt_mean_squared_err_best, 
    'Root Mean Squared Error - Underfit model: ': train_rt_mean_squared_err_underfit, 
    'Root Mean Squared Error - Overfit model: ': train_rt_mean_squared_err_overfit}), 200)

def nr_test_data_errors(id, test_size, scaleMode):
  #Errors on Test data
  (_, _, _, y_test) = nr_spliting(id, test_size, scaleMode)
  (y_pred_underfit, y_pred_overfit, y_pred_bestfit, _, _, _) = nr_three_models(id, test_size, scaleMode)
  test_mean_ab_err_best = str(metrics.mean_absolute_error(y_test, y_pred_bestfit))
  test_mean_ab_err_underfit = str(metrics.mean_absolute_error(y_test, y_pred_underfit))
  test_mean_ab_err_overfit = str(metrics.mean_absolute_error(y_test, y_pred_overfit))

  test_mean_squared_err_best = str(metrics.mean_squared_error(y_test, y_pred_bestfit))
  test_mean_squared_err_underfit = str(metrics.mean_squared_error(y_test, y_pred_underfit)) 
  test_mean_squared_err_overfit = str(metrics.mean_squared_error(y_test, y_pred_overfit))

  test_rt_mean_squared_err_best = str(np.sqrt(metrics.mean_squared_error(y_test, y_pred_bestfit)))
  test_rt_mean_squared_err_underfit = str(np.sqrt(metrics.mean_squared_error(y_test, y_pred_underfit)))
  test_rt_mean_squared_err_overfit = str(np.sqrt(metrics.mean_squared_error(y_test, y_pred_overfit)))

  return (json.dumps({
    'Mean Absolute Error - Best fit model: ': test_mean_ab_err_best, 
    'Mean Absolute Error - Underfit model: ': test_mean_ab_err_underfit, 
    'Mean Absolute Error - Overfit model:': test_mean_ab_err_overfit,
    'Mean Squared Error - Best fit model: ': test_mean_squared_err_best, 
    'Mean Squared Error - Underfit model: ': test_mean_squared_err_underfit, 
    'Mean Squared Error - Overfit model: ': test_mean_squared_err_overfit,
    'Root Mean Squared Error - Best fit model: ': test_rt_mean_squared_err_best, 
    'Root Mean Squared Error - Underfit model: ': test_rt_mean_squared_err_underfit, 
    'Root Mean Squared Error - Overfit model: ': test_rt_mean_squared_err_overfit,}), 200)

"""## Training errors 
Underfit model > Best fit model > Overfit model

##Testing errors 
Overfit model <> Underfit model > Best fit model

## Overfit model errors
Overfit Model Testing errors > Overfit Model Training errors

## Bestfit model errors
Testing errors ~ Training errors
"""

def nr_img_to_base64(plt):
  chart = BytesIO()
  plt.savefig(chart, format = 'png')
  chart.seek(0)
  output_chart = base64.b64encode(chart.getvalue())
  return str(output_chart, 'utf-8')