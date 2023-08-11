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
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from uuid import uuid4

csv_file = {}
X_database = {}
Y_database = {}
# Phase (1). data collection (file upload):
def lgr_fileUpload(files):
  #  if request.method == 'POST':
  if files['file'].content_type != 'text/csv':
    return (json.dumps({'message': 'File must be a CSV'}), 400)
  
  df = pd.read_csv(files.get('file'))

  uuid = str(uuid4())

  csv_file[uuid] = df
  #print(df)
  return (json.dumps({'id': uuid}), 200)

# Phase (2). data preprocessing (removing missing values and scaling, return processed dataframe preview): 
def lgr_rmMissingvalues(id):
  df = csv_file[id]
  df_new = df.dropna()
  return ((df_new.to_json()), 200)

def lgr_scaling(id, scaleMode):
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

def lgr_explore(id, scaleMode):
  (_, _, feature_names, _) = lgr_scaling(id, scaleMode)
  index_as_list = feature_names.tolist()
  return (json.dumps({'Features: ': index_as_list}), 200)


  # Split the Dataset into Training and Test Set
def lgr_spliting(id, test_size, random_state, scaleMode="normalization"):
  (X, y, _, _) = lgr_scaling(id, scaleMode)
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  X_train_split, X_test_split, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return (X_train_split, X_test_split, y_train, y_test)

def lgr_getShape(id, test_size, random_state, scaleMode="normalization"):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (X_train_split, X_test_split, y_train, y_test) = lgr_spliting(id, test_size=test_size, random_state=random_state, scaleMode=scaleMode)
  XTrainShape = X_train_split.shape
  XTestShape = X_test_split.shape
  yTrainShape = y_train.shape
  yTestShape = y_test.shape
  return (json.dumps({'X_train shape: ': XTrainShape, 'X_test shape: ': XTestShape, 'y_train shape: ': yTrainShape, 'y_test shape: ': yTestShape}), 200)

# Phase 4: model training
# proving X_train, X_test, regressor, and Y_pred
def lgr_pre_train(id, test_size, random_state):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (X_train, X_test, y_train, y_test) = lgr_spliting(id, test_size, random_state)
  # Create a 2D array for training and test data to make it compatible with
  # scikit-learn (This is specific to scikit-learn because of the way it accepts input data)
  # Create Logistic Regression object
  clf = LogisticRegression()

# Train Logistic Regression Classifer
  clf.fit(X_train, y_train)
  #Predict the response for test dataset
  y_pred = clf.predict(X_test)

  return (X_train, X_test, y_train, y_test, y_pred)

def lgr_confusionMatrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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
  return (lgr_img_to_base64(plt))

def lgr_makeConfusionMatrix(id, test_size, random_state):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (_, _, _, y_test, y_pred) = lgr_pre_train(id, test_size, random_state)
  cm = confusion_matrix(y_test, y_pred)
  cm_labels = ["malignant", "benign"]
  confsMatrix = lgr_confusionMatrix(cm, cm_labels)  
  return (json.dumps({'confsMatrix': confsMatrix}), 200)

# Phase 5: accuracy
def lgr_accuracy(id, test_size, random_state):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (_, _, _, y_test, y_pred) = lgr_pre_train(id, test_size, random_state)
  modelAccuracy = str(metrics.accuracy_score(y_test, y_pred))
  modelPrecision = str(metrics.precision_score(y_test, y_pred))
  modelRecall = str(metrics.recall_score(y_test, y_pred))
  # Evaluate the quality of the training (Generate Evaluation Metrics)
  return (json.dumps({'Model Accuracy:': modelAccuracy, 'Model Precision:': modelPrecision, 'Model Recall:': modelRecall}), 200)


def lgr_img_to_base64(plt):
  chart = BytesIO()
  plt.savefig(chart, format = 'png')
  chart.seek(0)
  output_chart = base64.b64encode(chart.getvalue())
  return str(output_chart, 'utf-8')