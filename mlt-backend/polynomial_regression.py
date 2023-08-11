import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure
from sklearn import metrics
from uuid import uuid4
from operator import itemgetter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import SGDRegressor
from sklearn import metrics

csv_file = {}
X_database = {}
Y_database = {}
# Phase (1). data collection (file upload):
def poly_fileUpload(files):
  #  if request.method == 'POST':
  if files['file'].content_type != 'text/csv':
    return (json.dumps({'message': 'File must be a CSV'}), 400)
  
  df = pd.read_csv(files.get('file'))

  uuid = str(uuid4())

  csv_file[uuid] = df
  #print(df)
  return (json.dumps({'id': uuid}), 200)

# Phase (2). data preprocessing (removing missing values and scaling, return processed dataframe preview): 
def poly_rmMissingvalues(id):
  df = csv_file[id]
  df_new = df.dropna()
  return ((df_new.to_json()), 200)

def poly_scaling(id, scaleMode):
  df = csv_file[id]
  df_new = df.dropna()
  X = df_new.atemp.to_numpy()
  Y = df_new.cnt.to_numpy()
  if scaleMode == "standardization":
    # standardization
    Y = (Y - np.mean(Y)) / np.std(Y)
  elif scaleMode == "normalization":
    # normalization
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
  # error catching logic required here
  return (X, Y)

# Phase 3: data visualization (whole data visualization, training data visualization, and testing data visualization, return charts)

def poly_scatterImg(id, scaleMode):
  (X, Y) = poly_scaling(id, scaleMode)
  plt.clf()
  figure(figsize=(8, 6), dpi=80)
  plt.scatter(X, Y)
  
  plt.title("Visualize the full Dataset")
  plt.xlabel('X')
  plt.ylabel('Y')

  return (json.dumps({'imgScatter':poly_img_to_base64(plt)}), 200)


  # Split the Dataset into Training and Test Set
def poly_spliting(id, test_size=0.2, random_state=0, scaleMode="normalization"):
  (X, Y) = poly_scaling(id, scaleMode)
  X_train_split, X_test_split, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
  return (X_train_split, X_test_split, Y_train, Y_test)

def poly_train_test_imgs(id, test_size, random_state):
  (X_train, X_test, Y_train, Y_test) = poly_spliting(id, float(test_size), int(random_state))
  enumerate_x = enumerate(X_test)
  sorted_pairs = sorted(enumerate_x, key=itemgetter(1))
  sorted_indices = [index for index, element in sorted_pairs]
  X_test = sorted(X_test)
  Y_test = itemgetter(*sorted_indices)(Y_test)
  X_test = np.array(X_test)
  Y_test = np.array(Y_test)
  plt.clf()
  figure(figsize=(18, 6), dpi=80)

  plt.subplot(1, 2, 1) # row 1, col 2 index 1
  plt.scatter(X_train, Y_train)
  plt.title("Training Data")
  plt.xlabel('X_train')
  plt.ylabel('Y_train')

  plt.subplot(1, 2, 2) # index 2
  plt.scatter(X_test, Y_test)
  plt.title("Test Data")
  plt.xlabel('X_test')
  plt.ylabel('Y_test')

  plt.tight_layout()
  trainTestImg = poly_img_to_base64(plt)

  return (json.dumps({'trainTestImg': trainTestImg}), 200)


"""# Initialize the Model"""
def poly_pre_train(id, test_size, random_state):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (X_train, X_test, Y_train, Y_test) = poly_spliting(id, test_size, random_state)
  X_train = X_train.reshape(-1, 1)
  X_test = X_test.reshape(-1, 1)
  X_train.shape, X_test.shape
  degree=3
  polyreg_fit = make_pipeline(PolynomialFeatures(degree), SGDRegressor(alpha = .00001, max_iter=500))
  polyreg_fit.fit(X_train, Y_train)
  return (X_train, X_test, polyreg_fit, Y_train, Y_test)

def poly_modelTraining(id, test_size, random_state):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (_, X_test, polyreg_fit, _, Y_test)=poly_pre_train(id, test_size, random_state)
  X_test_sorted = sorted(X_test)
  Y_pred = polyreg_fit.predict(X_test_sorted)
  plt.clf()
  figure(figsize=(8, 6), dpi=80)
  plt.plot(X_test, Y_test, 'go', label='True data')
  plt.plot(X_test_sorted, Y_pred, '--', label='Predictions')
  plt.legend(loc='best')
  return (json.dumps({'imgPrediction':poly_img_to_base64(plt)}), 200)

# Phase 5: accuracy
def poly_accuracy(id, test_size, random_state):
  if test_size is not None:
    test_size = float(test_size)
  else:
    test_size = test_size
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  (_, X_test, polyreg_fit, _, Y_test) = poly_pre_train(id, test_size, random_state)
  X_test_sorted = sorted(X_test)
  Y_pred = polyreg_fit.predict(X_test_sorted)
  meanAbErr = str(metrics.mean_absolute_error(Y_test, Y_pred))
  meanSqErr = str(metrics.mean_squared_error(Y_test, Y_pred))
  rootMeanSqErr = str(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
  # Evaluate the quality of the training (Generate Evaluation Metrics)
  return (json.dumps({'Mean Absolute Error:': meanAbErr, 'Mean Squared Error:': meanSqErr, 'Root Mean Squared Error:': rootMeanSqErr}), 200)

def poly_img_to_base64(plt):
  chart = BytesIO()
  plt.savefig(chart, format = 'png')
  chart.seek(0)
  output_chart = base64.b64encode(chart.getvalue())
  return str(output_chart, 'utf-8')
