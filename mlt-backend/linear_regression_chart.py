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
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
from uuid import uuid4

csv_file = {}
# Phase (1). data collection (file upload):
def lr_fileUpload(files):
  #  if request.method == 'POST':
  if files['file'].content_type != 'text/csv':
    return (json.dumps({'message': 'File must be a CSV'}), 400)
  
  df = pd.read_csv(files.get('file'))

  uuid = str(uuid4())

  csv_file[uuid] = df
  #print(df)
  return (json.dumps({'id': uuid}), 200)

# Phase (2). data preprocessing (removing missing values and scaling, return processed dataframe preview): 
def lr_rmMissingvalues(id):
  df = csv_file[id]
  df_new = df.dropna()
  return ((df_new.to_json()), 200)

# Phase 3: data visualization (whole data visualization, training data visualization, and testing data visualization, return charts)
def lr_getParas(id, data):
  df = csv_file[id]
  df_new = df.dropna()
  x_column = df_new[data['x_index']]
  y_column = df_new[data['y_index']]
  scaleMode = str(data['scaleMode'])
  return (x_column, y_column, scaleMode)

def lr_getColumns(id, data):
  (x_column, y_column, scaleMode) = lr_getParas(id, data)
  X = x_column.to_numpy()
  Y = y_column.to_numpy()
  
  if "standardization" in scaleMode.lower():
    # standardization
    Y_scaled = (Y - np.mean(Y)) / np.std(Y)
      
  elif "normalization" in scaleMode.lower():
    # normalization
    Y_scaled = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
  return (X, Y_scaled)

def lr_scatterImg(id, data):
  (X, Y_scaled) = lr_getColumns(id, data)
  plt.clf()
  figure(figsize=(8, 6), dpi=80)
  plt.scatter(X, Y_scaled)
  plt.title("Visualize the full Dataset")
  plt.xlabel('X')
  plt.ylabel('Y')
  imgScatter = lr_img_to_base64(plt)

  return (json.dumps({'imgScatter': imgScatter}), 200)

  # Split the Dataset into Training and Test Set
def lr_spliting(id, data, test_size, random_state): 
  (x_column, y_column, scaleMode) = lr_getParas(id, data)
  X = x_column.to_numpy()
  Y = y_column.to_numpy()
  
  if "standardization" in scaleMode.lower():
    # standardization
    Y_scaled = (Y - np.mean(Y)) / np.std(Y)
      
  elif "normalization" in scaleMode.lower():
    # normalization
    Y_scaled = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
  X_train_split, X_test_split, Y_train, Y_test = train_test_split(X, Y_scaled, test_size = test_size, random_state = random_state)
  return (X_train_split, X_test_split, Y_train, Y_test)

def lr_train_test_imgs(id, data, test_size, random_state):
  print("test return")
  (X_train, X_test, Y_train, Y_test) = lr_spliting(id, data, test_size = float(test_size), random_state = int(random_state))
  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  X_test = np.array(X_test)
  Y_test = np.array(Y_test)
  plt.clf()
  figure(figsize=(15, 6), dpi=80)
  plt.subplot(1, 2, 1) # row 1, col 2 index 1
  plt.scatter(X_train, Y_train)
  plt.title("Training Data")
  plt.xlabel('X_train')
  plt.ylabel('Y_train')

  plt.subplot(1, 2, 2) # index 2
  plt.scatter(X_test, Y_test)
  plt.title("Testing Data")
  plt.xlabel('X_test')
  plt.ylabel('Y_test')
  plt.tight_layout()
  trainTestImg = lr_img_to_base64(plt)
  

  #return (json.dumps({'trainTestImg': trainTestImg}), 200)

# Phase 4: model training
# proving X_train, X_test, regressor, and Y_pred
def lr_pre_train(id, data, test_size, random_state):
  (X_train_split, X_test_split, Y_train, Y_test) = lr_spliting(id, data, test_size = float(test_size), random_state = int(random_state))
  # Create a 2D array for training and test data to make it compatible with
  # scikit-learn (This is specific to scikit-learn because of the way it accepts input data)
  X_train = X_train_split.reshape(-1, 1)
  X_test = X_test_split.reshape(-1, 1)

  # Initialize Model

  regressor = SGDRegressor()

  # Run Model Training
  regressor.fit(X_train, Y_train)

  # Predict on the Test Data
  Y_pred = regressor.predict(X_test)

  return (X_train, X_test, X_train_split, X_test_split, regressor, Y_train, Y_test, Y_pred)

  # plt.tight_layout()
def lr_modelTraining(id, data, test_size, random_state):
  (_, _, _, X_test_split, _, _, Y_test, Y_pred) = lr_pre_train(id, data, test_size = test_size, random_state = random_state)

  # Plot the predictions and the original test data
  plt.clf()
  figure(figsize=(8, 6), dpi=80)
  plt.plot(X_test_split, Y_test, 'go', label='True data', alpha=0.5)
  plt.plot(X_test_split, Y_pred, '--', label='Predictions', alpha=0.5)
  
  plt.title("Prediction")
  
  plt.legend(loc='best')

  return (json.dumps({'imgPrediction':lr_img_to_base64(plt)}), 200)

# Phase 5: accuracy
def lr_accuracy(id, data, test_size, random_state):
  (_, _, _, _, _, _, Y_test, Y_pred) = lr_pre_train(id, data, test_size = test_size, random_state = random_state)
  meanAbErr = str(metrics.mean_absolute_error(Y_test, Y_pred))
  meanSqErr = str(metrics.mean_squared_error(Y_test, Y_pred))
  rootMeanSqErr = str(np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
  # Evaluate the quality of the training (Generate Evaluation Metrics)
  return (json.dumps({'Mean Absolute Error:': meanAbErr, 'Mean Squared Error:': meanSqErr, 'Root Mean Squared Error:': rootMeanSqErr}), 200)


def lr_img_to_base64(plt):
  chart = BytesIO()
  plt.savefig(chart, format = 'png')
  chart.seek(0)
  output_chart = base64.b64encode(chart.getvalue())
  return str(output_chart, 'utf-8')

