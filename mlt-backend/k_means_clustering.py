import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
from io import BytesIO
import base64
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from uuid import uuid4

csv_file = {}
X_database = {}
Y_database = {}
# Phase (1). data collection (file upload):
def km_fileUpload(files):
  #  if request.method == 'POST':
  if files['file'].content_type != 'text/csv':
    return (json.dumps({'message': 'File must be a CSV'}), 400)
  
  df = pd.read_csv(files.get('file'))

  uuid = str(uuid4())

  csv_file[uuid] = df
  #print(df)
  return (json.dumps({'id': uuid}), 200)

# Phase (2). data preprocessing (removing missing values and scaling, return processed dataframe preview): 
def km_rmMissingvalues(id):
  df = csv_file[id]
  df_new = df.dropna()
  return ((df_new.to_json()), 200)

def km_scaling(id, scaleMode):
  df = csv_file[id]
  df_new = df.dropna()
  X = df_new.iloc[:, 2:].to_numpy()
  y = df_new.iloc[:, 1].to_numpy()

  # Map labels from ['M', 'B'] to [0, 1] space
  y = np.where(y=='M', 0, 1)
  if scaleMode == "standardization":
    # standardization
    X = (X - np.mean(X)) / np.std(X)
  elif scaleMode == "normalization":
    # normalization
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
  # error catching logic required here
  return (X, y)

# Phase 3: data visualization (whole data visualization, training data visualization, and testing data visualization, return charts)

def km_scatterImg(id, scaleMode):
  (X, _) = km_scaling(id, scaleMode)
  plt.clf()
  plt.scatter(
    X[:, 0], X[:, 1],
    c='white', marker='o',
    edgecolor='black', s=50
  )
  imgScatter = km_img_to_base64(plt)
  return (json.dumps({'imgScatter': imgScatter}), 200)

# Phase 4: model training
def km_generate_model(id, random_state, scaleMode):
  (X, _) = km_scaling(id, scaleMode)
  if random_state is not None:
    random_state = int(random_state)
  else:
    random_state = random_state
  km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=random_state
  )
  y_km = km.fit_predict(X)
  return (km, y_km)

def km_plot_cluster(id, random_state, scaleMode):
  (km, y_km) = km_generate_model(id, random_state, scaleMode)
  (X, _) = km_scaling(id, scaleMode)
  plt.clf()
  # plot the 3 clusters
  plt.scatter(
    X[y_km == 0, 0], X[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
  )

  plt.scatter(
    X[y_km == 1, 0], X[y_km == 1, 1],
    s=50, c='orange',
    marker='o', edgecolor='black',
    label='cluster 2'
  )

  # plot the centroids
  plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
  )
  plt.legend(scatterpoints=1)
  plt.grid()
  KMClusterImg = km_img_to_base64(plt)
  return (json.dumps({'KMClusterImg': KMClusterImg}), 200)

"""# Estimate the number of clusters in the Dataset (Useful for real world data where it's hard to know the actual number of clusters beforehand)"""
def km_estimate(id, random_state, scaleMode):
  (X, _) = km_scaling(id, scaleMode=scaleMode)
  # calculate distortion for a range of number of clusters
  distortions = []
  for i in range(1, 11):
    # Update hyperparameters like n_clusters, n_init, max_iter, tol etc. following details here -
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    km = KMeans(
        n_clusters=i, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=random_state
    )
    km.fit(X)
    distortions.append(km.inertia_)

  # plot
  plt.clf()
  plt.plot(range(1, 11), distortions, marker='o')
  plt.xlabel('Number of clusters')
  plt.ylabel('Distortion')
  KMEstimateImg = km_img_to_base64(plt)
  return (json.dumps({'KMEstimateImg': KMEstimateImg}), 200)


# Phase 5: accuracy
def km_accuracy(id, random_state, scaleMode):
  (y_km, _) = km_generate_model(id, random_state, scaleMode)
  # Within-Cluster-Sum-of-Squares (WCSS)
  # Measures the sum of squared distances between each point and the
  # centroid of its assigned cluster. Lower values of WCSS indicate better clustering performance
  wcss = str(y_km.inertia_)

  return (json.dumps({'Within-Cluster-Sum-of-Squares (WCSS):': wcss}), 200)

def km_img_to_base64(plt):
  chart = BytesIO()
  plt.savefig(chart, format = 'png')
  chart.seek(0)
  output_chart = base64.b64encode(chart.getvalue())
  return str(output_chart, 'utf-8')