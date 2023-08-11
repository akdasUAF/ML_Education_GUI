from linear_regression_chart import *
from logistic_regression import *
from polynomial_regression import *
from k_means_clustering import *
from svm import *
from neural_network import *

from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
app.debug = True
CORS(app, resources={r"/*": {"origins": "*"}})

#---------------------------------------------------------------------#
# Linear Regression
@app.route('/datasets/linear_regression',methods=['POST'])
def lr_getUploadedFile():
    (data, res) = lr_fileUpload(request.files)
    return make_response(data, res)

@app.route('/datasets/<id>/linear_regression/missing_values',methods=['GET'])
def lr_removeMissingValues(id):
    rmResult = lr_rmMissingvalues(id)
    return make_response(rmResult)

@app.route('/datasets/<id>/linear_regression/get_columns',methods=['POST'])
def lr_post_columns(id):
    # Parse the request data as JSON
    request_data = json.loads(request.data)

    # Call the lr_getColumns function
    (X, Y_scaled) = lr_getColumns(id, request_data)

    # Convert the NumPy arrays to lists and return the JSON response
    return jsonify({'data': (X.tolist(), Y_scaled.tolist())})

@app.route('/datasets/<id>/linear_regression/scatter',methods=['GET'])
def lr_make_scatter(id):
    x_index = request.args.get('x_index')
    y_index = request.args.get('y_index')
    scaleMode = request.args.get('scaleMode')
    #data = {
    #    'x_index': x_index,
    #    'y_index': y_index,
    #    'scaleMode': scaleMode
    #}
    scatterChart = lr_scatterImg(id, x_index, y_index, scaleMode)
    return make_response(scatterChart)

@app.route('/datasets/<id>/linear_regression/train_test_datasets', methods = ['GET'])
def lr_trainTest_imgs(id, data):
    test_size = request.args.get('test_size')
    random_state = request.args.get('random_state')
    
    print(id, data, test_size, random_state)
    charts = lr_train_test_imgs(id, data, test_size, random_state)
    return make_response(charts)

@app.route('/datasets/<id>/<data>/linear_regression/model_training_result', methods = ['GET'])
def lr_prediction(id, data):
    pre_chart = lr_modelTraining(id, data, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(pre_chart)

@app.route('/datasets/<id>/<data>/linear_regression/calculation', methods = ['GET'])
def lr_calculations(id, data):
    results = lr_accuracy(id, data, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(results)

#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
# Logistic Regression
@app.route('/datasets/logistic_regression',methods=['POST'])
def lgr_getUploadedFile():
    (data, res) = lgr_fileUpload(request.files)
    return make_response(data, res)

@app.route('/datasets/<id>/logistic_regression/missing_values',methods=['GET'])
def lgr_removeMissingValues(id):
    rmResult = lgr_rmMissingvalues(id)
    return make_response(rmResult)

@app.route('/datasets/<id>/logistic_regression/get_features', methods = ['GET'])
def lgr_show_features(id):
    featuresResponse = lgr_explore(id, request.args.get('scaleMode'))
    return make_response(featuresResponse)

@app.route('/datasets/<id>/logistic_regression/datasets_shapes', methods = ['GET'])
def lgr_getShapes(id):
    shapes = lgr_getShape(id, request.args.get('test_size'), request.args.get('random_state'), request.args.get('scaleMode'))
    return make_response(shapes)

@app.route('/datasets/<id>/logistic_regression/model_training_result', methods = ['GET'])
def lgr_getMatrix(id):
    confMatrixImg = lgr_makeConfusionMatrix(id, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(confMatrixImg)

@app.route('/datasets/<id>/logistic_regression/calculation', methods = ['GET'])
def lgr_calculations(id):
    results = lgr_accuracy(id, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(results)
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Polynomial Regression
@app.route('/datasets/polynomial_regression',methods=['POST'])
def poly_getUploadedFile():
    (data, res) = poly_fileUpload(request.files)
    return make_response(data, res)

@app.route('/datasets/<id>/polynomial_regression/missing_values',methods=['GET'])
def poly_removeMissingValues(id):
    rmResult = poly_rmMissingvalues(id)
    return make_response(rmResult)

@app.route('/datasets/<id>/polynomial_regression/scatter', methods = ['GET'])
def poly_make_scatter(id):
    scatterChart = poly_scatterImg(id, request.args.get('scaleMode'))
    return make_response(scatterChart)

@app.route('/datasets/<id>/polynomial_regression/train_test_results', methods = ['GET'])
def poly_traintest_imgs(id):
    charts = poly_train_test_imgs(id, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(charts)

@app.route('/datasets/<id>/polynomial_regression/model_training_result', methods = ['GET'])
def poly_prediction(id):
    pre_chart = poly_modelTraining(id, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(pre_chart)

@app.route('/datasets/<id>/polynomial_regression/calculation', methods = ['GET'])
def poly_calculations(id):
    results = poly_accuracy(id, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(results)
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# K-Means Clustering
@app.route('/datasets/k_means_clustering',methods=['POST'])
def km_getUploadedFile():
    (data, res) = km_fileUpload(request.files)
    return make_response(data, res)

@app.route('/datasets/<id>/k_means_clustering/missing_values',methods=['GET'])
def km_removeMissingValues(id):
    rmResult = km_rmMissingvalues(id)
    return make_response(rmResult)

@app.route('/datasets/<id>/k_means_clustering/scatter', methods = ['GET'])
def km_make_scatter(id):
    scatterChart = km_scatterImg(id, request.args.get('scaleMode'))
    return make_response(scatterChart)

@app.route('/datasets/<id>/k_means_clustering/train_test_results', methods = ['GET'])
def km_traintest_imgs(id):
    charts = km_plot_cluster(id, request.args.get('random_state'), request.args.get('scaleMode'))
    return make_response(charts)

@app.route('/datasets/<id>/k_means_clustering/model_training_result', methods = ['GET'])
def km_prediction(id):
    pre_chart = km_estimate(id, request.args.get('random_state'), request.args.get('scaleMode'))
    return make_response(pre_chart)

@app.route('/datasets/<id>/k_means_clustering/calculation', methods = ['GET'])
def km_calculations(id):
    results = km_accuracy(id, request.args.get('random_state'), request.args.get('scaleMode'))
    return make_response(results)

#---------------------------------------------------------------------#

#---------------------------------------------------------------------#
# SVM Model
@app.route('/datasets/svm',methods=['POST'])
def svm_getUploadedFile():
    (data, res) = svm_fileUpload(request.files)
    return make_response(data, res)

@app.route('/datasets/<id>/svm/missing_values',methods=['GET'])
def svm_removeMissingValues(id):
    rmResult = svm_rmMissingvalues(id)
    return make_response(rmResult)

@app.route('/datasets/<id>/svm/scatter', methods = ['GET'])
def svm_scatterPlot(id):
    scatterPlot = svm_scatter_plot(id, request.args.get('scaleMode'))
    return make_response(scatterPlot)

@app.route('/datasets/<id>/svm/train_test_results', methods = ['GET'])
def svm_trainTest(id):
    trainTestPlots = svm_train_test_plot(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(trainTestPlots)

@app.route('/datasets/<id>/svm/show_solution', methods = ['GET'])
def svm_showSolution(id):
    solutionPlot = svm_solution(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(solutionPlot)

@app.route('/datasets/<id>/svm/show_confusion_matrix', methods = ['GET'])
def svm_showConfMatrix(id):
    confMatrixImg = svm_confusion_matrix(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(confMatrixImg)

@app.route('/datasets/<id>/svm/calculation', methods = ['GET'])
def svm_calculations(id):
    results = svm_evaluation(id, request.args.get('test_size'), request.args.get('random_state'))
    return make_response(results)
#---------------------------------------------------------------------#
#---------------------------------------------------------------------#
# Neural Network
@app.route('/datasets/neural_network',methods=['POST'])
def nr_getUploadedFile():
    (data, res) = nr_fileUpload(request.files)
    return make_response(data, res)

@app.route('/datasets/<id>/neural_network/missing_values',methods=['GET'])
def nr_removeMissingValues(id):
    rmResult = nr_rmMissingvalues(id)
    return make_response(rmResult)

@app.route('/datasets/<id>/neural_network/get_features', methods = ['GET'])
def nr_show_features(id):
    featuresResponse = nr_explore(id, request.args.get('scaleMode'))
    return make_response(featuresResponse)

@app.route('/datasets/<id>/neural_network/datasets_shapes', methods = ['GET'])
def nr_getShapes(id):
    shapes = nr_getShape(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(shapes)

@app.route('/datasets/<id>/neural_network/model_training_result', methods = ['GET'])
def nr_getMatrix(id):
    confMatrixImg = nr_confusion_matrix_plot(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(confMatrixImg)

@app.route('/datasets/<id>/neural_network/calculation', methods = ['GET'])
def nr_calculations(id):
    results = nr_evaluate(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(results)

@app.route('/datasets/<id>/neural_network/get_train_data_errors', methods = ['GET'])
def nr_train_data_errors_method(id):
    trainErrors = nr_train_data_errors(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(trainErrors)

@app.route('/datasets/<id>/neural_network/get_test_data_errors', methods = ['GET'])
def nr_test_data_errors_method(id):
    testErrors = nr_test_data_errors(id, request.args.get('test_size'), request.args.get('scaleMode'))
    return make_response(testErrors)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port = 5001)