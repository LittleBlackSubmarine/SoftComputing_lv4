# -*- coding: utf-8 -*-

import sys
import os
import random
import math

import networkx as nx

from sklearn.neural_network import MLPClassifier, MLPRegressor

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtWidgets import QFileDialog, QWidget, QVBoxLayout

# Make sure that we are using QT5
import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy as np

#Change current working directory to this script path
import pathlib
pathlib.Path(__file__).parent.absolute()
os.chdir(pathlib.Path(__file__).parent.absolute())

####Global NN parameters######
N_PER_LAYER = (5) #Neurons per layer
ALPHA = 0.00001 #Alpha value, regularization term
ACTIVATION_F = 1 #Neuron activation function, 1 = "logistic"
SOLVER = 0 #Solver algorithm, 1 = "lbfgs"
MAX_ITER = 1000 #Max number of training iterations
SAMPLE_SIZE = 10
##############################


####Other global variables####
#Load learn and test data
learn_data_10 = np.genfromtxt("func_10.csv", delimiter=',', dtype='float32')
learn_data_30 = np.genfromtxt("func_30.csv", delimiter=',', dtype='float32')
learn_data_60 = np.genfromtxt("func_60.csv", delimiter=',', dtype='float32')
test_data = np.genfromtxt("func_75.csv", delimiter=',', dtype='float32')
##############################


def print_results(Neurons,Layers,Alpha,Activation,Solver,MaxIterations,SampleSize):
    print("\nNeurons per layer: " + str(Neurons) + ", Layers: " + str(Layers) + ", Alpha: " + str(Alpha) +
          ", MaxIterations: " + str(MaxIterations) + ", Activation: " + str(Activation) + ", Solver: " + str(Solver) +
          ", SampleSize: " + str(SampleSize))


#Function for visualizing scikit-learn neural networks
def VisualizeNN(mpl_axes, mpl_figure, mlp, use_weighted_edges = True, show_neuron_labels = True, show_bias = False):
    #Determine number of layers. Needed for width of the graph
    no_layers = mlp.n_layers_

    g_height = 0
    layer_sizes = ()
    try:
        #Determine max number of neurons per layer. It will represent the height of the graph
        g_height = max(mlp.hidden_layer_sizes)
        #Determining number of neurons per layer
        layer_sizes = (mlp.coefs_[0].shape[0],) + mlp.hidden_layer_sizes + (mlp.n_outputs_,) #Imput layer + hidden layers + output layer
    except:
        #Determine max number of neurons per layer. It will represent the height of the graph
        g_height = mlp.hidden_layer_sizes
        #Determining number of neurons per layer
        layer_sizes = (mlp.coefs_[0].shape[0],) + (mlp.hidden_layer_sizes,) + (mlp.n_outputs_,) #Imput layer + hidden layers + output layer
    
    
    #Crating graph
    g = nx.Graph()
    
    #Adding neurons per layer
    curr_n = 0
    for l in range(no_layers):
        for n in range(layer_sizes[l]):
            g.add_node (curr_n, pos=(l * 3, (-1.0) * (g_height / 2.0 - layer_sizes[l] / 2.0 + n))) #(-1.0 * ...) in order to be inverted on Y axes, othervise (0,0) is lower left corner
            curr_n += 1
    
    #Adding edges
    curr_n = 0
    start_next_layer_idx = 0
    for l in range(no_layers-1):
        start_next_layer_idx += layer_sizes[l]
        for n in range(layer_sizes[l]):
            for n_next in range(layer_sizes[l+1]):
                g.add_edge(curr_n, start_next_layer_idx + n_next, weight=mlp.coefs_[l][n][n_next])
            curr_n += 1
    
    #Add bias nodes if requested
    if show_bias:
        #Adding bias nodes
        for l in range(1, no_layers): #Start from first hidden layer
            g.add_node ('b' + str(l), pos=(l * 3 + 1.5, 1.0)) 
        
        #Adding bias edges
        curr_n = layer_sizes[0]
        for l in range(1, no_layers): #Start from first hidden layer
            for n in range(layer_sizes[l]):
                g.add_edge(curr_n, 'b' + str(l), weight=mlp.intercepts_[l-1][n])
                curr_n += 1
                
    #Drawing
    pos=nx.get_node_attributes(g,'pos')
    if use_weighted_edges:
        weights = nx.get_edge_attributes(g, "weight")
        weights_v = list(weights.values())
        nx.draw(g, pos, ax = mpl_axes, edge_color=weights_v, edge_cmap=plt.cm.Spectral, with_labels=show_neuron_labels, font_weight='bold')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral, norm=plt.Normalize(vmin = min(weights_v), vmax=max(weights_v)))
        sm._A = []
        mpl_figure.colorbar(sm)
        
    else:
        nx.draw(g, pos, ax = mpl_axes, with_labels=show_neuron_labels, font_weight='bold')      

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        FigureCanvas.__init__(self, self.fig)
        
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    def Reset(self):
        self.fig.clf()
        self.axes = self.fig.add_subplot(111)
        

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1010, 915)
        self.setWindowTitle("NN - Approximation")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        self.lineChartWidget = QtWidgets.QWidget(self.centralwidget)
        self.lineChartWidget.setGeometry(QtCore.QRect(10, 10, 600, 450))
        
        self.lineChart = MplCanvas(self.lineChartWidget, 600, 450)
        self.lineChart.setObjectName("lineChart")
        
        self.lineChartWidgetlayout = QtWidgets.QVBoxLayout()
        self.lineChartWidgetlayout.addWidget(self.lineChart)
        self.lineChartWidgetlayout.addWidget(NavigationToolbar(self.lineChart, self.lineChartWidget))
        self.lineChartWidget.setLayout(self.lineChartWidgetlayout)
        
        self.NNChartWidget = QtWidgets.QWidget(self.centralwidget)
        self.NNChartWidget.setGeometry(QtCore.QRect(10, 470, 1000, 450))
        
        self.NNChart = MplCanvas(self.NNChartWidget, 1000, 450)
        self.NNChart.setObjectName("NNChart")
        
        self.NNChartWidgetlayout = QtWidgets.QVBoxLayout()
        self.NNChartWidgetlayout.addWidget(self.NNChart)
        self.NNChartWidgetlayout.addWidget(NavigationToolbar(self.NNChart, self.NNChartWidget))
        self.NNChartWidget.setLayout(self.NNChartWidgetlayout)
        
        self.nnParams = QtWidgets.QGroupBox(self.centralwidget)
        self.nnParams.setGeometry(QtCore.QRect(630, 10, 200, 165))
        self.nnParams.setObjectName("nnParams")
        self.nnParams.setTitle("NN parameters")
        
        self.label1 = QtWidgets.QLabel(self.nnParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 91, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Neurons per layer:")
        
        self.label2 = QtWidgets.QLabel(self.nnParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("Alpha:")
        
        self.label3 = QtWidgets.QLabel(self.nnParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 91, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("Activation function:")
        
        self.label4 = QtWidgets.QLabel(self.nnParams)
        self.label4.setGeometry(QtCore.QRect(10, 110, 91, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("Solver:")
        
        self.label5 = QtWidgets.QLabel(self.nnParams)
        self.label5.setGeometry(QtCore.QRect(10, 140, 91, 16))
        self.label5.setObjectName("label5")
        self.label5.setText("Max iterations:")
        
        self.tbxNeuroPerLayer = QtWidgets.QLineEdit(self.nnParams)
        self.tbxNeuroPerLayer.setGeometry(QtCore.QRect(120, 20, 71, 20))
        self.tbxNeuroPerLayer.setObjectName("tbxNeuroPerLayer")
        
        self.tbxAlpha = QtWidgets.QLineEdit(self.nnParams)
        self.tbxAlpha.setGeometry(QtCore.QRect(120, 50, 71, 20))
        self.tbxAlpha.setObjectName("tbxAlpha")
        
        self.comboActivation = QtWidgets.QComboBox(self.nnParams)
        self.comboActivation.setGeometry(QtCore.QRect(120, 80, 71, 20))
        self.comboActivation.setObjectName("comboActivation")
        self.comboActivation.addItem("identity")
        self.comboActivation.addItem("logistic")
        self.comboActivation.addItem("tanh")
        self.comboActivation.addItem("relu")
        
        self.comboSolver = QtWidgets.QComboBox(self.nnParams)
        self.comboSolver.setGeometry(QtCore.QRect(120, 110, 71, 20))
        self.comboSolver.setObjectName("comboSolver")
        self.comboSolver.addItem("lbfgs")
        self.comboSolver.addItem("sgd")
        self.comboSolver.addItem("adam")
        
        self.tbxMaxIter = QtWidgets.QLineEdit(self.nnParams)
        self.tbxMaxIter.setGeometry(QtCore.QRect(120, 140, 71, 20))
        self.tbxMaxIter.setObjectName("tbxMaxIter")
        
        self.label6 = QtWidgets.QLabel(self.centralwidget)
        self.label6.setGeometry(QtCore.QRect(620, 230, 91, 16))
        self.label6.setObjectName("label6")
        self.label6.setText("Training sample:")
        
        self.comboSample = QtWidgets.QComboBox(self.centralwidget)
        self.comboSample.setGeometry(QtCore.QRect(720, 230, 100, 20))
        self.comboSample.setObjectName("comboSample")
        self.comboSample.addItem("func_10.csv")
        self.comboSample.addItem("func_30.csv")
        self.comboSample.addItem("func_60.csv")
        
        self.btnStart = QtWidgets.QPushButton(self.centralwidget)
        self.btnStart.setGeometry(QtCore.QRect(650, 270, 150, 50))
        self.btnStart.setObjectName("btnStart")
        self.btnStart.setText("Start")

        self.cbxShowNeuronLabels = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowNeuronLabels.setGeometry(QtCore.QRect(300, 460, 100, 17))
        self.cbxShowNeuronLabels.setObjectName("cbxShowNeuronLabels")
        self.cbxShowNeuronLabels.setText("Show neuron labels")
        
        self.cbxShowWeights = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowWeights.setGeometry(QtCore.QRect(500, 460, 100, 17))
        self.cbxShowWeights.setObjectName("cbxShowWeights")
        self.cbxShowWeights.setText("Show weights")

        self.cbxShowBias = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxShowBias.setGeometry(QtCore.QRect(700, 460, 100, 17))
        self.cbxShowBias.setObjectName("cbxShowBias")
        self.cbxShowBias.setText("Show bias")
        
        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)
        
        #Connect events
        self.btnStart.clicked.connect(self.btnStart_Click)

            #Set default GA variables
        self.tbxNeuroPerLayer.insert(str(N_PER_LAYER))
        self.tbxAlpha.insert(str(ALPHA))
        self.comboActivation.setCurrentIndex(ACTIVATION_F)
        self.comboSolver.setCurrentIndex(SOLVER)
        self.tbxMaxIter.insert(str(MAX_ITER))
        
    def btnStart_Click(self):
        #Set global variables from information on UI
        global N_PER_LAYER
        global ALPHA
        global ACTIVATION_F
        global SOLVER
        global MAX_ITER




        N_PER_LAYER = eval(self.tbxNeuroPerLayer.text())
        ALPHA = float(self.tbxAlpha.text())
        ACTIVATION_F = self.comboActivation.currentText()  # --Altered by Denis Lazor // ...currentIndex() --> ...currentText()
        SOLVER = self.comboSolver.currentText()            # -||-
        MAX_ITER = int(self.tbxMaxIter.text())
        SAMPLE_SIZE = int(self.comboSample.currentText()[5:7])  # Getting the size of sample used for learning -- Added by Denis Lazor
        LAYERS = self.tbxNeuroPerLayer.text().count(',') + 1  # Counting the number of layers --  Added by Denis Lazor

        #Create neural network --Added by Denis Lazor
        mlp = MLPRegressor(hidden_layer_sizes=N_PER_LAYER, activation=ACTIVATION_F, solver=SOLVER, alpha=ALPHA, max_iter=MAX_ITER)
        
        #Learn neural network
        if self.comboSample.currentText() == "func_10.csv":
            mlp.fit(learn_data_10[:,0].reshape(learn_data_10.shape[0],1), learn_data_10[:,1])
        elif self.comboSample.currentText() == "func_30.csv":
            mlp.fit(learn_data_30[:,0].reshape(learn_data_30.shape[0],1), learn_data_30[:,1])
        elif self.comboSample.currentText() == "func_60.csv":
            mlp.fit(learn_data_60[:,0].reshape(learn_data_60.shape[0],1), learn_data_60[:,1])
        
        output = mlp.predict(test_data[:,0].reshape(test_data.shape[0],1))
        
        #Draw function
        self.lineChart.Reset()
        self.lineChart.axes.plot(test_data[:,1])
        self.lineChart.axes.plot(output)
        self.lineChart.draw()
        #Draw NN chart
        self.NNChart.Reset()
        VisualizeNN(self.NNChart.axes, self.NNChart.figure, mlp, use_weighted_edges=self.cbxShowWeights.isChecked(), show_neuron_labels=self.cbxShowNeuronLabels.isChecked(), show_bias=self.cbxShowBias.isChecked())
        plt.close() #Close anomalous popping figure?!?
        self.NNChart.draw()

        print_results(N_PER_LAYER,LAYERS,ALPHA,ACTIVATION_F,SOLVER,MAX_ITER,SAMPLE_SIZE)
        #Calculate MSE
        mse = np.mean((test_data[:,1] - output)**2)
        print("MSE = ", mse)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.show()
    sys.exit(app.exec_())