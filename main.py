import scipy.linalg
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from lib import NonLinearFunction1, NonLinearFunction2 #non linear functions, from assignment and custom one
from lib import VectorNorm, pinv, SolveNormalEquation, GenerateRandomPoints #utility functions
from lib import NeuralNetworkPass, CalculateNeuralNetworkGradient, CalculateLoss, CalculateLossJacobian #neural network and loss function utility functions
from lib import EvaluatePerformance #performance evaluation utility for neural network
from lib import TrainNetwork #core training function of neural network

#main script

NonLinearFunction = NonLinearFunction1 #change non linear function here

trainIterations = 300 #change this to change runtime; but this will affect performance




#varying loss reg. lambda of network, then observing performance
#plot loss during training



testingPoints = GenerateRandomPoints(100, 1) #generate 100 random points, where random vector elements stay within -1 and 1
trainingPoints = GenerateRandomPoints(200, 1)

lossRegLambda = 0.00001
initialTrustLambda = 1
initalWeights = np.random.uniform(-10, 10, 13) #randomly initialize 16 weights uniformly in between -10 and 10
'Plot Grad Norm vs iteratons'

trainedWeights, loss, gradNorm = TrainNetwork(trainingPoints, trainIterations, initalWeights, lossRegLambda, initialTrustLambda, NonLinearFunction) #train network with reg lambda
performance = EvaluatePerformance(trainingPoints, trainedWeights, NonLinearFunction) #evaluate performance

plt.plot(np.arange(trainIterations), gradNorm, label = "$\lambda$: " + str(lossRegLambda)) #plot loss vs iterations
plt.title("The norm of gradients versus iteration",fontsize=20)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('Grad Norm',fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.grid()
plt.legend(prop={'size':15})
plt.show()

'Plot Grad Norm vs iteratons'
lossRegLambda = 0.00001
trainedWeights, loss, gradNorm = TrainNetwork(trainingPoints, trainIterations, initalWeights, lossRegLambda, initialTrustLambda, NonLinearFunction) #train network with reg lambda
performance = EvaluatePerformance(trainingPoints, trainedWeights, NonLinearFunction) #evaluate performance
plt.plot(np.arange(trainIterations), loss, label = "$\lambda$: " + str(lossRegLambda)) #plot loss vs iterations
plt.title("Training loss versus iteration",fontsize=20)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('The value of f',fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.grid()
plt.legend(prop={'size':15})
plt.show()

'Different Lamda & RMSE'
lossRegLambdas = [0.00001, 0.0001, 0.01, 0.1, 1, 10]
for lossRegLambda in lossRegLambdas:
    trainedWeights, loss, gradNorm = TrainNetwork(trainingPoints, trainIterations, initalWeights, lossRegLambda, initialTrustLambda, NonLinearFunction) #train network with reg lambda
    performance = EvaluatePerformance(trainingPoints, trainedWeights, NonLinearFunction) #evaluate performance
    plt.plot(np.arange(trainIterations), loss, label = "$\lambda$: " + str(lossRegLambda) + ", Final RMSE: " + str(('%.4f' %performance))) #plot loss vs iterations

plt.title("Training loss vs iteration,\n varying loss regularization lambda", fontsize=20)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('The value of f',fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylim(0,100)
plt.grid()
plt.legend(prop={'size':10})
plt.show()






#vary bounds of weights that are randomly initialized, then test performance
#plot loss during training
#everything else constant


#wider random range should prevent the loss from converging to a local minima fast; the weights are taking different "routes" to a minima


testingPoints = GenerateRandomPoints(100, 1)
trainingPoints = GenerateRandomPoints(500, 1)


bounds = [0.1, 10, 100]
lossRegLambda = 1e-5
initialTrustLambda = 1

for bound in bounds:
#    for i in range(3): #train three times with the same bound; idea is to see if the loss converges to the same value and gets "trapped" in a minima
    initalWeights = np.random.uniform(-bound, bound, 13) #randomly initialize 16 weights uniformly in between the specified bound
    trainedWeights, loss, gradNorm = TrainNetwork(trainingPoints, trainIterations, initalWeights, lossRegLambda, initialTrustLambda, NonLinearFunction)
    performance = EvaluatePerformance(trainingPoints, trainedWeights, NonLinearFunction)
    plt.plot(np.arange(trainIterations), loss, label = "initial weights random bound: " + str(bound))
plt.title("Training loss vs iteration, \n varying random bounds of initialized weights ",fontsize=20)
plt.xlabel('Iteration',fontsize=18)
plt.ylabel('The value of f',fontsize=18)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.ylim(0,300)
plt.grid()
plt.legend(prop={'size':10})
plt.show()












#vary number of test points for performance
#everything else constant



numberOfTestPoints = [50, 150, 300, 500, 1000]
performanceResults = []

trainingPoints = GenerateRandomPoints(500, 1)

initalWeights = np.random.uniform(-10, 10, 16) #randomly initialize 16 weights uniformly in between -10 and 10
lossRegLambda = 0.1
initialTrustLambda = 1

trainedWeights, loss = TrainNetwork(trainingPoints, trainIterations, initalWeights, lossRegLambda, initialTrustLambda, NonLinearFunction)

for testCount in numberOfTestPoints:
    testingPoints = GenerateRandomPoints(testCount, 1)
    performance = EvaluatePerformance(testingPoints, trainedWeights, NonLinearFunction)
    performanceResults.append(performance)


plt.plot(numberOfTestPoints, performanceResults)
plt.title("Test performance vs number of test points, loss reg. lambda of " + str(lossRegLambda))
plt.xlabel('Number of test points')
plt.ylabel('Test performance')
plt.show()

















