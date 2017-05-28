# numpy is fundamental package for scientific computing with Python
# sklearn is machine learning package for python, and sklearn.datasets contains sample data set
import numpy as np
from sklearn import datasets

# matplotlib and mpl_toolkits for math plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from collections import Counter


# Create a learnset
# iris label has [0, 1, 2] three elements
# iris data has four elements
# so label's element number is not equaled to iris's element number
iris = datasets.load_iris()
iris_data = iris.data
iris_labels = iris.target


# Use permutation from np.random to split the data randomly
np.random.seed(42)
indices = np.random.permutation(len(iris_data))
n_training_samples = 12
learnset_data = iris_data[indices[ : -n_training_samples]]
learnset_labels = iris_labels[indices[ : -n_training_samples]]
testset_data = iris_data[indices[-n_training_samples :]]
testset_labels = iris_labels[indices[-n_training_samples: ]]
# print sorted(indices)
# print (learnset_data[ :4], learnset_labels[ :4])
# print (testset_data[ :4], testset_labels[ :4])


# The following code is only necessary to visualize the data of our learnset.
# Our data consists of four values per iris item, so we will reduce the data to three values by summing up the third and fourth value.
# This way, we are capable of depicting the data in 3-dimensional space:

# Following line is only necessary, if you use ipython notebook!
colours = ('r', 'b')
X = []
for iclass in range(3):
    X.append([[], [], []])
    for i in range(len(learnset_data)):
        if learnset_labels[i] == iclass:
            X[iclass][0].append(learnset_data[i][0])
            X[iclass][1].append(learnset_data[i][1])
            X[iclass][2].append(sum(learnset_data[i][2:]))

colours = ('r', 'g', 'y')
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
for iclass in range(3):
    ax.scatter(X[iclass][0], X[iclass][1], X[iclass][2], c=colours[iclass])
plt.show()


# Function to calculate Euclidean distance between two instances
def distance(instance1, instance2):
    # just in case, if the instance are lists or tuples
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)


# The function 'get_neighbors' returns a list with 'k' neighbors, which are closest to the instance 'test_instance'
def get_neighbors(training_set, labels, test_instance, k): # pay attention to k
    """
    get_neighbors calculates a list of the k nearest neighbors of an instance 'test_instance'.
    The list neighbors contains 3-tuples with (index, dist, label) where :
        index   is the index from the training_set,
        dist    is the distance between the test_instance
        distance    is a reference to a function used to calculate the distances
    """
    distances = []
    for index in range(len(training_set)):
        dist = distance(test_instance, training_set[index])
        distances.append((training_set[index], dist, labels[index]))
    distances.sort(key=lambda x: x[1])
    neighbors = distances[:k]
    return neighbors

# use for test get_neighbors() function
for i in range(5):
    neighbors = get_neighbors(learnset_data,
                             learnset_labels,
                             testset_data[i],
                             3)
    print (i, testset_data[i], testset_labels[i], neighbors)
    print ('')

# ========================================== simple vote ==================================================
# the vote function, use the class 'Counter' from collections to count the quantity of the classes inside of an instance list.
# This instance list will be the neighbors of course.
# The function 'vote' returns the most common class:
def vote(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1 # label is the type of classes, label:number (key:value)
    return class_counter.most_common(1)[0][0] # get the most neighbor's label


def vote_prob(neighbors):
    class_counter = Counter()
    for neighbor in neighbors:
        class_counter[neighbor[2]] += 1
    labels, votes = zip(*class_counter.most_common()) # * means unzip tuple into two separated elements
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    return winner, votes4winner/sum(votes)


# TEST VOTE FUNCTION, We can see that the predictions correspond to the labelled results, except in case of the item with the index 8.
for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,
                             learnset_labels,
                             testset_data[i],
                             5)
    print ("index: ", i,
           ", result of vote: ", vote(neighbors),
           ", label: ", testset_labels[i],
           ", data: ", testset_data[i])


# ======================================= The Weighted Nearest Neighbour Classifier (1) =================================================
def vote_harmonic_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        class_counter[neighbors[index][2]] += 1/(index+1)
    labels, votes = zip(*class_counter.most_common()) #label is the first element(winner label), votes is the second(weight)
    # print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1] # the calculated weight
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
            class_counter[key] /= total
            return winner, class_counter.most_common()
    else:
        return winner, votes4winner/sum(votes)

# TEST VOTE FUNCTION
for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,
                              learnset_labels,
                              testset_data[i],
                              6)
    print("index: ", i,
          ", result of vote: ",
          vote_harmonic_weights(neighbors,
                                all_results=True))


# ====================================== The Weighted Nearest Neighbour Classifier (2)=======================================================
# The previous approach took only the ranking of the neighbors according to their distance in account.
# We can improve the voting by using the actual distance.

def vote_distance_weights(neighbors, all_results=True):
    class_counter = Counter()
    number_of_neighbors = len(neighbors)
    for index in range(number_of_neighbors):
        dist = neighbors[index][1]
        label = neighbors[index][2]
        class_counter[label] += 1 / (dist**2 + 1) # add weight using the actual distance
    labels, votes = zip(*class_counter.most_common())
    #print(labels, votes)
    winner = class_counter.most_common(1)[0][0]
    votes4winner = class_counter.most_common(1)[0][1]
    if all_results:
        total = sum(class_counter.values(), 0.0)
        for key in class_counter:
             class_counter[key] /= total
        return winner, class_counter.most_common()
    else:
        return winner, votes4winner / sum(votes)

# TEST VOTE FUNCTION
for i in range(n_training_samples):
    neighbors = get_neighbors(learnset_data,
                              learnset_labels,
                              testset_data[i],
                              6)
    print("index: ", i, ", result of vote: ", vote_distance_weights(neighbors, all_results=True))
