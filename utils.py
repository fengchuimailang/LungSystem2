import os
import numpy as np
import tensorflow as tf
import h5py
import math
import cv2
import shutil


# def load_dataset():
#     train_dataset = h5py.File('datasets/train_signs.h5', "r")
#     train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
#     train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels
#
#     test_dataset = h5py.File('datasets/test_signs.h5', "r")
#     test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
#     test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
#
#     classes = np.array(test_dataset["list_classes"][:])  # the list of classes
#
#     train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
#     test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
#
#     return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_dataset_ct():
    train_set_x_orig = []
    train_set_y_orig = []
    test_set_x_orig = []
    test_set_y_orig = []
    # train_dir = "/Users/liubo/Documents/graduation_project/mnist_train_512"
    # test_dir = "/Users/liubo/Documents/graduation_project/mnist_test_512"
    train_dir = "/home/liubo/data/MNIST/mnist_train_512"
    test_dir = "/home/liubo/data/MNIST/mnist_test_512"
    for label in range(5):
        for file_path in os.listdir(train_dir + "/" + str(label))[:100]:
            file_path = train_dir + "/" + str(label) + "/" + file_path
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
            train_set_x_orig.append(img)
            train_set_y_orig.append(label)
    for label in range(5):
        for file_path in os.listdir(test_dir + "/" + str(label))[:20]:
            file_path = test_dir + "/" + str(label) + "/" + file_path
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
            test_set_x_orig.append(img)
            test_set_y_orig.append(label)
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # shuffle trainset
    m = train_set_x_orig.shape[0]
    permutation = list(np.random.permutation(m))
    train_set_x_orig = train_set_x_orig[permutation, :, :, :]
    train_set_y_orig = train_set_y_orig[permutation]

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def load_dataset_pet():
    train_set_x_orig = []
    train_set_y_orig = []
    test_set_x_orig = []
    test_set_y_orig = []
    # train_dir = "/Users/liubo/Documents/graduation_project/mnist_train_512"
    # test_dir = "/Users/liubo/Documents/graduation_project/mnist_test_512"
    train_dir = "/home/liubo/data/MNIST/mnist_train_128"
    test_dir = "/home/liubo/data/MNIST/mnist_test_128"
    for label in range(5):
        for file_path in os.listdir(train_dir + "/" + str(label))[:100]:
            file_path = train_dir + "/" + str(label) + "/" + file_path
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
            train_set_x_orig.append(img)
            train_set_y_orig.append(label)
    for label in range(5):
        for file_path in os.listdir(test_dir + "/" + str(label))[:20]:
            file_path = test_dir + "/" + str(label) + "/" + file_path
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            img = img[:, :, np.newaxis]
            test_set_x_orig.append(img)
            test_set_y_orig.append(label)
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # shuffle trainset
    m = train_set_x_orig.shape[0]
    permutation = list(np.random.permutation(m))
    train_set_x_orig = train_set_x_orig[permutation, :, :, :]
    train_set_y_orig = train_set_y_orig[permutation]

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def load_dataset_ct_pet_1():
    train_set_x_ct_orig = []
    train_set_x_pet_orig = []
    train_set_y_orig = []
    test_set_x_ct_orig = []
    test_set_x_pet_orig = []
    test_set_y_orig = []
    # train_dir = "/Users/liubo/Documents/graduation_project/mnist_train_512"
    # test_dir = "/Users/liubo/Documents/graduation_project/mnist_test_512"
    train_ct_dir = "/home/liubo/data/MNIST/mnist_train_512"
    test_ct_dir = "/home/liubo/data/MNIST/mnist_test_512"
    train_pet_dir = "/home/liubo/data/MNIST/mnist_train_128"
    test_pet_dir = "/home/liubo/data/MNIST/mnist_test_128"
    for label in range(5):
        for file_path in os.listdir(train_ct_dir + "/" + str(label))[:100]:
            ct_file_path = train_ct_dir + "/" + str(label) + "/" + file_path
            img_ct = cv2.imread(ct_file_path, cv2.IMREAD_GRAYSCALE)
            # img_ct = img_ct[:, :, np.newaxis]
            train_set_x_ct_orig.append(img_ct)

            pet_file_path = train_pet_dir + "/" + str(label) + "/" + file_path
            img_pet = cv2.imread(pet_file_path, cv2.IMREAD_GRAYSCALE)
            # 放缩到与ct一样大小
            img_pet = cv2.resize(img_pet, (512, 512))
            # img_pet = img_pet[:, :, np.newaxis]
            train_set_x_pet_orig.append(img_pet)

            train_set_y_orig.append(label)
    for label in range(5):
        for file_path in os.listdir(test_ct_dir + "/" + str(label))[:20]:
            ct_file_path = test_ct_dir + "/" + str(label) + "/" + file_path
            img_ct = cv2.imread(ct_file_path, cv2.IMREAD_GRAYSCALE)
            # img_ct = img_ct[:, :, np.newaxis]
            test_set_x_ct_orig.append(img_ct)

            pet_file_path = test_pet_dir + "/" + str(label) + "/" + file_path
            img_pet = cv2.imread(pet_file_path, cv2.IMREAD_GRAYSCALE)
            # 放缩到与ct一样大小
            img_pet = cv2.resize(img_pet, (512, 512))
            # img_pet = img_pet[:, :, np.newaxis]
            test_set_x_pet_orig.append(img_pet)

            test_set_y_orig.append(label)
    train_set_x_ct_orig = np.array(train_set_x_ct_orig)
    train_set_x_pet_orig = np.array(train_set_x_pet_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_ct_orig = np.array(test_set_x_ct_orig)
    test_set_x_pet_orig = np.array(test_set_x_pet_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # train test ct和pet进行合并
    train_set_x_orig = np.stack((train_set_x_ct_orig, train_set_x_pet_orig), axis=3)
    test_set_x_orig = np.stack((test_set_x_ct_orig, test_set_x_pet_orig), axis=3)

    # shuffle trainset
    m = train_set_x_orig.shape[0]
    permutation = list(np.random.permutation(m))
    train_set_x_orig = train_set_x_orig[permutation, :, :, :]
    train_set_y_orig = train_set_y_orig[permutation]

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def load_dataset_ct_pet_2():
    train_set_x_ct_orig = []
    train_set_x_pet_orig = []
    train_set_y_orig = []
    test_set_x_ct_orig = []
    test_set_x_pet_orig = []
    test_set_y_orig = []
    # train_dir = "/Users/liubo/Documents/graduation_project/mnist_train_512"
    # test_dir = "/Users/liubo/Documents/graduation_project/mnist_test_512"
    train_ct_dir = "/home/liubo/data/MNIST/mnist_train_512"
    test_ct_dir = "/home/liubo/data/MNIST/mnist_test_512"
    train_pet_dir = "/home/liubo/data/MNIST/mnist_train_128"
    test_pet_dir = "/home/liubo/data/MNIST/mnist_test_128"
    for label in range(5):
        for file_path in os.listdir(train_ct_dir + "/" + str(label))[:100]:
            ct_file_path = train_ct_dir + "/" + str(label) + "/" + file_path
            img_ct = cv2.imread(ct_file_path, cv2.IMREAD_GRAYSCALE)
            img_ct = img_ct[:, :, np.newaxis]
            train_set_x_ct_orig.append(img_ct)

            pet_file_path = train_pet_dir + "/" + str(label) + "/" + file_path
            img_pet = cv2.imread(pet_file_path, cv2.IMREAD_GRAYSCALE)
            img_pet = img_pet[:, :, np.newaxis]
            train_set_x_pet_orig.append(img_pet)

            train_set_y_orig.append(label)
    for label in range(5):
        for file_path in os.listdir(test_ct_dir + "/" + str(label))[:20]:
            ct_file_path = test_ct_dir + "/" + str(label) + "/" + file_path
            img_ct = cv2.imread(ct_file_path, cv2.IMREAD_GRAYSCALE)
            img_ct = img_ct[:, :, np.newaxis]
            test_set_x_ct_orig.append(img_ct)

            pet_file_path = test_pet_dir + "/" + str(label) + "/" + file_path
            img_pet = cv2.imread(pet_file_path, cv2.IMREAD_GRAYSCALE)
            img_pet = img_pet[:, :, np.newaxis]
            test_set_x_pet_orig.append(img_pet)

            test_set_y_orig.append(label)

    train_set_x_ct_orig = np.array(train_set_x_ct_orig)
    train_set_x_pet_orig = np.array(train_set_x_pet_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_ct_orig = np.array(test_set_x_ct_orig)
    test_set_x_pet_orig = np.array(test_set_x_pet_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # shuffle trainset
    m = train_set_x_ct_orig.shape[0]
    permutation = list(np.random.permutation(m))
    train_set_x_ct_orig = train_set_x_ct_orig[permutation, :, :, :]
    train_set_x_pet_orig = train_set_x_pet_orig[permutation, :, :, :]
    train_set_y_orig = train_set_y_orig[permutation]

    return train_set_x_ct_orig, train_set_x_pet_orig, train_set_y_orig, test_set_x_ct_orig, test_set_x_pet_orig, test_set_y_orig


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[0]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :, :, :]
    shuffled_Y = Y[permutation, :]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :, :, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size: k * mini_batch_size + mini_batch_size, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m, :, :, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m, :]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3

    return Z3


def predict(X, parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)

    sess = tf.Session()
    prediction = sess.run(p, feed_dict={x: X})

    return prediction

