import os
import numpy as np
import tensorflow as tf
import h5py
import math
import cv2
import shutil
import csv


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


def augu_one_input(x, y):
    x_5class = [[], [], [], [], []]
    y_5class = [[], [], [], [], []]
    for i in range(len(x)):
        label = y[i]
        x_5class[label].append(x[i])
        y_5class[label].append(y[i])
    max_count = max([len(l) for l in x_5class])
    augu_x = []
    augu_y = []
    for label in range(5):
        n = int(max_count / len(x_5class[label]))
        augu_x.extend(x_5class[label] * n)
        augu_y.extend(y_5class[label] * n)
    return np.array(augu_x), np.array(augu_y)


def augu_two_input(x1, x2, y):
    x1_5class = [[], [], [], [], []]
    x2_5class = [[], [], [], [], []]
    y_5class = [[], [], [], [], []]
    for i in range(len(x1)):
        label = y[i]
        x1_5class[label].append(x1[i])
        x2_5class[label].append(x2[i])
        y_5class[label].append(y[i])
    max_count = max([len(l) for l in x1_5class])
    augu_x1 = []
    augu_x2 = []
    augu_y = []
    for label in range(5):
        n = int(max_count / len(x1_5class[label]))
        augu_x1.extend(x1_5class[label] * n)
        augu_x2.extend(x2_5class[label] * n)
        augu_y.extend(y_5class[label] * n)
    return np.array(augu_x1), np.array(augu_x2), np.array(augu_y)


def augu_three_input(x1, x2, x3, y):
    x1_5class = [[], [], [], [], []]
    x2_5class = [[], [], [], [], []]
    x3_5class = [[], [], [], [], []]
    y_5class = [[], [], [], [], []]
    for i in range(len(x1)):
        label = y[i]
        x1_5class[label].append(x1[i])
        x2_5class[label].append(x2[i])
        x3_5class[label].append(x3[i])
        y_5class[label].append(y[i])
    max_count = max([len(l) for l in x1_5class])
    augu_x1 = []
    augu_x2 = []
    augu_x3 = []
    augu_y = []
    for label in range(5):
        n = int(max_count / len(x1_5class[label]))
        augu_x1.extend(x1_5class[label] * n)
        augu_x2.extend(x2_5class[label] * n)
        augu_x3.extend(x3_5class[label] * n)
        augu_y.extend(y_5class[label] * n)
    return np.array(augu_x1), np.array(augu_x2), np.array(augu_x3), np.array(augu_y)


def load_dataset_ct():
    train_set_x_orig = []
    train_set_y_orig = []
    test_set_x_orig = []
    test_set_y_orig = []
    train_dirs = ["/home/liubo/data/graduate/CTSlice/fold0",
                  "/home/liubo/data/graduate/CTSlice/fold1",
                  "/home/liubo/data/graduate/CTSlice/fold2",
                  "/home/liubo/data/graduate/CTSlice/fold3"
                  ]
    test_dir = "/home/liubo/data/graduate/CTSlice/fold4"
    for train_dir in train_dirs:
        for file_path in os.listdir(train_dir):
            file_path = train_dir + "/" + file_path
            img = np.load(file_path)
            img = img[:, :, np.newaxis]
            label = int(file_path.split(".")[0].split("_")[-1])
            train_set_x_orig.append(img)
            train_set_y_orig.append(label)

    for file_path in os.listdir(test_dir):
        file_path = test_dir + "/" + file_path
        img = np.load(file_path)
        img = img[:, :, np.newaxis]
        label = int(file_path.split(".")[0].split("_")[-1])
        test_set_x_orig.append(img)
        test_set_y_orig.append(label)
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # 平衡 不同种类的样本
    train_set_x_orig, train_set_y_orig = augu_one_input(train_set_x_orig, train_set_y_orig)

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
    train_dirs = ["/home/liubo/data/graduate/PETSlice/fold0",
                  "/home/liubo/data/graduate/PETSlice/fold1",
                  "/home/liubo/data/graduate/PETSlice/fold2",
                  "/home/liubo/data/graduate/PETSlice/fold3"
                  ]
    test_dir = "/home/liubo/data/graduate/PETSlice/fold4"
    for train_dir in train_dirs:
        for file_path in os.listdir(train_dir):
            file_path = train_dir + "/" + file_path
            img = np.load(file_path)
            img = cv2.resize(img, (128, 128))
            img = img[:, :, np.newaxis]
            label = int(file_path.split(".")[0].split("_")[-1])
            train_set_x_orig.append(img)
            train_set_y_orig.append(label)

    for file_path in os.listdir(test_dir):
        file_path = test_dir + "/" + file_path
        img = np.load(file_path)
        img = cv2.resize(img, (128, 128))
        img = img[:, :, np.newaxis]
        label = int(file_path.split(".")[0].split("_")[-1])
        test_set_x_orig.append(img)
        test_set_y_orig.append(label)
    train_set_x_orig = np.array(train_set_x_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_orig = np.array(test_set_x_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # 平衡 不同种类的样本
    train_set_x_orig, train_set_y_orig = augu_one_input(train_set_x_orig, train_set_y_orig)

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

    train_ct_dirs = ["/home/liubo/data/graduate/CTSlice/fold0",
                     "/home/liubo/data/graduate/CTSlice/fold1",
                     "/home/liubo/data/graduate/CTSlice/fold2",
                     "/home/liubo/data/graduate/CTSlice/fold3"
                     ]
    test_ct_dir = "/home/liubo/data/graduate/CTSlice/fold4"

    train_pet_dirs = ["/home/liubo/data/graduate/PETSlice/fold0",
                      "/home/liubo/data/graduate/PETSlice/fold1",
                      "/home/liubo/data/graduate/PETSlice/fold2",
                      "/home/liubo/data/graduate/PETSlice/fold3"
                      ]
    test_pet_dir = "/home/liubo/data/graduate/PETSlice/fold4"

    for i in range(len(train_ct_dirs)):
        train_ct_dir = train_ct_dirs[i]
        train_pet_dir = train_pet_dirs[i]
        for ct_file_name in os.listdir(train_ct_dir):
            ct_file_path = train_ct_dir + "/" + ct_file_name
            img_ct = np.load(ct_file_path)
            # img_ct = img_ct[:, :, np.newaxis]
            train_set_x_ct_orig.append(img_ct)

            pet_file_name = "PETSlice".join(ct_file_name.split("CTSlice"))
            pet_file_path = train_pet_dir + "/" + pet_file_name
            img_pet = np.load(pet_file_path)
            img_pet = cv2.resize(img_pet, (512, 512))
            # img_pet = img_pet[:, :, np.newaxis]
            train_set_x_pet_orig.append(img_pet)

            label = int(ct_file_name.split(".")[0].split("_")[-1])
            train_set_y_orig.append(label)

    for ct_file_name in os.listdir(test_ct_dir):
        ct_file_path = test_ct_dir + "/" + ct_file_name
        img_ct = np.load(ct_file_path)
        # img_ct = img_ct[:, :, np.newaxis]
        test_set_x_ct_orig.append(img_ct)

        pet_file_name = "PETSlice".join(ct_file_name.split("CTSlice"))
        pet_file_path = test_pet_dir + "/" + pet_file_name
        img_pet = np.load(pet_file_path)
        # 放缩到与ct一样大小
        img_pet = cv2.resize(img_pet, (512, 512))
        # img_pet = img_pet[:, :, np.newaxis]
        test_set_x_pet_orig.append(img_pet)

        label = int(ct_file_name.split(".")[0].split("_")[-1])
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

    # 平衡 不同种类的样本
    train_set_x_orig, train_set_y_orig = augu_one_input(train_set_x_orig, train_set_y_orig)

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

    train_ct_dirs = ["/home/liubo/data/graduate/CTSlice/fold0",
                     "/home/liubo/data/graduate/CTSlice/fold1",
                     "/home/liubo/data/graduate/CTSlice/fold2",
                     "/home/liubo/data/graduate/CTSlice/fold3"
                     ]
    test_ct_dir = "/home/liubo/data/graduate/CTSlice/fold4"

    train_pet_dirs = ["/home/liubo/data/graduate/PETSlice/fold0",
                      "/home/liubo/data/graduate/PETSlice/fold1",
                      "/home/liubo/data/graduate/PETSlice/fold2",
                      "/home/liubo/data/graduate/PETSlice/fold3"
                      ]
    test_pet_dir = "/home/liubo/data/graduate/PETSlice/fold4"

    for i in range(len(train_ct_dirs)):
        train_ct_dir = train_ct_dirs[i]
        train_pet_dir = train_pet_dirs[i]
        for ct_file_name in os.listdir(train_ct_dir):
            ct_file_path = train_ct_dir + "/" + ct_file_name
            img_ct = np.load(ct_file_path)
            img_ct = img_ct[:, :, np.newaxis]
            train_set_x_ct_orig.append(img_ct)

            pet_file_name = "PETSlice".join(ct_file_name.split("CTSlice"))
            pet_file_path = train_pet_dir + "/" + pet_file_name
            img_pet = np.load(pet_file_path)
            img_pet = cv2.resize(img_pet, (128, 128))
            img_pet = img_pet[:, :, np.newaxis]
            train_set_x_pet_orig.append(img_pet)
            label = int(ct_file_name.split(".")[0].split("_")[-1])
            train_set_y_orig.append(label)

    for ct_file_name in os.listdir(test_ct_dir):
        ct_file_path = test_ct_dir + "/" + ct_file_name
        img_ct = np.load(ct_file_path)
        img_ct = img_ct[:, :, np.newaxis]
        test_set_x_ct_orig.append(img_ct)

        pet_file_name = "PETSlice".join(ct_file_name.split("CTSlice"))
        pet_file_path = test_pet_dir + "/" + pet_file_name
        img_pet = np.load(pet_file_path)
        img_pet = cv2.resize(img_pet, (128, 128))
        img_pet = img_pet[:, :, np.newaxis]
        test_set_x_pet_orig.append(img_pet)

        label = int(ct_file_name.split(".")[0].split("_")[-1])
        test_set_y_orig.append(label)

    train_set_x_ct_orig = np.array(train_set_x_ct_orig)
    train_set_x_pet_orig = np.array(train_set_x_pet_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_ct_orig = np.array(test_set_x_ct_orig)
    test_set_x_pet_orig = np.array(test_set_x_pet_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # 平衡 不同种类的样本
    train_set_x_ct_orig, train_set_x_pet_orig, train_set_y_orig = augu_two_input(train_set_x_ct_orig,
                                                                                 train_set_x_pet_orig, train_set_y_orig)

    # shuffle trainset
    m = train_set_x_ct_orig.shape[0]
    permutation = list(np.random.permutation(m))
    train_set_x_ct_orig = train_set_x_ct_orig[permutation, :, :, :]
    train_set_x_pet_orig = train_set_x_pet_orig[permutation, :, :, :]
    train_set_y_orig = train_set_y_orig[permutation]

    return train_set_x_ct_orig, train_set_x_pet_orig, train_set_y_orig, test_set_x_ct_orig, test_set_x_pet_orig, test_set_y_orig


def load_dataset_ct_pet_4():
    train_set_x_ct_orig = []
    train_set_x_pet_orig = []
    train_set_xyr = []
    train_set_y_orig = []
    test_set_x_ct_orig = []
    test_set_x_pet_orig = []
    test_set_xyr = []
    test_set_y_orig = []

    train_ct_dirs = ["/home/liubo/data/graduate/CTSlice/fold0",
                     "/home/liubo/data/graduate/CTSlice/fold1",
                     "/home/liubo/data/graduate/CTSlice/fold2",
                     "/home/liubo/data/graduate/CTSlice/fold3"
                     ]
    test_ct_dir = "/home/liubo/data/graduate/CTSlice/fold4"

    train_pet_dirs = ["/home/liubo/data/graduate/PETSlice/fold0",
                      "/home/liubo/data/graduate/PETSlice/fold1",
                      "/home/liubo/data/graduate/PETSlice/fold2",
                      "/home/liubo/data/graduate/PETSlice/fold3"
                      ]
    test_pet_dir = "/home/liubo/data/graduate/PETSlice/fold4"

    # label.csv
    label_csv = "/home/liubo/nn_project/LungSystem2/Cancercla_data_set/label.csv"
    idx_x_y_r_dict = {}
    with open("label.csv") as f_r:
        reader = csv.DictReader(f_r)
        for row in reader:
            idx = row["idx"]
            x = int(row["x_pix"]) / 512
            y = int(row["y_pix"]) / 512
            r = int(row["ct_r_pix"]) / 512
            idx_x_y_r_dict[idx] = [x, y, r]

    for i in range(len(train_ct_dirs)):
        train_ct_dir = train_ct_dirs[i]
        train_pet_dir = train_pet_dirs[i]
        for ct_file_name in os.listdir(train_ct_dir):
            ct_file_path = train_ct_dir + "/" + ct_file_name
            img_ct = np.load(ct_file_path)
            img_ct = img_ct[:, :, np.newaxis]
            train_set_x_ct_orig.append(img_ct)

            pet_file_name = "PETSlice".join(ct_file_name.split("CTSlice"))
            pet_file_path = train_pet_dir + "/" + pet_file_name
            img_pet = np.load(pet_file_path)
            img_pet = cv2.resize(img_pet, (128, 128))
            img_pet = img_pet[:, :, np.newaxis]
            train_set_x_pet_orig.append(img_pet)

            xyr = idx_x_y_r_dict[ct_file_name.split("_")[0]]
            train_set_xyr.append(xyr)

            label = int(ct_file_name.split(".")[0].split("_")[-1])
            train_set_y_orig.append(label)

    for ct_file_name in os.listdir(test_ct_dir):
        ct_file_path = test_ct_dir + "/" + ct_file_name
        img_ct = np.load(ct_file_path)
        img_ct = img_ct[:, :, np.newaxis]
        test_set_x_ct_orig.append(img_ct)

        pet_file_name = "PETSlice".join(ct_file_name.split("CTSlice"))
        pet_file_path = test_pet_dir + "/" + pet_file_name
        img_pet = np.load(pet_file_path)
        img_pet = cv2.resize(img_pet, (128, 128))
        img_pet = img_pet[:, :, np.newaxis]
        test_set_x_pet_orig.append(img_pet)

        xyr = int(ct_file_name.split(".")[0].split("_")[-1])
        test_set_xyr.append(xyr)

        label = int(ct_file_name.split(".")[0].split("_")[-1])
        test_set_y_orig.append(label)

    train_set_x_ct_orig = np.array(train_set_x_ct_orig)
    train_set_x_pet_orig = np.array(train_set_x_pet_orig)
    train_set_y_orig = np.array(train_set_y_orig)
    test_set_x_ct_orig = np.array(test_set_x_ct_orig)
    test_set_x_pet_orig = np.array(test_set_x_pet_orig)
    test_set_y_orig = np.array(test_set_y_orig)

    # 平衡 不同种类的样本
    train_set_x_ct_orig, train_set_x_pet_orig, train_set_xyr, train_set_y_orig = augu_three_input(train_set_x_ct_orig,
                                                                                                  train_set_x_pet_orig,
                                                                                                  train_set_xyr,
                                                                                                  train_set_y_orig)

    # shuffle trainset
    m = train_set_x_ct_orig.shape[0]
    permutation = list(np.random.permutation(m))
    train_set_x_ct_orig = train_set_x_ct_orig[permutation, :, :, :]
    train_set_x_pet_orig = train_set_x_pet_orig[permutation, :, :, :]
    train_set_y_orig = train_set_y_orig[permutation]

    return train_set_x_ct_orig, train_set_x_pet_orig, train_set_xyr, train_set_y_orig, test_set_x_ct_orig, test_set_x_pet_orig, test_set_xyr, test_set_y_orig


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
