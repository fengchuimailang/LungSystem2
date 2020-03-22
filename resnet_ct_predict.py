import os
import sys
sys.path.append(os.path.realpath("."))
print(sys.path)
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from resnet_ct import ResNet50
from utils import *


model_path = "/home/liubo/nn_project/LungSystem2/models/guaduate/resnet50_ct/resnet50_ct_best.hd5"

def predict():
    """
    进行预测
    """

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset_ct()

    # 不需要进行归一化
    # X_train = X_train_orig / 255.
    # X_test = X_test_orig / 255.

    # 模型
    model = ResNet50()
    model.load_weights(model_path, by_name=False)

    # 开始计时
    start_time = datetime.datetime.now()
    predIdxs = model.predict(X_test_orig)

    # 测试花费时间
    current_time = datetime.datetime.now()
    res = current_time - start_time
    print("Done in : ", res.total_seconds(), " seconds")
    print("model: " + model_path)
    print("label:")
    print(Y_test_orig.flatten())
    print("predict:")
    predict_label = np.argmax(predIdxs, axis=1)
    print(predict_label)
    # print(len(predict_label))

    # 计算预测的混淆矩阵
    confus_predict = confusion_matrix(Y_test_orig, predict_label)
    print("confusion matrix:")
    print(confus_predict)

    # 结果评估
    target_names = ['class 0', 'class 1', 'class 2', 'class 3', 'class 4']
    classification_show = classification_report(Y_test_orig, predict_label, labels=None, target_names=target_names)
    print(classification_show)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    predict()
