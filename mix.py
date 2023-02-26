import argparse
import os
import sys
import time

from tqdm import tqdm

import cv2
import numpy as np
from numpy import random

import matplotlib as mpl
from matplotlib import pyplot as plt
import torch
#报错：OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
#OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
#添加：
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

sys.path.append('./yolov5')
from yolov5.cmodel import Model as yolo

sys.path.remove('./yolov5')

sys.path.append('./CSRNet')
from CSRNet.cmodel import Model as csr

sys.path.remove('./CSRNet')

myolo = yolo()
mcsr = csr()

import sys
from PyQt5 import QtWidgets,QtGui
from PyQT_Form import Ui_Form
from PyQt5.QtWidgets import QFileDialog
def pred(img):
    out = {}
    oyolo = myolo.predict(img)
    ocsr = mcsr.predict(img)
    out['yolo'] = oyolo[1]
    out['lyolo'] = len(oyolo[0])
    plt.clf()# Clear figure清除所有轴，但是窗口打开，这样它可以被重复使用。
    plt.imshow(ocsr, cmap=mpl.cm.hot)
    plt.savefig('hot.jpg')
    out['csr'] = cv2.imread('hot.jpg')
    out['ocsr'] = ocsr.copy()
    out['lcsr'] = np.abs(np.sum(ocsr))
    h, w = img.shape[: 2]
    ch, cw = ocsr.shape[: 2]
    sw, sh = w / cw, h / ch

    for rect in oyolo[0]:
        r = rect.astype(np.int)
        ocsr[int(r[1] / sh): int((r[3]) / sh), int(r[0] / sw): int((r[2]) / sw)] = 0
    ocsr[ocsr <= 0.001] = 0
    out['num'] = np.abs(np.sum(ocsr)) + len(oyolo[0])
    return out

class MyPyQT_Form(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(MyPyQT_Form,self).__init__()
        self.setupUi(self)

        # 利用qlabel显示图片
        png = QtGui.QPixmap('f0.jpg').scaled(self.label_1.width(), self.label_1.height())
        self.label_1.setPixmap(png)
        png = QtGui.QPixmap('f1.jpg').scaled(self.label_2.width(), self.label_2.height())
        self.label_2.setPixmap(png)
        png = QtGui.QPixmap('f2.jpg').scaled(self.label_3.width(), self.label_3.height())
        self.label_3.setPixmap(png)
        png = QtGui.QPixmap('f3.jpg').scaled(self.label_4.width(), self.label_4.height())
        self.label_4.setPixmap(png)

if __name__ == '__main__':
    # torch.cuda.empty_cache()
    # img = cv2.imread('3641.jpg')
    # out = pred(img)

    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form()
    my_pyqt_form.show()

    vw = cv2.VideoWriter('show_2.avi', cv2.VideoWriter_fourcc(*'XVID'), 25.0, (1250, 1000), True)
    ayolo = [0] * 20
    acsr = [0] * 20
    anum = [0] * 20
    syolo = [0] * 25
    scsr = [0] * 25
    snum = [0] * 25
    plt.figure(figsize=(7, 6))
    video = cv2.VideoCapture('06.mp4')
    video_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(video_count)):  #它的作用就是在终端上出现一个进度条，使得代码进度可视化。
        ret, frame = video.read()
        f0 = frame.copy()
        out = pred(frame)
        f1 = out['yolo']
        f2 = out['csr']
        syolo[i % 25] = out['lyolo']
        scsr[i % 25] = out['lcsr']
        snum[i % 25] = out['num']
        if i % 25 == 0:
            ayolo.append(np.mean(syolo))
            acsr.append(np.mean(scsr))
            anum.append(np.mean(snum))
            plt.clf()
            plt.subplot(3, 1, 1)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title('yolo', fontsize=20)
            plt.ylim(0, 40)
            plt.bar(range(15), ayolo[: len(ayolo) - 16: -1], fc='b')
            plt.subplot(3, 1, 2)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title('csr', fontsize=20)
            plt.ylim(0, 40)
            plt.bar(range(15), acsr[: len(ayolo) - 16: -1], fc='r')
            plt.subplot(3, 1, 3)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.title('mix', fontsize=20)
            plt.ylim(0, 40)
            plt.bar(range(15), anum[: len(ayolo) - 16: -1], fc='b')
            plt.subplots_adjust(hspace=0.4)
            plt.savefig('tmp.jpg')
            f3 = cv2.imread('tmp.jpg')
        f = np.ones((1000, 1250, 3), dtype=np.uint8) * 255
        f[25: 475, 50: 600] = cv2.resize(f0, (550, 450))
        f[25: 475, 650: 1200] = cv2.resize(f1, (550, 450))
        f[525: 975, 50: 600] = cv2.resize(f2, (550, 450))
        f[525: 975, 650: 1200] = cv2.resize(f3, (550, 450))
        # cv2.imshow("result", f)
        # if cv2.waitKey(1) == 27:
        #     print('stop')
        #     break
        cv2.imwrite("f0.jpg",f0);
        cv2.imwrite("f1.jpg", f1);
        cv2.imwrite("f2.jpg",f2);
        cv2.imwrite("f3.jpg", f3);

        my_pyqt_form = MyPyQT_Form()
        my_pyqt_form.show()

        #sys.exit(app.exec_())
        # vw.write(f)  #存储为视频



