
# 2021/02/04 EEG_button
#  https://stackoverflow.com/a/6981055/6622587
from json import decoder
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from Ui_NewGUI import Ui_MainWindow
import shutil
import sys
import multiprocessing
import serial
import time
import numpy as np
from datetime import datetime
import os


from eeg_decoder import Decoder, Filter
import matplotlib.animation as animation
from multiprocessing import Queue

from scipy import signal

import torch
import einops
from EEGNetModel import EEGNet

plt.style.use('ggplot')

class MyMainWindow(QtWidgets. QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        # 好像沒用，不影響程式執行
        # manager = multiprocessing.Manager()
        # self.readDataBuffer = manager.Queue(1024)
        # self.sockets = list()

        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle('BCI_DATA')
        # 按鍵功能
        self.btnCon.clicked.connect(self.StartConnection)  # 連線
        self.btnDisCon.clicked.connect(self.Disconnection)  # 斷線
        self.btnSave.clicked.connect(self.Savedata)  # 存檔



        # 多線程
        self.queue_data = Queue()
        self.queue_flag = Queue()
        # 建立dt class
        self.dt               = DataReceiveThreads()  
        # 多線程 1 : 開始接收封包
        self.multipDataRecv   = multiprocessing.Process(target=self.dt.data_recv, 
                                                        args=(self.queue_data, self.queue_flag))  
       
        # 多線程 2 : 模型預測
        self.multipOnlineBCI = multiprocessing.Process(target=self.dt.online_bci, 
                                                       args=(self.queue_data, self.queue_flag))              
        # # 多線程 2 : 開始繪圖
        # self.multipRealtimePlot = multiprocessing.Process(target=self.dt.realtime_plot, 
        #                                                   args=(self.queue, )) 
        self.texbConStatus.append('****** Program is running ******')

    def StartConnection(self):  # 連線
        self.texbConStatus.append("Waiting for Connections...")
        self.multipDataRecv.start()
        
        self.texbConStatus.append("Data Receiving...")
        self.multipOnlineBCI.start()

    def Disconnection(self):  # 斷線 將check_save 變 0
        self.multipDataRecv.terminate() # 多線程1 關閉
        self.multipOnlineBCI.terminate()
        # self.multipRealtimePlot.terminate()
        self.dt.endRecv = True
        with open("check_save.txt", "w") as f:
            f.write("0")
        # time.sleep(1)
        
        self.texbConStatus.append("Data Receive Terminated.")
        localtime2 = time.asctime(time.localtime(time.time()))
        self.texbConStatus.append(localtime2)

    def Savedata(self):  # 存檔 將check_save 變 1
        self.texbConStatus.append("Data Saving...")
        with open("check_save.txt", "w") as f:
            f.write("1")
        localtime1 = time.asctime(time.localtime(time.time()))
        self.texbConStatus.append(localtime1)

        
        # time.sleep(1)
        # self.multipRealtimePlot.start()


# My code
# ------------------------------------------------------------------------ #
# Misc settings
VERSION = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Which model to use
modelName = "EEGNet"
# modelName = "EEGConformer"

# This decide which subject's data to load
subjectName = "fred"
# subjectName = "charli"
# subjectName = "eddi"

# ------------------------------------------------------------------------ #


# Path setting
currentWorkingDir = os.getcwd()
modelSaveDir = os.path.join(currentWorkingDir, "models")
modelPath = os.path.join(modelSaveDir, f"{modelName}-{subjectName}-v{VERSION}.pt")


# Load existing model
print("Loading model...")
if os.path.exists(modelPath) :
    with open(modelPath, mode='rb') as f:
        model = torch.jit.load(f)
        model.eval()
        model.to(DEVICE)

    print(f"Model is ready and running on {DEVICE}.")
else :
    print(f"{modelPath} not found.")
    exit()
print()


class DataReceiveThreads(Ui_MainWindow):
    def __init__(self):
        self.endRecv = False
        self.if_save = "0"
        self.if_done = True
        self.data = ""
        self.count = 0
        self.total_data = ""
        self.small_data = ""

        self.decoder = Decoder()
        self.filter  = Filter()

        # 創立當前時間的txt檔案
        ts = time.time()
        data_time = datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H%M')
        self.fileDir = './exp/{}'.format(data_time)

        if not os.path.isdir(self.fileDir):
            os.mkdir(self.fileDir)
            os.mkdir(self.fileDir + '\\1\\')
        else:
            shutil.rmtree(self.fileDir)
            os.mkdir(self.fileDir)
            os.mkdir(self.fileDir + '\\1\\')
        
        self.fileName = 'EEG.txt'

    def data_recv(self, queue_data, queue_flag):      
        ser = serial.Serial('COM5', 460800)
        print("------------------------ Successfull Receive! ------------------------")
        while True:
            # 很重要，不加這幾行會讀到之前存在腦波機buffer的資料
            ser.reset_output_buffer()
            ser.reset_input_buffer()

            # 判斷 Savedata 按鍵是否觸發
            with open("check_save.txt", "r") as f:
                self.if_save = f.read()                    
                
            if self.if_save == "1":                
                time_start = time.time()
                while True:                        
                    with open("check_save.txt", "r") as f:
                        self.if_save = f.read()

                    # 結束後寫入最後收到的資料到EEG.txt
                    if self.if_save == "0" and self.if_done:
                        with open('{}/1/{}'.format(self.fileDir, self.fileName), "a") as f:
                            f.write(self.total_data)
                        self.if_done = False                        
                    elif self.if_save == "1":
                        # 每次讀取 32 bytes(一組EEG data的大小)並轉成16進位。收一次等於 1ms 的時間
                        self.data       = ser.read(32).hex()
                        self.total_data = self.total_data + self.data                        
                        self.count      = self.count + 1

                        # 存 3000 ms 資料後，每 100ms 資料到txt的最尾端
                        # total_data的長度理論上等於 192000 = 64 timepoints * 3000 ms (timepoints)                        
                        if self.count == 3000:
                            queue_flag.put(True)
                            # 將3s的資料丟進queue
                            self.small_data = self.small_data + self.total_data
                            queue_data.put(self.small_data)
                            
                            # 寫入txt檔案
                            with open('{}/1/{}'.format(self.fileDir, self.fileName), "a") as f:
                                f.write(self.total_data)
                            
                            # 經過 100 ms，raw長度 = 64*100 = 6400
                            self.count     -= 100
                            self.total_data = ""
                            self.small_data = self.small_data[6400:]
                            # 打印經過時間
                            time_end = time.time()
                            time_gap = time_end - time_start
                            # print('time: {:.1f}'.format(time_gap))
                        else:
                            queue_flag.put(False)
  

    def online_bci(self, queue_data, queue_flag):
        # 預先載入模型，避免延遲
        temp   = np.zeros((1, 8, 500, 1))
        temp = torch.from_numpy(temp)
        temp = einops.rearrange(temp, "b h w c -> b c h w")
        temp = temp.type(torch.float32)
        temp = temp.to(DEVICE)

        y_prob = model.predict(temp)
        y_prob = y_prob.detach().numpy()
        y_pred = np.argmax(y_prob, axis=1)
        print("------------------------ Ready to start ------------------------")  

        # 初始化滑動窗和平滑輸出值
        WINDOW_SIZE = 15        
        NUM_CLASSES = 5
        pred_window = np.zeros((WINDOW_SIZE, NUM_CLASSES))
        while True:
            flag = queue_flag.get()

            if flag == True:  
                print(queue_data.qsize(), end = ' ')
                raw = queue_data.get()

                # Decode & filtering
                eeg_raw = self.decoder.get_BCI(raw, show_progress = False)
                eeg_small_filtered = self.filter.filter(eeg_raw)

                temp = eeg_small_filtered[900:-100].T
                temp = temp[0:8]
                temp = np.expand_dims(temp, axis = 0)
                temp = signal.resample(temp, 500, axis = 2) # resample signal based on re_samplefreq
                temp = temp.reshape(temp.shape[0], 8, 500, 1)
                # temp = einops.rearrange(temp, "b h w c -> b c h w")
                temp = torch.from_numpy(temp)
                temp = temp.type(torch.float32)
                temp = einops.rearrange(temp, "b h w c -> b c h w")
                temp = temp.to(DEVICE)

                temp = (temp - 3.3937137411810585e-10)/3.4349564577068914e-06 # 穩定
                # temp = (temp - 5.721349780654223e-10)/3.330297919169182e-06 # 敏感

                # prediction
                # y_prob      = model.predict(temp, verbose=0)
                y_prob      = model.predict(temp)
                y_prob = y_prob.detach().numpy()
                y_pred = np.argmax(y_prob, axis=1)

                pred_window[1:, :] = pred_window[:-1, :]
                pred_window[0, y_pred[0]] += 1

                # 计算加权平均值
                weights = np.sum(pred_window, axis=0)
                avg_output = np.sum(pred_window * np.arange(NUM_CLASSES), axis=0) / weights

                # 如果平均值超过阈值0.5，认为处于激活状态，否则为休息状态
                if avg_output[-1] > 0.5:
                    smooth_output = np.argmax(avg_output[:-1]) + 1
                else:
                    smooth_output = 0

                # # 平滑化輸出結果
                # # 將新的輸出放進窗口
                # pred_window[1:] = pred_window[:-1]
                # pred_window[0]  = y_pred[0]
                # # 計算滑動窗口輸出平均值
                # avg_output = np.mean(pred_window)
                # # 如果平均值超過閾值0.5，視為激活狀態，否則為休息狀態
                # if avg_output > 0.8:
                #     smooth_output = 1
                # else:
                #     smooth_output = 0

                now = datetime.now()
                time_str = now.strftime('%H-%M-%S-%f')[:-5]
                print(f"{time_str} {smooth_output}")
                with open("pred.txt", mode="w") as f:
                    # f.write(str(y_pred[0]))
                    f.write(str(smooth_output))

                flag = False


    # def realtime_plot(self, queue):
    #     rtplot = RealtimePlot(self.fileDir, queue, max_t = 2000)



class RealtimePlot():
    def __init__(self, fileDir, queue, max_t=2000):
        # self.ax = ax
        self.max_t = max_t # 視窗顯示秒數長度
        self.tdata = np.empty(0) 
        self.ydata = np.empty((10, 1))
        self.t_total = 0    
        self.fileDir = fileDir   
        self.raw_total = '' # for test
        self.cut_point = 1000 # remove filtered data points before cut_point
        self.decoder = Decoder()
        self.queue = queue
        self.eeg_total = np.empty((1, 10))
        self.filter = Filter()

        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, 1, figsize=(10, 6))
        self.ax = (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8)

        # plot parameter
        self.color = '#17395C' #17395C # steelblue
        self.linewidth = 0.8
        self.channel_list = ['F3', 'F4', 'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4']
 
        ani = animation.FuncAnimation(fig=fig, 
                                      init_func=self.init_func,
                                      func=self.update, 
                                      interval=30, 
                                      frames = self.data_gen, 
                                      blit=True,
                                      repeat=False,
                                      save_count=200)
        plt.show()

    def init_func(self):
        # 要分開寫，用for產生會有問題
        line1, = self.ax[0].plot([], [], c=self.color, lw=self.linewidth)
        line2, = self.ax[1].plot([], [], c=self.color, lw=self.linewidth)
        line3, = self.ax[2].plot([], [], c=self.color, lw=self.linewidth)
        line4, = self.ax[3].plot([], [], c=self.color, lw=self.linewidth)
        line5, = self.ax[4].plot([], [], c=self.color, lw=self.linewidth)
        line6, = self.ax[5].plot([], [], c=self.color, lw=self.linewidth)
        line7, = self.ax[6].plot([], [], c=self.color, lw=self.linewidth)
        line8, = self.ax[7].plot([], [], c=self.color, lw=self.linewidth)
        self.line = [line1, line2, line3, line4, line5, line6, line7, line8]
        xticks = [x for x in range(0, self.max_t+1, 1000)]
        xticklabels = [str(int(time/1000)) for time in range(0,  self.max_t+1, 1000)]        
        for i in range(8):
            self.ax[i].set_xticks(xticks)            
            self.ax[i].set_xlim(0, self.max_t)
            # remove x label except the bottom plot
            if i == 7: # ax8
                self.ax[i].set_xticklabels(xticklabels)
            else: # ax1 ~ ax7
                self.ax[i].axes.xaxis.set_ticklabels([])
            self.ax[i].set_ylabel(self.channel_list[i], fontsize=14, rotation=0) # channel name    
            self.ax[i].yaxis.set_label_coords(-0.1, .35) # set the label position  

        return self.line

    def update(self, y):     
        # input 
        #   y : raw data from EEG.txt

        self.raw_total += y
        eeg_raw = np.array(self.decoder.decode(self.raw_total)) # shape = (n, 10)
        # print(eeg_raw.shape)
        
        # rest the parameter
        if len(eeg_raw)-self.cut_point-1 >= self.max_t: # 長度超過顯示秒數就重畫x座標軸
            self.t_total += len(eeg_raw[0])
            self.ydata =np.empty((10, 1))
            
            self.raw_total = ''
            xticklabels = [str(int(time/1000)-1) for time in range(self.t_total, self.t_total + self.max_t+1, 1000)] # name of x_ticks (time)
            self.ax[7].set_xticklabels(xticklabels)
            self.ax[7].figure.canvas.draw() # redraw everything, would made animation slow        
        
        # 捨棄前一秒及最後一筆濾波後的資料，因為有問題
        self.tdata = np.arange(len(eeg_raw)-self.cut_point-1)
        self.ydata = eeg_raw[self.cut_point:-1].T # shape = (n, 10)
        for i in range(8):
            self.line[i].set_data(self.tdata, self.ydata[i])
            self.ax[i].set_ylim([-4e-5, 4e-5]) # 腦波電壓振幅範圍
            # self.ax[i].relim() 
            # self.ax[i].autoscale_view()
        return self.line
    
    def data_gen(self):        
        # retrun raw data each interval time        
        while os.path.exists('{}/1/{}'.format(self.fileDir, 'EEG.txt')):
            try:
                # f = open('{}/1/{}'.format(self.fileDir, 'EEG_small.txt'), "r")
                # raw = f.read()
                # f.close()     
                print(self.queue.qsize())                 
                raw = self.queue.get()     
                print(self.queue.qsize())
                yield raw 
            except Exception:
                pass        
      
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MyMainWindow()
    mainWindow.show()
    sys.exit(app.exec_())    




