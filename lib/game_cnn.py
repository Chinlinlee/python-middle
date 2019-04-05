import time
import cv2
import sys
import gym
import tensorflow as tf
import numpy as np
import random
import datetime
import copy

from pathlib import Path
from collections import deque
from lib.Test_forai import *


# src 
# http://blog.topspeedsnail.com/archives/10459
# need to modify the model by referring the following
# http://blog.csdn.net/superCally/article/details/54784103

class CNN_CONFIG():
  CNN_INPUT_WIDTH = 80
  CNN_INPUT_HEIGHT = 80
  CNN_INPUT_DEPTH = 1
  SERIES_LENGTH = 4
  
  ACTION_DIM=Game_Config().ACTION_DIM
  
  REWARD_COFF = 3.0 # unused

  INITIAL_EPSILON = 1.0
  FINAL_EPSILON = 0.0001
  REPLAY_SIZE = 50000
  BATCH_SIZE = 32
  GAMMA = 0.99
  OBSERVE_TIME = 500
  ENV_NAME = 'Breakout-v0' # unused
  EPISODE = 100000
  STEP = 1500
  TEST = 10
  
  FPS = 50.0 # for game running
  
  FILENAME_PATH = "./0=data/"
  
  FILENAME_LOG = FILENAME_PATH + "0=log.txt"
  
  loading_instinct =  True
   
  # input會有(SERIES_LENGTH = 4)張圖疊起來
  INPUT_DEPTH = SERIES_LENGTH
pass



########## CNN

class AIGame():
  def __init__(self):
    self.env = Game()
    self.agent = TrainingModel()
  pass

  def ai_play_start(self):
    n = 0
    total_text="0"
    total_success =0.0
    total_error =1.0
    total_ratio =0.0
    total_time =0.0

    state_shadow = None
    next_state_shadow = None

    total_reward_decade = 0

    ##xrange跟range最大的差別就是：
    ##1. range 是全部產生完後，return一個 list 回來使用。
    ##2. xrange 是一次產生一個值，並return一個值回來，所以xrange只適用於loop。
    ## EPISODE = 100000
    ## => 下面是跑100,000次
    for episode in range(CNN_CONFIG.EPISODE):

      total_reward = 0
      
      self.agent.init_sample( self.env.current_game_img())
      new_sample=None
      ## STEP = 1500
      for step in range(CNN_CONFIG.STEP):
          
        # 取得訓練用所要求的遊戲動作
        action = self.agent.get_action_by_ai(new_sample)
        
        # 執行遊戲動作，並取得畫面(next_state)，獎勵(reward)，是否遊戲結束(done)
        new_sample, reward, done = self.env.step(action)

        time.sleep(1/CNN_CONFIG.FPS)

        ############## 設定log分析
        total_reward += reward # for log
        n = n+1
        if reward !=0:
          if reward > 0:
            total_success=total_success+1.0
          elif reward < 0:
            total_error =total_error+1.0
          pass
        pass
          
        if n%10000 == 0:
          total_time = (total_success+total_error)
          total_ratio = total_success / (total_success+total_error)
          now = datetime.datetime.now()
          now = now.strftime("%Y%m%d=%H.%M.%S")
          total_text = "{}, {}, {}, {:.0f}, {:.0f}, {:0.2}\n".format(n, now, n,total_time,total_success, total_ratio)
          print(total_text)

          # 開啟檔案
          fp = open(CNN_CONFIG.FILENAME_LOG, 'a+')
           
          # 將 lines 所有內容寫入到檔案
          fp.write(total_text)
          total_success=0.0
          total_error=0.0
          total_text=""
          # 關閉檔案
          fp.close()
        pass
        
        ##############
        
        # 停止遊戲
        if done:
          break
        pass
      pass
    pass
    print ('Episode:', episode, 'Total Point this Episode is:', total_reward)
  pass
  
  def train_start(self):
    n = 0
    total_text="0"
    total_success =0.0
    total_error =1.0
    total_ratio =0.0
    total_time =0.0

    state_shadow = None
    next_state_shadow = None

    total_reward_decade = 0

    ##xrange跟range最大的差別就是：
    ##1. range 是全部產生完後，return一個 list 回來使用。
    ##2. xrange 是一次產生一個值，並return一個值回來，所以xrange只適用於loop。
    ## EPISODE = 100000
    ## => 下面是跑100,000次
    for episode in range(CNN_CONFIG.EPISODE):

      total_reward = 0
      
      self.agent.init_sample( self.env.current_game_img())
   
      ## STEP = 1500
      for step in range(CNN_CONFIG.STEP):
          
        # 取得訓練用所要求的遊戲動作
        action = self.agent.get_action_for_training(self.agent.state_shadow)
        
        # 執行遊戲動作，並取得畫面(next_state)，獎勵(reward)，是否遊戲結束(done)
        new_sample, reward, done = self.env.step(action)

        # 開始訓練
        self.agent.train_instinct(action, reward, new_sample, done, episode)
        
        # save instinct 
        if n % 10000 == 0:
          self.agent.save_instinct(n)
        pass

        ############## 設定log分析
        total_reward += reward # for log
        n = n+1
        if reward !=0:
          if reward > 0:
            total_success=total_success+1.0
          elif reward < 0:
            total_error =total_error+1.0
          pass
        pass
          
        if n%10000 == 0:
          total_time = (total_success+total_error)
          total_ratio = total_success / (total_success+total_error)
          now = datetime.datetime.now()
          now = now.strftime("%Y%m%d=%H.%M.%S")
          total_text = "{}, {}, {}, {:.0f}, {:.0f}, {:0.2}\n".format(n, now, n,total_time,total_success, total_ratio)
          print(total_text)

          # 開啟檔案
          fp = open(CNN_CONFIG.FILENAME_LOG, 'a+')
           
          # 將 lines 所有內容寫入到檔案
          fp.write(total_text)
          total_success=0.0
          total_error=0.0
          total_text=""
          # 關閉檔案
          fp.close()
        pass
        
        ##############
        
        # 停止遊戲
        if done:
          break
        pass
      pass
    pass
    print ('Episode:', episode, 'Total Point this Episode is:', total_reward)
  pass
pass

class ImageProcess():
    def ColorMat2B(self,state):   # this is the function used for the game flappy bird
        height = CNN_CONFIG.CNN_INPUT_HEIGHT
        width = CNN_CONFIG.CNN_INPUT_WIDTH
        state_gray = cv2.cvtColor( cv2.resize( state, ( height, width ) ) , cv2.COLOR_BGR2GRAY )
        _,state_binary = cv2.threshold( state_gray, 5, 255, cv2.THRESH_BINARY )
        state_binarySmall = cv2.resize( state_binary, ( width, height ))
        cnn_inputImage = state_binarySmall.reshape( ( height, width ) )
        return cnn_inputImage

    def ColorMat2Binary(self, state):
        # state_output = tf.image.rgb_to_grayscale(state_input)
        # state_output = tf.image.crop_to_bounding_box(state_output, 34, 0, 160, 160)
        # state_output = tf.image.resize_images(state_output, 80, 80, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # state_output = tf.squeeze(state_output)
        # return state_output

        height = state.shape[0]
        width = state.shape[1]
        nchannel = state.shape[2]

        sHeight = int(height * 0.5)
        sWidth = CNN_CONFIG.CNN_INPUT_WIDTH

        state_gray = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        # print state_gray.shape
        # cv2.imshow('test2', state_gray)
        # cv2.waitKey(0)

        _, state_binary = cv2.threshold(state_gray, 5, 255, cv2.THRESH_BINARY)

        state_binarySmall = cv2.resize(state_binary, (sWidth, sHeight), interpolation=cv2.INTER_AREA)

        cnn_inputImg = state_binarySmall[25:, :]
        # rstArray = state_graySmall.reshape(sWidth * sHeight)
        cnn_inputImg = cnn_inputImg.reshape((CNN_CONFIG.CNN_INPUT_WIDTH, CNN_CONFIG.CNN_INPUT_HEIGHT))
        # print cnn_inputImg.shape

        return cnn_inputImg

    def ShowImageFromNdarray(self, state, p):
        imgs = np.ndarray(shape=(4, CNN_CONFIG.CNN_INPUT_WIDTH, CNN_CONFIG.CNN_INPUT_HEIGHT))

        for i in range(0, CNN_CONFIG.CNN_INPUT_WIDTH):
            for j in range(0, CNN_CONFIG.CNN_INPUT_HEIGHT):
                for k in range(0, 4):
                    imgs[k][i][j] = state[i][j][k]

        cv2.imshow(str(p + 1), imgs[0])
        cv2.imshow(str(p + 2), imgs[1])
        cv2.imshow(str(p + 3), imgs[2])
        cv2.imshow(str(p + 4), imgs[3])
pass

class AI_Logic():

  def __init__(self):
    self.action_dim= CNN_CONFIG.ACTION_DIM
    self.init_ai_logic()
    
  pass

  def init_ai_logic(self):


    ###########
    # 0. 初始化
    ###########
    
    ######################
    # 0.1. 初始化self.X_input_pic，用來存sample X1
    # array([]x32)
    # input的資料為80x80x4 (四張圖)
    self.X_input_pic = tf.placeholder(tf.float32, 
                    [None, CNN_CONFIG.CNN_INPUT_WIDTH, CNN_CONFIG.CNN_INPUT_HEIGHT, CNN_CONFIG.INPUT_DEPTH],
                    name='status-input') # input的變數是status-input
    # 設定input_layer是80*80(一張圖)*4(四張圖)的陣列，這個陣列變數名字叫做'status-input'
    # tf.placeholder(dtype, shape=None, name=None) 
    # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
    # shape：数据形状。默认是None，就是一维值；多维，比如[2,3]；【行】可變動，3列，設定是 [None, 3]
    # name： 變數名称
    # CNN_INPUT_WIDTH = 80
    # CNN_INPUT_HEIGHT = 80
    # CNN_INPUT_DEPTH = 1
    # INPUT_DEPTH = 4
    
    # 0.2. 初始化self.X_input_action，用來存sample X2，型態為陣列(1x3) (向左，不動，向右)
    # array([0., 0., 1.]x32)
    self.X_input_action = tf.placeholder(tf.float32, [None, self.action_dim])
    
    # 0.3. 初始化self.Y_expect_action_reward，用來存Y
    # reward: [9.106881952285766]x32
    self.Y_expect_action_reward = tf.placeholder(tf.float32, [None])

    ################
    # 1. 開始取樣
    ################
    
    ###############################
    # 1.1. 第1次取樣，使用relu
     
    # 1.1.1. 初始化 filter (32個filter 8*8*4，filter內容為常態分配的random值)
    #        初始化 bias (一個擁有32個元素的一維陣列)
    W1 = self.init_weight_var([8, 8, 4, 32])
    b1 = self.init_bias_var([32])

    # 1.1.2. 設定第1次取樣的公式
    h_conv1 = tf.nn.relu(tf.nn.conv2d(self.X_input_pic, W1, strides=[1, 4, 4, 1], padding='SAME') + b1)
    # tf.nn.conv2d是取樣，取樣完之後再加上b1這個bias，然後再使用relu函式
    # stride = 4
    # 【stride設定維：strides[0] = strides[3] = 1，strides[2]是y軸，strides[3]是x軸】。
    # 绝大多数情况下，水平的stride和竖直的stride一样，即strides = [1, stride, stride, 1]
    
    ################################
    # 1.2. 第2次取樣，使用max_pool
    conv1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    ################################
    # 1.3. 第3次取樣
    # 1.3.1. 初始化兩個 filter (64個filter 4*4*32，filter內容為常態分配的random值)
    W2 = self.init_weight_var([4, 4, 32, 64])
    b2 = self.init_bias_var([64])
    # 1.3.2. 設定第3次取樣的公式
    h_conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W2, strides=[1, 2, 2, 1], padding='SAME') + b2)

    ################################
    # 1.4. 第4次取樣
    # 1.4.1. 初始化 filter (64個filter 3*3*32，filter內容為常態分配的random值)
    W3 = self.init_weight_var([3, 3, 64, 64])
    b3 = self.init_bias_var([64])
    # 1.4.2. 設定第4次取樣的公式
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, W3, strides=[1, 1, 1, 1], padding='SAME') + b3)

    ################################
    # 1.4. 第5次取樣
    
    # 1.4.0.【轉成類神經網路的一維陣列輸入格式: Reshape成N*1*1600】
    # (5*5*64)/(1*1600) => N=512
    conv3_flat = tf.reshape(h_conv3, [-1, 1600])
     
    # 1.4.1. 建立 filter 
    W_fc1 = self.init_weight_var([1600, 512])
    b_fc1 = self.init_bias_var([512])
    
    # 1.4.2. 設定第5次取樣的公式，將1x1x1600陣列，取樣轉成1x512陣列的樣本
    h_fc1 = tf.nn.relu(tf.matmul(conv3_flat, W_fc1) + b_fc1)
    
    ################################
    # 2. 完成取樣，建立AI_Logic 模型: Y=A1*X1+A2*X2+B
    
    # 2.1. 預測三個動作的reward value
    # 2.1.1. 初始化filter，(全連接 512個中間層，最後面接上3個output: [左移, 不動, 右移])
    W_fc2 = self.init_weight_var([512, self.action_dim])
    b_fc2 = self.init_bias_var([self.action_dim])

    # 2.1.2. 根據(a)2.1.1.的參數、(b)self.X_input_pic抽樣後的樣本 (h_fc1)
    #        ，預測三個動作的reward值
    self.Exp_Action_Reward = tf.matmul(h_fc1, W_fc2) + b_fc2
    # Exp_Action_Reward [1,3]  = (h_fc=[512] )x (W_fc2 = [512,3]) + bias [1,3]
    # 在這個步驟，即是左移、不動、右移的reward
    # 挑出最大的reward，就可以決定要左移、右移、不動
    
    # 2.2. 計算目前的動作所得到的reward值
    Curr_Action_Reward = tf.reduce_sum(
                            tf.multiply(self.Exp_Action_Reward, self.X_input_action)
                            , reduction_indices=1)
    
    # X_input_action =[0,1,0], 
    # self.Exp_Action_Reward = [6.2,2.3,5.6]
    # self.Curr_Action_Reward = [0,2.3,0]
    # self.Curr_Action_Reward 是做出X_input_action的動作的獎勵
    
    ######################################
    # 3. 利用Adam函式與大量樣本，來求解 Y=A1*X1+A2*X2+B關係式

    ############
    # 3.1. 設定目標函數cost的計算方式
    self.cost = tf.reduce_mean(tf.square(self.Y_expect_action_reward - Curr_Action_Reward))
    # reduce_mean 陣列元素平均
    # square 陣列元素平方
    # y_input 是目前這個動作的q value，Action_Reward是預估的q value
    
    
    ############
    # 3.2. 指定AI_Logic所使用的優化函式 Adam Optimizer
    self.optimizer = tf.train.AdamOptimizer(1e-6).minimize(self.cost)

    #############
    # Adam:  adaptive moment estimation，自适应矩估计。
    # 概率论中矩的含义是：如果一个随机变量 X 服从某个分布，X 的一阶矩是 E(X)，
    # 也就是样本平均值，X 的二阶矩就是 E(X^2)，也就是样本平方的平均值。
    # Adam 算法根据损失函数对每个参数的梯度的一阶矩估计和二阶矩估计动态调整针
    # 对于每个参数的学习速率。Adam  也是基于梯度下降的方法，
    # 但是每次迭代参数的学习步长都有一个确定的范围，不会因为很大的梯度导致很大
    # 的学习步长，参数的值比较稳定，可以避免陷入區域最佳值後，跳不出來。
    # it does not require stationary objective, works with sparse gradients, 
    # naturally performs a form of step size annealing。
    ##############
    
  pass
  
  def init_weight_var(self, shape):
    weight = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(weight)
  pass
  
  def init_bias_var(self, shape):
    bias = tf.constant(0.01, shape=shape)
    return tf.Variable(bias)
  pass
pass


class AI_Session():

  def __init__(self):
    # 初始化圖像處理工具
    self.init_img_process()
    
    # 初始化sampling的參數
    self.init_sampling_setting()

    # 初始化ai_logic的參數
    self.ai_logic=None
    
  pass
  
  def loading_instinct(self):
    # max_to_keep 参数，这个是用来设置保存模型的个数，默认为5，
    # 即 max_to_keep=5，保存最近的5个模型。如果你想每训练一代（epoch)
    # 就想保存一次模型，则可以将 max_to_keep设置为None或者0

    self.saver = tf.train.Saver(max_to_keep=0)

    # 如果已經之前跑過了，load過去的資料 (檢查checkpoint檔案)
    # 如果之前沒跑過，重新初始化變數
    # http://www.itwendao.com/article/detail/350878.html
    if CNN_CONFIG.loading_instinct:
      tf.reset_default_graph()
      module_file=None
      
      self.epsilon = 0.01
      
      module_file =  tf.train.latest_checkpoint(CNN_CONFIG.FILENAME_PATH)

      self.saver.restore(self.session, module_file)
      print("Model restored")
    else:
      self.session.run(tf.global_variables_initializer())
      print("Initialized")
    pass
  pass
  
  def init_sampling_setting(self):
    self.epsilon = CNN_CONFIG.INITIAL_EPSILON
    self.replay_buffer = deque()
    self.recent_history_queue = deque()
    
    self.action_dim= CNN_CONFIG.ACTION_DIM
    self.state_dim = CNN_CONFIG.CNN_INPUT_HEIGHT * CNN_CONFIG.CNN_INPUT_WIDTH
    
    self.time_step = 0
    self.observe_time = 0
    
    #self.session = tf.InteractiveSession()
    self.session = tf.Session()
  pass
  
  
  def init_img_process(self):
    self.imageProcess = ImageProcess()
  pass

  def train_instinct(self):
    self.time_step += 1
     
    # 1. 先從replay buffer的50,000樣本裡面，隨機挑出CNN_CONFIG.BATCH_SIZE=32個樣本
    minibatch = random.sample(self.replay_buffer, CNN_CONFIG.BATCH_SIZE)
    
    # 2. 把CNN_CONFIG.BATCH_SIZE=32樣本內的五個資料拿出來
    state_batch = [data[0] for data in minibatch]  # state_shadow: 4個連續照片
    action_batch = [data[1] for data in minibatch] # action: 左移、不動、右移 [0,1,0]
    reward_batch = [data[2] for data in minibatch] # reward: 接到球、沒接到球、球還在空中的reward
    next_state_batch = [data[3] for data in minibatch] # state_shadow_next: 把新的圖放進去的4個連續照片 (去掉最舊的)
    done_batch = [data[4] for data in minibatch] # done: death or win, False (遊戲還沒結束)/True (遊戲結束)
    
    # 3. 把【4個連續照片】存進去self.X_input_pic這個變數，透過取樣，轉成【左移、不動、右移】的預期rewards
    #    裡面是存32個樣本的【左移、不動、右移】相對應的rewards
    #    q value [[ 3.7393541   3.6004448   3.5963404 ], ...]
    Q_value_batch = self.session.run(self.ai_logic.Exp_Action_Reward,feed_dict={self.ai_logic.X_input_pic: next_state_batch})

    # 4. 計算Y_expect_action_reward
    # y_batch 是32個rewards值,陣列大小是[32] (一維陣列)
    y_batch = []

    for i in range(CNN_CONFIG.BATCH_SIZE):

      if done_batch[i]:
        y_batch.append(reward_batch[i])
      else:
        y_batch.append(reward_batch[i] + CNN_CONFIG.GAMMA * np.max(Q_value_batch[i]))
      pass
    pass
   
    # 5. 透過這次取樣，利用AI_Logic，調整Ai與Bi，來建立  Y=Ai*X+Bi  的線性關係
    # 把state_batch. action_batch, y_batch 這三個樣本丟到類神經網路，讓optmizer去調整上面一系列取樣的參數
    self.session.run(self.ai_logic.optimizer,feed_dict={
        # 設定X1: X_input_pic
        # self.X_input_pic是運算function: 把80x80x4的圖(state_batch)，取樣成 1X1024的一維陣列的算式
        self.ai_logic.X_input_pic: state_batch,
        
        # 設定X2: X_input_action
        # self.X_input_action是儲存資料的變數: 負責記錄下一步是怎麼走的的變數: 左移、右移、不動
        #  => self.X_input_action = tf.placeholder(tf.float32, [None, self.action_dim])
        # 這邊存的是32個sample的動作，值為
        # [array([0., 0., 1.]),...]
        self.ai_logic.X_input_action: action_batch,
        
        # 設定Y: Y_expect_action_reward 
        # self.input: 用來更新R-value (Y) 後，ann更新Ai時，所需使用的變數
        #  => self.Y_expect_action_reward = tf.placeholder(tf.float32, [None])
        # self.Y_expect_action_reward 是32個rewards值,陣列大小是[32] (一維陣列)
        # [9.106881952285766, ...]
        # 做了X_input_action後的reward值，是一個值，不是陣列
        self.ai_logic.Y_expect_action_reward: y_batch

    })
    
  pass

  def insert_ai_logic(self, ai_logic_input):
    self.ai_logic=ai_logic_input
  pass
  
  def update_sample_and_training(self, state_shadow, action_input, reward, state_shadow_next, done, episode):
  
    # 新增action matrix，把這次採取的行動action index的位置設成1
    action  = action_input
    # replay_buffer 裡面會放CNN_CONFIG.REPLAY_SIZE=50,000個樣本
    self.replay_buffer.append([state_shadow, action, reward, state_shadow_next, done])

    self.observe_time += 1
    if self.observe_time % 1000 and self.observe_time <= CNN_CONFIG.OBSERVE_TIME == 0:
      print (self.observe_time)
    pass
    
    # 如果 replay_buffer 放超過CNN_CONFIG.REPLAY_SIZE=50,000個樣本，丟掉最舊的樣本
    if len(self.replay_buffer) > CNN_CONFIG.REPLAY_SIZE:
      self.replay_buffer.popleft()
    pass
    
    # 如果 replay_buffer 裡面已經有超過CNN_CONFIG.BATCH_SIZE=32個樣本
    # 且 self.observe_time超過 CNN_CONFIG.OBSERVE_TIME= 500
    # 則開始訓練
    if len(self.replay_buffer) > CNN_CONFIG.BATCH_SIZE and self.observe_time > CNN_CONFIG.OBSERVE_TIME:
      self.train_instinct()
    pass
  pass
  
  def get_action_by_ai(self, state_shadow):
  
    # 把next_state_batch (四張80x80的圖)，存進去self.X_input_pic這個變數
    # 然後self.Exp_Action_Reward這個function會把next_state_batch轉成 Action[1x3]的矩陣
    # 並把Action[1x3]這個矩陣存在rst
    rst = self.session.run(self.ai_logic.Exp_Action_Reward,feed_dict={self.ai_logic.X_input_pic: [state_shadow]})[0]
    # print rst
    # print (np.max( rst ))
    # 回傳action裡面reward最高的action
    action_array= np.zeros(self.action_dim)
    action_array[np.argmax(rst)]=1
    
    # return np.argmax(rst)
    return action_array
  pass

  def get_action_for_training(self, state_shadow):
    action_index = None
    
    if self.epsilon >= CNN_CONFIG.FINAL_EPSILON and self.observe_time > CNN_CONFIG.OBSERVE_TIME:
        self.epsilon -= (CNN_CONFIG.INITIAL_EPSILON - CNN_CONFIG.FINAL_EPSILON) / 10000
    pass
    
    action = np.zeros(self.action_dim)
    action_array=None
    # change these code while using the model (just need self.get_action())
    if random.random() < self.epsilon:
        action[random.randint(0, self.action_dim - 1)]=1

    else:
        action = self.get_action_by_ai(state_shadow)
    pass
    

    return action
  pass


pass

class TrainingModel():
  def __init__(self):
    self.ai_session = None
    
    self.init_ai_session(AI_Logic())
  pass
  
  def init_ai_session(self, ai_logic_input):
    self.ai_session=AI_Session()
    
    self.ai_session.insert_ai_logic(ai_logic_input)
    self.ai_session.loading_instinct()
  pass
  
  def init_sample(self, sample_input):

    sample = self.ai_session.imageProcess.ColorMat2B(sample_input)  # now state is a binary image of 80 * 80
    ##state = self.agent.imageProcess.ColorMat2Binary(state)  # now state is a binary image of 80 * 80
    ## 存4個sample
    self.state_shadow = np.stack((sample, sample, sample, sample), axis=2)
  pass
  
  def get_action_by_ai(self, sample):
    if sample is not None:
      # 將遊戲畫面縮圖成黑白與80x80大小的圖
      next_state = np.reshape( self.ai_session.imageProcess.ColorMat2B( sample ), ( CNN_CONFIG.CNN_INPUT_HEIGHT,CNN_CONFIG.CNN_INPUT_WIDTH,1 ) )

      # 將新的遊戲畫面存進去 AI所需要的四張圖 X_input_pic
      next_state_shadow = np.append( next_state, self.state_shadow[ :,:,:3 ], axis= 2 )

      
      # 將X_input_pic的資料存起來，當作下個動作所需要的X_input_pic資料
      self.state_shadow = next_state_shadow
    pass
    
    return self.ai_session.get_action_by_ai(self.state_shadow)

  pass
  
  def get_action_for_training(self, sample):
    return self.ai_session.get_action_for_training(sample)
  pass
  
  def train_instinct(self, action, reward, sample , done, episode):
        
    # 將遊戲畫面縮圖成黑白與80x80大小的圖
    next_state = np.reshape( self.ai_session.imageProcess.ColorMat2B( sample ), ( CNN_CONFIG.CNN_INPUT_HEIGHT,CNN_CONFIG.CNN_INPUT_WIDTH,1 ) )

    # 將新的遊戲畫面存進去 AI所需要的四張圖 X_input_pic
    next_state_shadow = np.append( next_state, self.state_shadow[ :,:,:3 ], axis= 2 )
    
    self.ai_session.update_sample_and_training(self.state_shadow, action, reward, next_state_shadow, done, episode)
    
    # 將X_input_pic的資料存起來，當作下個動作所需要的X_input_pic資料
    self.state_shadow = next_state_shadow
  pass
  
  def save_instinct(self, n):
    now = datetime.datetime.now()
    now = now.strftime("=%Y%m%d.%H%M%S=")
    file_name = './0=data/game.' + now +'.ckpt'
    # 第一个参数sess,这个就不用说了。第二个参数设定保存的路径和名字
    # 第三个参数将训练的次数作为后缀加入到模型名字中。
    self.ai_session.saver.save(self.ai_session.session, file_name, global_step = n)  # 保存模型
  pass
pass