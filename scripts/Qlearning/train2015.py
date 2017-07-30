#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Ros node for Reinforcement learning
Edited by Jeong-Hwan Moon, IRL, Pusan National UNIV. mwg900@naver.com

Original algorithm : Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
""" 
import numpy as np
import tensorflow as tf
import random
import rospy
import dqn
import policy
import register
import sys
import csv              

from collections import deque

from IRL_learning_ros.srv import SpawnPos
from IRL_learning_ros.msg import State
from std_msgs.msg import Int8
from argparse import Action
from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch

STOP = 99

#Hyper parameter
ENVIRONMENT = rospy.get_param('/driving_train/environment', 'v0')
MODEL_PATH = rospy.get_param('/driving_train/model_path', default = 'model')
INIT_EPISODE = rospy.get_param('/driving_train/init_episode', default = 0)
LEARNING_RATE = rospy.get_param('/driving_train/learning_rate', default = 0.001)
DISCOUNT_RATE = rospy.get_param('/driving_train/discount_rate', default = 0.9)
REPLAY_MEMORY = rospy.get_param('/driving_train/replay_memory', default = 10000)
BATCH_SIZE = rospy.get_param('/driving_train/batch_size', default = 64)
TARGET_UPDATE_FREQUENCY = rospy.get_param('/driving_train/target_update_freq', default = 10)


if ENVIRONMENT == 'v0':
    INPUT_SIZE =  register.environment.v0['input_size']
    OUTPUT_SIZE = register.environment.v0['output_size']
    POLICY =      register.environment.v0['policy']
    print('Autonomous_driving training v0 is ready')
    print(register.environment.v0)

elif ENVIRONMENT == 'v1':
    INPUT_SIZE =  register.environment.v1['input_size']
    OUTPUT_SIZE = register.environment.v1['output_size']
    POLICY =      register.environment.v1['policy']
    print('Autonomous_driving training v1 is ready')
    print(register.environment.v1)

else:
    print("E: you select wrong environment. you must select ex) env:=v1 or env:=v0")
    sys.exit()
    
class training:
    def __init__(self): 
        node_name = "training2015" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        self.respawn = rospy.ServiceProxy('/model_respawn', SpawnPos)   # 모델 위치 리셋용 Model state set 서비스 요청 함수 선언
        state_sub = rospy.Subscriber("/state", State, self.state_callback, queue_size=100)
        self.pub = rospy.Publisher('/action', Int8, queue_size=10)
        self.rate = rospy.Rate(5) # 10hz -> 5Hz
        self.F = False 
        self.call_once = 1          #done 플래그는 1번만 셋되게 해줌
        self.episode = INIT_EPISODE
        
        #self.model_path = "/home/moon/model/driving.ckpt"
        self.model_path = MODEL_PATH +"/"+ENVIRONMENT+"/driving15"+ENVIRONMENT+".ckpt"
        
        
        
    def save_score(self, episode, score, cost):
        if episode == 1:
            with open(MODEL_PATH+"/result"+"/"+ENVIRONMENT+'/score.csv', 'w') as csvfile:       #새로 쓰기
                writer = csv.writer(csvfile, delimiter=',') 
                writer.writerow([episode]+[score]+[cost]) 
        else:
            with open(MODEL_PATH+"/result"+"/"+ENVIRONMENT+'/score.csv', 'a') as csvfile:       #추가
                writer = csv.writer(csvfile, delimiter=',') 
                writer.writerow([episode]+[score]+[cost]) 
    
    
    def write_batch(self, batch_buffer):
        with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+ENVIRONMENT+'batch.csv', 'w') as csvfile: 
            writer = csv.writer(csvfile, delimiter='\t') 
            state = batch_buffer[0][0]
            action = batch_buffer[0][1]
            reward = batch_buffer[0][2]
            next_state = batch_buffer[0][3]
            done = batch_buffer[0][4]
            writer.writerow([state]+[action]+[reward]+[next_state]+[done])
        
        with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+ENVIRONMENT+'batch.csv', 'a') as csvfile: 
            writer = csv.writer(csvfile, delimiter='\t') 
            for row in batch_buffer:
                state = row[0]
                action = row[1]
                reward = row[2]
                next_state = row[3]
                done = row[4]
                writer.writerow([state]+[action]+[reward]+[next_state]+[done])


    def load_batch(self):
        with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+ENVIRONMENT+'batch.csv', 'rb') as csvfile: 
            reader = csv.reader(csvfile, delimiter='\t')
            row_count = sum(1 for row in reader)
            print ("batch size is {}".format(row_count))
            
        if row_count > 1:
            with open(MODEL_PATH+'/'+ENVIRONMENT+'/'+ENVIRONMENT+'batch.csv', 'rb') as csvfile: 
                print("loading minibatch....")
                buf = deque(maxlen=REPLAY_MEMORY)
                reader = csv.reader(csvfile, delimiter='\t')
                for row in reader:
                    state = row[0]
                    #Parsing
                    state = state.replace("(","")
                    state = state.replace(")","")
                    state = state.replace("[","")
                    state = state.replace("]","")
                    state = [float(x) for x in state.split(', ')]
                    tuple(state)
                    
                    action = int(row[1])
                    
                    reward = float(row[2])
                    
                    next_state = row[3]
                    next_state = next_state.replace("(","")
                    next_state = next_state.replace(")","")
                    next_state = next_state.replace("[","")
                    next_state = next_state.replace("]","")
                    next_state = [float(x) for x in next_state.split(', ')]
                    tuple(next_state)
                    
                    done = row[4]
                    if done == 'True':
                        done = True
                    else:
                        done = False
                    
                    buf.append((state, action, reward, next_state, done))
                   
                #print(buf)
        else: 
            buf = deque(maxlen=REPLAY_MEMORY)
        return buf

    
    
    #Laser 토픽 콜백
    def state_callback(self, msg):
        self.state = msg.ranges
        self.done = msg.done
        self.F = True
            
    
    def get_copy_var_ops(self, dest_scope_name, src_scope_name):  # main과 target의 W값을 카피해줌
        """Creates TF operations that copy weights from `src_scope` to `dest_scope`
        Args:
            dest_scope_name (str): Destination weights (copy to)
            src_scope_name (str): Source weight (copy from)
        Returns:
            List[tf.Operation]: Update operations are created and returned
        """
        # Copy variables src_scope to dest_scope
        op_holder = []
    
        src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
        dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
        for src_var, dest_var in zip(src_vars, dest_vars):          # zip - 하나로 묶어주는 함수
            op_holder.append(dest_var.assign(src_var.value()))      # dest_var = src_var와 동일한 의미
    
        return op_holder


    def train_minibatch(self, mainDQN, targetDQN, train_batch):

        state_array = np.vstack([x[0] for x in train_batch])        #state 배열     [[BATCH_SIZE][INPUT_SIZE]]
        action_array = np.array([x[1] for x in train_batch])        #action 배열    [[BATCH_SIZE][OUTPUT_SIZE]]
        reward_array = np.array([x[2] for x in train_batch])        #reward 배열    [[BATCH_SIZE][1]]
        next_state_array = np.vstack([x[3] for x in train_batch])   #nstate 배열    [[BATCH_SIZE][INPUT_SIZE]]
        done_array = np.array([x[4] for x in train_batch])          #done 배열      [[BATCH_SIZE][1]]
         
        X_batch = state_array                    #state 배열     [[BATCH_SIZE][9]]
        y_batch = mainDQN.predict(state_array, 0.5)                # [[BATCH_SIZE][OUTPUT_SIZE]]
         
         
         
        Q_target = reward_array + DISCOUNT_RATE * np.max(targetDQN.predict(next_state_array, 0.5), axis=1) * ~done_array       # not done 플래그를 곱해주어 done일 시  reward를 제외한 Q의 값은 0으로 만들어준다.
         
        y_batch[np.arange(len(X_batch)), action_array] = Q_target
         
        # Train our network using target and predicted Q values on each episode
        loss, _ = mainDQN.update(X_batch, y_batch, 0.5)
        return loss
    
    def talker(self):
        # store the previous observations in replay memory
        #replay_buffer = deque(maxlen=REPLAY_MEMORY)
        replay_buffer = self.load_batch()       #파일로부터 배치 로드
        last_20_episode_reward = deque(maxlen=20)
        highest = -9999.
        loss = 10.
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, learning_rate = LEARNING_RATE, name = "main")    #DQN class 선언
            targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, learning_rate = LEARNING_RATE, name ="target")
            
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep= 5)
            sess.run(init)
            
            #------------------------------------------------------------------------------ 
            # Model load
            #------------------------------------------------------------------------------ 
            if INIT_EPISODE is not 0:
                load_epi = str(INIT_EPISODE)
                model_name = self.model_path+"-"+load_epi
                saver.restore(sess, model_name)            #저장된 데이터 불러오기
                print("Model restored from {}".format(model_name))
            #------------------------------------------------------------------------------ 
            
            copy_ops = self.get_copy_var_ops(dest_scope_name ="target", src_scope_name ="main")
            sess.run(copy_ops)
            
            print('Q-learning 2015 Traning ready')
            print('l_rate : {}, update_freq : {}, batch_size : {}'.format(LEARNING_RATE, TARGET_UPDATE_FREQUENCY, BATCH_SIZE))
            while not rospy.is_shutdown():
                while self.F:
                    self.episode += 1
                    e = 1. / ((self.episode / 10) + 1)
                    done = False
                    reward_sum = 0
                    frame_count = 0
                    
                    # The DQN traning
                    while not done:
                        frame_count += 1
                        state = self.state
                        done = self.done
                        if np.random.rand() < e:
                            action = random.randrange(0,OUTPUT_SIZE)
                        else:
                            try:
                                action = np.argmax(mainDQN.predict(state, 0.5))
                            except:
                                print('error, state is {}'.format(state))
                        if done == False:  
                            self.pub.publish(action)            #액션 값 퍼블리시
                            rospy.sleep(0.1)                    #0.1초 딜레이
                         
                        next_state = self.state
                        done = self.done
                        # Reward Policy
                        try:
                            if POLICY == 'autonomous_driving':
                                reward = policy.autonomous_driving(action, done)     #reward 리턴
                            elif POLICY == 'autonomous_driving1':
                                reward = policy.autonomous_driving1(action, done)     #reward 리턴
                        except:
                            print('there is no policy') 
                            
                        
                        replay_buffer.append((state, action, reward, next_state, done))
                        #print(replay_buffer)
                        if len(replay_buffer) > BATCH_SIZE:
                            minibatch = random.sample(replay_buffer, BATCH_SIZE)
                            loss = self.train_minibatch(mainDQN, targetDQN, minibatch)   #학습

                        if frame_count % TARGET_UPDATE_FREQUENCY == 0:               #네트워크 복사
                            sess.run(copy_ops)
                        
                        state = next_state
                        reward_sum += reward
                        print("action : {:>5}, current score : {:>5}".format(action, reward_sum))
                    
                    
    
                    print("[episode {:>5}] score was {:>5} in {:>5} frame, batch_size:{:>5}".format(self.episode, reward_sum, frame_count, len(replay_buffer)))
                    if reward_sum > highest:
                        highest = reward_sum
                    #------------------------------------------------------------------------------ 
                    #save model
                    self.save_score(self.episode, reward_sum, loss)            # score, cost 저장
                    
                    if self.episode % 30 == 0:
                        self.write_batch(replay_buffer)                     # 배치 저장
                        save_path = saver.save(sess, self.model_path, global_step=self.episode) #모델 저장
                        print("Data save in {}".format(save_path))     
                    #------------------------------------------------------------------------------ 
                    
                    #Traning complete condition
                    if reward_sum >= 1000:
                        result = 1
                    else:
                        result = 0
                    self.pub.publish(STOP)          #액션 값 퍼블리시
                    rospy.sleep(0.2)
                    self.respawn()                  #리스폰 요청은 한번만
                    rospy.wait_for_service('gazebo/reset_world')    #reset 될 때까지 대기
                    rospy.sleep(0.1)
                    self.F = False                  #플래그 언셋 후 다음 학습까지 대기
                    
                    
                    last_20_episode_reward.append(result)
                    if (len(last_20_episode_reward) == last_20_episode_reward.maxlen):            #20번 이상 시도
                        success_rate = np.mean(last_20_episode_reward) * 100
                        
                        if success_rate > 70.0:                 #20번 연속 학습의 평균 성공률이 50% 이상이면 학습 종료 후 저장
                            print("Traning Cleared within {} episodes with avg rate {}".format(episode, success_rate))
                            #save data
                            save_path = saver.save(sess, self.model_path, global_step=9999999999)
                            print("Data save in {}".format(save_path))
                            break
                        print("Success_rate : {:>5}%, Highest score : {:>5}".format(success_rate, highest))
                        


if __name__ == '__main__':
    try:
        main = training()
        main.talker()
    except rospy.ROSInterruptException:
        pass
