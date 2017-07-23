#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import rospy
import dqn
import policy
import register

from collections import deque

from IRL_learning_ros.srv import SpawnPos
from IRL_learning_ros.msg import State
from std_msgs.msg import Int8
from argparse import Action

STOP = 99
#Hyper parameter

ENVIRONMENT = rospy.get_param('/driving_train/environment', 'v0')
MODEL_PATH = rospy.get_param('/driving_train/model_path', default = 'model')
INIT_EPISODE = rospy.get_param('/driving_train/init_episode', default = 0)
MAX_EPISODE = rospy.get_param('/driving_train/max_episode', default = 5000)
DISCOUNT_RATE = rospy.get_param('/driving_train/discount_rate', default = 0.9)
REPLAY_MEMORY = rospy.get_param('/driving_train/replay_memory', default = 10000)
BATCH_SIZE = rospy.get_param('/driving_train/batch_size', default = 64)


if ENVIRONMENT == 'v0':
    INPUT_SIZE =  register.environment.v0['input_size']
    OUTPUT_SIZE = register.environment.v0['output_size']
    POLICY =      register.environment.v0['policy']
    print('Autonomous_driving training is ready')
    print(register.environment.v0)


class training:
    def __init__(self): 
        node_name = "training" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        self.respawn = rospy.ServiceProxy('/model_respawn', SpawnPos)   # 모델 위치 리셋용 Model state set 서비스 요청 함수 선언
        state_sub = rospy.Subscriber("/state", State, self.state_callback, queue_size=100)
        self.pub = rospy.Publisher('/action', Int8, queue_size=10)
        self.rate = rospy.Rate(5) # 10hz -> 5Hz
        self.F = False 
        #self.model_path = "/home/moon/model/driving.ckpt"
        self.model_path = MODEL_PATH + "/driving.ckpt"
    #Laser 토픽 콜백
    def state_callback(self, msg):
        self.state = msg.ranges
        self.done = msg.done
        self.F = True
        
    def bot_play(mainDQN):
        """Runs a single episode with rendering and prints a reward
        Args:
            mainDQN (dqn.DQN): DQN Agent
        """
        state = self.ranges
        total_reward = 0

        while True:
            env.render()
            action = np.argmax(mainDQN.predict(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print("Total score: {}".format(total_reward))
                break



    def train_minibatch(self, DQN, train_batch):

        state_array = np.vstack([x[0] for x in train_batch])        #state 배열     [[64][input_size]]
        action_array = np.array([x[1] for x in train_batch])        #action 배열    [[64][output_size]]
        reward_array = np.array([x[2] for x in train_batch])        #reward 배열    [[64][1]]
        next_state_array = np.vstack([x[3] for x in train_batch])   #nstate 배열    [[64][input_size]]
        done_array = np.array([x[4] for x in train_batch])          #done 배열      [[64][1]]
        
        X_batch = state_array               #state 배열     [[64][9]]
        y_batch = DQN.predict(state_array)
        
        
        
        Q_target = reward_array + DISCOUNT_RATE * np.max(DQN.predict(next_state_array), axis=1) * ~done_array       # not done 플래그를 곱해주어 done일 시  reward를 제외한 Q의 값은 0으로 만들어준다.
        y_batch[np.arange(len(X_batch)), action_array] = Q_target
        
        # Train our network using target and predicted Q values on each episode
        loss, _ = DQN.update(X_batch, y_batch)
        return loss

    
    
    def talker(self):
        # store the previous observations in replay memory
        replay_buffer = deque(maxlen=REPLAY_MEMORY)
        last_20_episode_reward = deque(maxlen=20)
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)    #DQN class 선언
            mypolicy = policy.policy()     #policy class 선언
                
                
            init = tf.global_variables_initializer()
            saver = tf.train.Saver(max_to_keep= 10)
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
            
            print('Traning ready')
            while not rospy.is_shutdown():
                if self.F is True:
                    
                    for episode in range(INIT_EPISODE, INIT_EPISODE + MAX_EPISODE+1):
                        e = 1. / ((episode / 10) + 1)
                        done = False
                        reward_sum = 0
                        frame_count = 0
                        
                        # The DQN traning
                        while not done:
                            frame_count += 1
                            state = self.state
                            if np.random.rand() < e:
                                action = random.randrange(0,OUTPUT_SIZE)
                            else:
                                try:
                                    action = np.argmax(mainDQN.predict(state))
                                except:
                                    print('error, state is {}'.format(state))
                                     
                            self.pub.publish(action)            #액션 값 퍼블리시
                            rospy.sleep(0.1)                    #0.1초 딜레이
                             
                            next_state = self.state
                            done = self.done
                                      
                            # Reward Policy
                            try:
                                if POLICY == 'autonomous_driving':
                                    reward = mypolicy.autonomous_driving(action, done)     #reward 리턴
                            except:
                                print('there is no policy') 
                                
                            # if 충돌 시 종료 구문
                            if done:                    
                                self.pub.publish(STOP)            #액션 값 퍼블리시
                                self.respawn()
                                     
                            replay_buffer.append((state, action, reward, next_state, done))
                             
                            state = next_state
                            reward_sum += reward
                            if len(replay_buffer) > BATCH_SIZE:
                                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                                loss = self.train_minibatch(mainDQN, minibatch)     #학습 시작
                                
                            print("action : {:>5}, current reward : {:>5}".format(action, reward_sum))
                        print("[episode {:>5}] Reward was {:>5} in {:>5} frame:".format(episode, reward_sum, frame_count))
                        
                        #save data
                        if episode % 30 == 0:
                            save_path = saver.save(sess, self.model_path, global_step=episode)
                            print("Data save in {}".format(save_path))     
                        
                        #Traning complete condition
                        last_20_episode_reward.append(reward_sum)
                        if len(last_20_episode_reward) == last_20_episode_reward.maxlen:
                            avg_reward = np.mean(last_20_episode_reward)
                            if avg_reward > 1000.0:                 #20번 연속 학습의 평균 스코어가 1000점 이상이면 학습 종료 후 저장
                                print("Traning Cleared within {} episodes with avg reward {}".format(episode, avg_reward))
                                #save data
                                
                                break
       
if __name__ == '__main__':
    try:
        main = training()
        main.talker()
    except rospy.ROSInterruptException:
        pass