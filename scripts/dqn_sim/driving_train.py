#!/usr/bin/env python
#-*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import random
import dqn
import rospy
import dqn

from collections import deque
from IRL_learning_ros.msg import State
from std_msgs.msg import Int8
from argparse import Action



INPUT_SIZE = 9                       # [-60, -45, -30, -15, 0, 15, 30, 45, 60] 
OUTPUT_SIZE = 3                      # [전진, 좌회전, 우회전]

DISCOUNT_RATE = 0.9
REPLAY_MEMORY = 10000
MAX_EPISODE = 5000
BATCH_SIZE = 64

class state_pub:
    def __init__(self): 
        node_name = "state_pub" 
        rospy.init_node(node_name)
        #ros topic 구독자 설정 및 콜백함수 정의
        state_sub = rospy.Subscriber("/state", State, self.state_callback, queue_size=100)
        self.pub = rospy.Publisher('/action', Int8, queue_size=10)
        self.rate = rospy.Rate(10) # 10hz
        self.F = False 
        
        
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
        last_10_game_reward = deque(maxlen=10)
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
            init = tf.global_variables_initializer()
            sess.run(init)
            print('Traning ready')
            while not rospy.is_shutdown():
                if self.F is True:
                    
                    for episode in range(MAX_EPISODE):
                        e = 1. / ((episode / 10) + 1)
                        done = False
                        reward_sum = 0
                        frame_count = 0
                        
                        # The DQN traning
                        while not done:
                            frame_count += 1
                            state = self.state
                            if np.random.rand() < 0:
                                action = random.randrange(0,OUTPUT_SIZE)
                            else:
                                try:
                                    action = np.argmax(mainDQN.predict(state))
                                except:
                                    print('error, state is {}'.format(state))
                                    
                            self.pub.publish(action)            #액션 값 퍼블리시
                            self.rate.sleep()           #ROS sleep
                            
                            next_state = self.state
                            done = self.done
                            # Reward Policy
                            if action == 0:
                                reward = 5
                            elif (action == 1) or (action == 2):
                                reward = 1
                            if done:            # 충돌  
                                reward = -200

                            replay_buffer.append((state, action, reward, next_state, done))
                            #print(replay_buffer)
                            state = next_state
                            reward_sum += reward
                            if len(replay_buffer) > BATCH_SIZE:
                                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                                loss = self.train_minibatch(mainDQN, minibatch)     #학습 시작
                            print("action : {:>5}, current reward : {:>5}".format(action, reward_sum))
                        print("[episode {:>5}] Reward was {:>5} in {:>5} frame:".format(episode, reward_sum, frame_count))
            
            
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass