#!/usr/bin/env python
#-*- coding: utf-8 -*-

import rospy
import dqn
from IRL_learning_ros.msg import State, Action



INPUT_SIZE = 11                      # [-60, -45, -30, -15, 0, 15, 30, 45, 60] 
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
        self.pub = rospy.Publisher('/action', Action, queue_size=10)
        self.rate = rospy.Rate(10) # 10hz
        
        
        
        
    #Laser 토픽 콜백
    def state_callback(self, msg):
        self.state = msg.state
        
    def bot_play(mainDQN):
        """Runs a single episode with rendering and prints a reward
        Args:
            mainDQN (dqn.DQN): DQN Agent
        """
        state = env.reset()
        total_reward = 0

        while True:
            env.render()
            action = np.argmax(mainDQN.predict(state))
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                print("Total score: {}".format(total_reward))
                break



    def train_minibatch(DQN, train_batch):
        """Prepare X_batch, y_batch and train them
        Recall our loss function is
            target = reward + discount * max Q(s',a)
                     or reward if done early
            Loss function: [target - Q(s, a)]^2
        Hence,
            X_batch is a state list
            y_batch is reward + discount * max Q
                       or reward if terminated early
        Args:
            DQN (dqn.DQN): DQN Agent to train & run
            train_batch (list): Minibatch of Replay memory
                Eeach element is a tuple of (s, a, r, s', done)
        Returns:
            loss: Returns a loss
        """
        state_array = np.vstack([x[0] for x in train_batch])
        action_array = np.array([x[1] for x in train_batch])
        reward_array = np.array([x[2] for x in train_batch])
        next_state_array = np.vstack([x[3] for x in train_batch])
        done_array = np.array([x[4] for x in train_batch])
    
        X_batch = state_array
        y_batch = DQN.predict(state_array)
    
        Q_target = reward_array + DISCOUNT_RATE * np.max(DQN.predict(next_state_array), axis=1) * ~done_array       # not done 플래그를 곱해주어 done일 시  reward를 제외한 Q의 값은 0으로 만들어준다.
        y_batch[np.arange(len(X_batch)), action_array] = Q_target
    
        # Train our network using target and predicted Q values on each episode
        loss, _ = DQN.update(X_batch, y_batch)
    
        return loss
    
    
    def main():
        # store the previous observations in replay memory
        replay_buffer = deque(maxlen=REPLAY_MEMORY)
        last_10_game_reward = deque(maxlen=10)
    
        with tf.Session() as sess:
            mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE)
            init = tf.global_variables_initializer()
            sess.run(init)
    
            for episode in range(MAX_EPISODE):
                e = 1. / ((episode / 10) + 1)
                done = False
                state = env.reset()
                reward_sum = 0
                frame_count = 0
                
                # The DQN traning
                while not done:
                    frame_count += 1
                    if np.random.rand() < e:
                        action = env.action_space.sample()
                    else:
                        action = np.argmax(mainDQN.predict(state))
    
                    next_state, reward, done, _ = env.step(action)
                    #Set frame limit 
                    
                    if done: 
                        if frame_count == 1000:
                            reward = -100
                        else:
                            pass
        
                    replay_buffer.append((state, action, reward, next_state, done))
    
                    state = next_state
                    
                        
                    if len(replay_buffer) > BATCH_SIZE:
                        minibatch = random.sample(replay_buffer, BATCH_SIZE)
                        train_minibatch(mainDQN, minibatch)
                    
                    reward_sum += reward
                self.rate.sleep()           #ROS sleep
                    
                print("[episode {:>5}] Reward was {:>5} in {:>5} frame:".format(episode, reward_sum, frame_count))
                # LunarLander-v2 Game Clear Logic
                last_10_game_reward.append(reward_sum)
                if len(last_10_game_reward) == last_10_game_reward.maxlen:
                    avg_reward = np.mean(last_10_game_reward)
                    if avg_reward > 200.0:
                        print("Game Cleared within {} episodes with avg reward {}".format(episode, avg_reward))
                        break
            
            
if __name__ == '__main__':
    try:
        main = state_pub()
        main.talker()
    except rospy.ROSInterruptException:
        pass