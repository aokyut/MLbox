import tensorflow as tf
import numpy as np
import random
import threading
import gym
from time import sleep

"""
今回のA3Cの実装

brain
/self.name
/self.model
/self.graph
/self.parameter_server
/self.build_model()
/self.build_graph()
/self.update_parameter_server()
/self.pull_parameter_server()
/self.push_parameter_server()

parameter_server
/self.model
/self.build_model()

agent
/self.name
/self.brain
/self.parameter_server
/self.action()
/self.greedy_action()
/self.push_advantage_reward()
/self.finish_leaning()

worker:  
/self.thread_type
/self.agent
/self.env
/self.parameter_server
/self.name
/self.run_thread()
/self.env_run()

main: mainroutin
"""
#define constants
WORKER_NUM=8
ENV_NAME="CartPole-v0"
ADVANTAGE=2



class brain:
    def __init__(self):
    def build_model(self):
    def build_graph(self):
    def update_parameter_server(self):
    def pull_parameter_server(self):
    def push_parameter_server(self):

class parameter_server:
    def __init__(self):
    def build_model(self):

class agent:
    def __init__(self):
    #get action without random
    def action(self,state):
    #get action with random
    def greedy_action(self,state):
    #push observation and action, reward, done, next observation
    def push_advantage_reward(self):
    def finish_leaning(self):

class worker:
    def __init__(self,thread_type,thread_name,parameter_server):
        self.thread_type=thread_type
        self.name=thread_name
        self.agent=agent(thread_name,parameter_server)
        self.parameter_server=parameter_server
        self.env=gym.make(ENV_NAME)
        self.leaning_memory=np.zeros(10)
        self.memory=[]

    def run_thread(self):
        self.env_run()

    def env_run(self):
        step=0
        observation=self.env.reset()
        global isLearned
        global frame

        while True:
            step+=1
            frame+=1
            if self.thread_type=="train":
                action=self.agent.greedy_action(observation)
            elif self.thread_type="test":
                self.env.render()
                action=self.agent.action(observation)
            
            next_observation,_,done,_=self.env.step(action)

            if done:
                if step>=199:
                    reward=1
                elif:
                    reward=-1
                reward+=0
            
            memory.append([observation,action,reward,done,next_observation])

        self.leaning_memory=np.hstack((self.leaning_memory[1:],step))

        if self.leaning_memory.mean()>=199:
            isLearned=True
            sleep(3)
            self.agent.finish_leaning()
        else:
            self.agent.push_advantage_reward(self.memory)


            
            

    
def main():
    frame=0
    isLearned=False
    sess=tf.Session()

    #meke thread
    with tf.device("/cpu:0"):
        parameter_server=parameter_server()
        thread=[]
        for i in range(WORKER_NUM):
            thread_name="local_thread"+str(i)
            thread.append(worker(thread_type="train",thread_name=thread_name,parameter_server=parameter_server))
        
        thread.append(worker(thread_type="test",thread_name="test_thread",parameter_serverr=parameter_server))
    
    coord=tf.train.Coordinator()
    sess.run(tf.global_variables_initializer)

    for worker in thread:
        job=lambda: worker.run_thread()
        t=threading.Thread(target=job)
        t.start()

main()
    