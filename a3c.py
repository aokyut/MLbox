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
/self.update_parameter()
/self.pull_parameter()
/self.push_parameter()

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
WORKER_NUM=5
ENV_NAME="CartPole-v0"
ADVANTAGE=3
STATE_NUM=4
ACTION_LIST=[0,1]
ACTION_NUM=2
GREEDY_EPS=0.01
GAMMA=0.99
LEARNING_RATE=0.01
RMS_DECAY=0.99
LOSS_V=0.5
LOSS_ENTROPY=0.01
HIDDEN_LAYERE=20


class brain:
    def __init__(self,name,parameter_server):
        self.name=name
        self.parameter_server=parameter_server
        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.name):
            self.input=tf.placeholder(dtype=tf.float32,shape=[None,STATE_NUM])
            hidden1=tf.layers.dense(self.input,HIDDEN_LAYERE,activation=tf.nn.leaky_relu)
            self.prob=tf.layers.dense(hidden1,ACTION_NUM,activation=tf.nn.softmax)
            self.v=tf.layers.dense(hidden1,1)

        self.reward=tf.placeholder(dtype=tf.float32,shape=(None,1))
        self.action=tf.placeholder(dtype=tf.float32,shape=(None,2))


        advantage=self.reward-self.v
        self.prob_loss=tf.reduce_sum(tf.log(self.prob*self.action+1e-10),axis=1,keepdims=True)
        self.policy_loss=-self.prob_loss*tf.stop_gradient(advantage)

        self.value_loss=tf.square(advantage)

        self.entropy=tf.reduce_sum(-self.prob*tf.log(self.prob+1e-10),axis=1,keepdims=True)

        self.loss=tf.reduce_mean(self.policy_loss+LOSS_V*self.value_loss+LOSS_ENTROPY*self.entropy)

        self.weight_param=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
        self.gradient=tf.gradients(self.loss,self.weight_param)

        #put gradients to parameter_server
        self.update_parameter_server=self.parameter_server.optimizer.apply_gradients(zip(self.gradient,self.parameter_server.weight_param))

        #get weight parameter from parameter_server
        self.pull_parameter_server=[l_p.assign(g_p) for l_p,g_p in zip(self.weight_param,self.parameter_server.weight_param)]

        #push weight parameter to parameter_server
        self.push_parameter_server=[g_p.assign(l_p) for l_p,g_p in zip(self.weight_param,self.parameter_server.weight_param)]

    def predict(self,state):
        state=np.array(state).reshape(-1,STATE_NUM)
        feed_dict={self.input:state}
        p,v=SESS.run([self.prob,self.v],feed_dict)
        return p.reshape(-1),v.reshape(-1)

    def update_parameter(self):
        feed_dict={self.input:self.s_, self.action:self.a_, self.reward:self.R}
        # print("::::::::",feed_dict,"::::::::")
        SESS.run(self.update_parameter_server,feed_dict)

    def pull_parameter(self):
        SESS.run(self.pull_parameter_server)

    def push_parameter(self):
        SESS.run(self.push_parameter_server)
    #preprocessing memory data [observation, action, R,done,next_observation],state_mask
    #make t 
    def make_train_table(self,memory):
        length=len(memory)
       
        self.s_=np.array([memory[j][0] for j in range(length)]).reshape(-1,4)
        self.a_=np.eye(2)[[memory[j][1] for j in range(length)]].reshape(-1,2)
        self.R_=np.array([memory[j][2] for j in range(length)]).reshape(-1,1)
       
        self.d_=np.array([memory[j][3] for j in range(length)]).reshape(-1,1)
        s_mask=np.array([memory[j][5] for j in range(length)]).reshape(-1,1)
        _s=np.array([memory[j][4] for j in range(length)]).reshape(-1,4)
        _, v=self.predict(_s)
        self.R=(np.where(self.d_,0,1)*v.reshape(-1,1))*s_mask+self.R_
        # print(self.R_)


class Parameter_server:
    def __init__(self):
        print("************model initialization****************")
        self.build_model()
        self.weight_param=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope="server_model")
        self.optimizer=tf.train.RMSPropOptimizer(LEARNING_RATE,RMS_DECAY)
    def build_model(self):
        with tf.variable_scope("server_model"):
            self.input=tf.placeholder(dtype=tf.float32,shape=[None,STATE_NUM])
            hidden1=tf.layers.dense(self.input,HIDDEN_LAYERE,activation=tf.nn.leaky_relu,name="server_layer_1")
            self.prov=tf.layers.dense(hidden1,ACTION_NUM,activation=tf.nn.softmax,name="server_prob")
            self.v=tf.layers.dense(hidden1,1,name="server_v")

class agent:
    def __init__(self,name,parameter_server):
        self.brain=brain(name,parameter_server)
        self.memory=[]

    #get action without random
    def action(self,state):
        prob,v = self.brain.predict(state)
        return np.random.choice(ACTION_LIST,p=prob)

    #get action with random
    def greedy_action(self,state):
        if np.random.random() <= GREEDY_EPS:
            return np.random.choice(ACTION_LIST)
        else:
            return self.action(state)

    def pull_parameter_server(self):
        self.brain.pull_parameter()

    #push observation and action, reward, done, next observation
    def push_advantage_reward(self,memory):
        R = sum([memory[j][2]*(GAMMA**j) for j in range(ADVANTAGE+1)])
        self.memory.append([memory[0][0],memory[0][1],R,memory[0][3],memory[0][4],GAMMA**ADVANTAGE])

        for i in range(1,len(memory)-ADVANTAGE):
            R = ((R-memory[i-1][2])/GAMMA) + memory[i+ADVANTAGE][2]*(GAMMA**(ADVANTAGE-1))
            self.memory.append([memory[i][0],memory[i][1],R,memory[i+ADVANTAGE][3],memory[i][4],GAMMA**ADVANTAGE])
            
        for i in range(ADVANTAGE):
            R=((R-memory[len(memory)-ADVANTAGE+i][2])/GAMMA)
            self.memory.append([memory[i][0],memory[i][1],R,True,memory[i][4],GAMMA**(ADVANTAGE-i)])
        self.brain.make_train_table(self.memory)
        # log=[memory[j][2] for j in range(len(memory))]
        # print(log)
        self.memory=[]

    def finish_leaning(self):
        self.brain.push_parameter()
    
    def train(self):
        self.brain.update_parameter()


class Worker:
    def __init__(self,thread_type,thread_name,parameter_server):
        self.thread_type=thread_type
        self.name=thread_name
        self.agent=agent(thread_name,parameter_server)
        self.parameter_server=parameter_server
        self.env=gym.make(ENV_NAME)
        self.leaning_memory=np.zeros(10)
        self.memory=[]
        self.total_trial=0

    def run_thread(self):
        while True:
            if self.thread_type=="train" and not isLearned:
                self.env_run()
            elif self.thread_type=="train" and isLearned:
                sleep(3)
            elif self.thread_type=="test" and not isLearned:
                sleep(3)
            elif self.thread_type=="test" and isLearned:
                self.env_run()

    def env_run(self):
        self.total_trial+=1
        step=0
        observation=self.env.reset()
        global isLearned
        global frame
        self.agent.pull_parameter_server()

        # if self.total_trial%100==0:
        #     print(SESS.run(self.agent.brain.weight_param))

        while True:
            step+=1
            frame+=1
            if self.thread_type=="train":
                action=self.agent.greedy_action(observation)
            elif self.thread_type=="test":
                self.env.render()
                action=self.agent.action(observation)
                sleep(0.1)
            
            next_observation,_,done,_=self.env.step(action)

            reward=0
            if done:
                if step>=199:
                    reward=1
                else:
                    reward=-1
            else:
                #when not falling
                reward+=0
            
            self.memory.append([observation,action,reward,done,next_observation])

            observation=next_observation

            if done:
                break

        self.leaning_memory=np.hstack((self.leaning_memory[1:],step))
        print("Thread:",self.name," Thread_trials:",self.total_trial," score:",step," mean_score:",self.leaning_memory.mean()," total_step:",frame)
        if self.leaning_memory.mean()>=199:
            isLearned=True
            sleep(3)
            self.agent.finish_leaning()
        else:
            self.agent.push_advantage_reward(self.memory)
            self.agent.train()
            self.memory=[]


def main():
    #make thread
    with tf.device("/cpu:0"):
        parameter_server=Parameter_server()
        thread=[]
        for i in range(WORKER_NUM):
            thread_name="local_thread"+str(i)
            thread.append(Worker(thread_type="train",thread_name=thread_name,parameter_server=parameter_server))
        thread.append(Worker(thread_type="test",thread_name="test_thread",parameter_server=parameter_server))
    
    COORD=tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    for worker in thread:
        job=lambda: worker.run_thread()
        t=threading.Thread(target=job)
        t.start()

if __name__=="__main__":
    SESS=tf.Session()
    frame=0
    isLearned=False
    main()

print("end")