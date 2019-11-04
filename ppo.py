import argparse
import tensorflow as tf
import numpy as np
import random
import threading
import gym
from time import sleep
from gym import wrappers
from os import path

parser=argparse.ArgumentParser(description="Reiforcement training with PPO",add_help=True)
parser.add_argument("--model",type=str,required=True,help="model base name. required")
parser.add_argument("--env_name",default="CartPole-v0",help="environment name. default is CartPole-v0")
parser.add_argument("--save",action="store_true",default=False,help="save command")
parser.add_argument("--load",action="store_true",default=False,help="load command")
parser.add_argument("--thread_num",type=int,default=5)
parser.add_argument("--video",action="store_true",default=False, help="write this if you want to save as video")
args=parser.parse_args()


ENV_NAME=args.env_name
WORKER_NUM=args.thread_num
#define constants
VIDEO_DIR="./train_info/video"
MODEL_DIR="./train_info/models"
MODEL_SAVE_PATH=path.join(MODEL_DIR,args.model)

ADVANTAGE=1
STATE_NUM=4
ACTION_LIST=[0,1]
ACTION_NUM=2
#epsiron parameter
EPS_START = 0.5
EPS_END = 0.2
EPS_STEPS = 200 * WORKER_NUM**2
#learning parameter
GAMMA=0.99
LEARNING_RATE=0.002
#loss constants
LOSS_V=0.5
LOSS_ENTROPY=0.02
HIDDEN_LAYERE=30

EPSIRON = 0.2


def huber_loss(advantage, delta=0.5):
    error = advantage
    cond = tf.abs(error) < delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(cond, squared_loss, linear_loss)


class ppo_brain:
    def __init__(self):
        self.build_model()
        self.name="brain"
        self.prob_old=1.0

    def build_model(self):
        self.input=tf.placeholder(dtype=tf.float32,shape=[None,STATE_NUM])
        with tf.variable_scope("current_brain"):
            hidden1=tf.layers.dense(self.input,HIDDEN_LAYERE,activation=tf.nn.leaky_relu)
            self.prob=tf.layers.dense(hidden1,ACTION_NUM,activation=tf.nn.softmax)
            self.v=tf.layers.dense(hidden1,1)
        with tf.variable_scope("old_brain"):
            old_hidden1=tf.layers.dense(self.input,HIDDEN_LAYERE,activation=tf.nn.leaky_relu)
            self.old_prob=tf.layers.dense(old_hidden1,ACTION_NUM,activation=tf.nn.softmax)
            self.old_v=tf.layers.dense(old_hidden1,1)

        self.reward=tf.placeholder(dtype=tf.float32,shape=(None,1))
        self.action=tf.placeholder(dtype=tf.float32,shape=(None,ACTION_NUM))

        #define loss function
        advantage = self.reward-self.v

        #define policy loss
        r_theta = tf.div(self.prob + 1e-10, tf.stop_gradient(self.old_prob) + 1e-10)
        action_theta = tf.reduce_sum(tf.multiply(r_theta, self.action), axis=1, keepdims=True)
        r_clip = tf.clip_by_value(action_theta,1-EPSIRON,1+EPSIRON)
        advantage_cpi = tf.multiply(action_theta , tf.stop_gradient(advantage))
        advantage_clip = tf.multiply(r_clip , tf.stop_gradient(advantage))
        self.policy_loss = tf.minimum(advantage_clip , advantage_cpi)

        #define value loss
        self.value_loss=tf.square(advantage)

        #define entropy
        self.entropy=tf.reduce_sum(self.prob*tf.log(self.prob+1e-10),axis=1,keepdims=True)

        #output loss function
        self.loss=tf.reduce_sum(-self.policy_loss+LOSS_V*self.value_loss-LOSS_ENTROPY*self.entropy)

        #update parameters
        self.opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        self.minimize = self.opt.minimize(self.loss)

        #get trainable parameters
        self.weight_param = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES, scope="current_brain")
        self.old_weight_param = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES, scope="old_brain")

        #new prarameter assign to old parameter 
        self.insert = [g_p.assign(l_p) for l_p,g_p in zip(self.weight_param,self.old_weight_param)]


    def predict(self,state):
        state=np.array(state).reshape(-1,STATE_NUM)
        feed_dict={self.input:state}
        p,v=SESS.run([self.prob,self.v],feed_dict)
        return p.reshape(-1),v.reshape(-1)


    #preprocessing memory data [observation, action, R,done,next_observation],state_mask
    #make t 
    def update(self,memory):
        length=len(memory)
       
        s_=np.array([memory[j][0] for j in range(length)]).reshape(-1,STATE_NUM)
        a_=np.eye(ACTION_NUM)[[memory[j][1] for j in range(length)]].reshape(-1,ACTION_NUM)
        R_=np.array([memory[j][2] for j in range(length)]).reshape(-1,1)
       
        d_=np.array([memory[j][3] for j in range(length)]).reshape(-1,1)
        s_mask=np.array([memory[j][5] for j in range(length)]).reshape(-1,1)
        _s=np.array([memory[j][4] for j in range(length)]).reshape(-1,STATE_NUM)
        _, v=self.predict(_s)
        R=(np.where(d_,0,1)*v.reshape(-1,1))*s_mask+R_
        #update params
        feed_dict={self.input:s_, self.action:a_, self.reward:R}
        SESS.run(self.insert)
        SESS.run(self.minimize,feed_dict)



class ppo_agent:
    def __init__(self,brain):
        self.brain=brain
        self.memory=[]


    #get action without random
    def action(self,state):
        prob,v = self.brain.predict(state)
        return np.random.choice(ACTION_LIST,p=prob)


    #get action with random
    def greedy_action(self,state):
        if frame >= EPS_STEPS:  
            eps = EPS_END
        else:
            eps = EPS_START + frame* (EPS_END - EPS_START) / EPS_STEPS  

        if np.random.random() <= eps:
            return np.random.choice(ACTION_LIST)
        else:
            return self.action(state)


    #push observation and action, reward, done, next observation
    def update(self,memory):
        R = sum([memory[j][2]*(GAMMA**j) for j in range(ADVANTAGE+1)])
        self.memory.append([memory[0][0],memory[0][1],R,memory[0][3],memory[0][4],GAMMA**ADVANTAGE])

        for i in range(1,len(memory)-ADVANTAGE):
            R = ((R-memory[i-1][2])/GAMMA) + memory[i+ADVANTAGE][2]*(GAMMA**(ADVANTAGE-1))
            self.memory.append([memory[i][0],memory[i][1],R,memory[i+ADVANTAGE][3],memory[i][4],GAMMA**ADVANTAGE])
            
        for i in range(ADVANTAGE-1):
            R=((R-memory[len(memory)-ADVANTAGE+i][2])/GAMMA)
            self.memory.append([memory[i][0],memory[i][1],R,True,memory[i][4],GAMMA**(ADVANTAGE-i)])
        self.brain.update(self.memory)
        # log=[memory[j][2] for j in range(len(memory))]
        # print(log)
        self.memory=[]



class Worker:
    def __init__(self,thread_type,thread_name,brain):
        self.thread_type=thread_type
        self.name=thread_name
        self.agent=ppo_agent(brain)
        self.env=gym.make(ENV_NAME)
        if self.thread_type=="test" and args.video:
            self.env=wrappers.Monitor(self.env, VIDEO_DIR, force=True)
        self.leaning_memory=np.zeros(10)
        self.memory=[]
        self.total_trial=0
        self.test_count=1


    def run_thread(self):
        while True:
            if self.thread_type=="train" and not isLearned:
                self.env_run()
            elif self.thread_type=="train" and isLearned:
                sleep(3)
                break
            elif self.thread_type=="test" and not isLearned:
                sleep(3)
            elif self.thread_type=="test" and isLearned:
                self.env_run()
                break


    def env_run(self):
        global isLearned
        global frame
        self.total_trial+=1

        step=0
        observation=self.env.reset()
        # if self.total_trial%100==0:
        #     print(SESS.run(self.agent.brain.weight_param))

        while True:
            step+=1
            frame+=1
            if self.thread_type=="train":
                action=self.agent.greedy_action(observation)
            elif self.thread_type=="test":
                self.env.render()
                sleep(0.01)
                action=self.agent.action(observation)
            
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
            pass
        else:
            self.agent.update(self.memory)
            self.memory=[]
        


def main(args):
    #make thread
    with tf.device("/cpu:0"):
        brain=ppo_brain()
        thread=[]
        for i in range(WORKER_NUM):
            thread_name="local_thread"+str(i)
            thread.append(Worker(thread_type="train",thread_name=thread_name,brain=brain))
    
    COORD=tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    
    if args.load:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        if ckpt:
            saver.restore(SESS,MODEL_SAVE_PATH)

    runnning_thread=[]
    for worker in thread:
        job=lambda: worker.run_thread()
        t=threading.Thread(target=job)
        t.start()
        runnning_thread.append(t)
    COORD.join(runnning_thread)
    test=Worker(thread_type="test",thread_name="test_thread",brain=brain)
    test.run_thread()
    if args.save:
        saver.save(SESS,MODEL_SAVE_PATH)
        print("saved")

if __name__=="__main__":


    SESS=tf.Session()

    frame=0
    isLearned=False
    
    main(args)

print("end")