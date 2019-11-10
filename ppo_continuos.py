"""
今回の実装
ppoを用いて連続な行動空間において強化学習を行うモデル
変更部分
ネットワーク部分の変更（連続行動空間に適応する）
ppo_brain
/build_model:　連続行動空間用に確率分布を用いて推論を行うように変更
ppo_agent
/action:　ppo_brainから推論された確率分布から行動を出力
/greedy_action:　一定確率で一様分布から行動を出力

入出力部分の変更
入力２８次元、出力８次元となるようにする
出力については
分布の平均（8次元）、分散（1次元）、状態価値（1次元）
行動は（−１、＋１）

今回用いるの環境用の変更("RoboschoolAnt-v1")
Worker 
/env_run: 報酬の与え方が変わるのでその設定を行う
1000ステップで環境をリセット
100ステップごとに学習を行う

学習全体の変更
定期的にsaveを行うようにする
--show引数を与えるようにする
logとしてlossを出力
"""
import argparse
import tensorflow as tf
import numpy as np
import random
import threading
import gym
import pybullet_envs
from time import sleep
from gym import wrappers
from os import path
from copy import deepcopy

import pickle

parser=argparse.ArgumentParser(description="Reiforcement training with PPO",add_help=True)
parser.add_argument("--model",type=str,required=True,help="model base name. required")
parser.add_argument("--env_name",default="HopperBulletEnv-v0",help="environment name. default is AntBulletEnv-v0")
parser.add_argument("--save",action="store_true",default=False,help="save command")
parser.add_argument("--load",action="store_true",default=False,help="load command")
parser.add_argument("--show",action="store_true",default=False,help="render environment")
parser.add_argument("--thread_num",type=int,default=5)
parser.add_argument("--video",action="store_true",default=False, help="write this if you want to save as video")
args=parser.parse_args()


ENV_NAME=args.env_name
WORKER_NUM=args.thread_num
#define constants
VIDEO_DIR="./train_info/video"
MODEL_DIR="./train_info/models"
SUMMARY_DIR="./logs"
SUMMARY_PATH=path.join(SUMMARY_DIR, args.model)
MODEL_SAVE_PATH=path.join(MODEL_DIR,args.model)

ADVANTAGE=2
STATE_NUM=15
ACTION_LIST=[0,1]
ACTION_NUM=3
#epsiron parameter
EPS_START = 0.5
EPS_END = 0.2
EPS_STEPS = 1000 * WORKER_NUM**2
#learning parameter
GAMMA=0.98
LEARNING_RATE=2e-5
DROPOUT_RATE=0.5
MAX_BUFFER = 10000
#loss constants
LOSS_V=0.5
LOSS_ENTROPY=0.02
HIDDEN_LAYERE=256
LAYERE2=8
LAYERE3=4

EPSIRON = 0.2


#define some of functions
def huber_loss(advantage, delta=0.5):
    with tf.name_scope("huber_loss"):
        error = advantage
        cond = tf.abs(error) < delta

        squared_loss = 0.5 * tf.square(error)
        linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(cond, squared_loss, linear_loss)

def random(min,max,size):
    return np.random.uniform(size=size)*(max-min)+min


class ppo_brain:
    def __init__(self):
        self.build_model()
        self.name="brain"
        self.prob_old=1.0
        self.train_num = 0
        self.memory=[[], [], []]

    def build_model(self):
        self.input = tf.placeholder(dtype=tf.float32,shape=[None,STATE_NUM],name="input")

        #define two network
        with tf.variable_scope("current_brain"):
            hidden1 = tf.layers.dense(self.input,HIDDEN_LAYERE,activation = tf.nn.leaky_relu)
            self.mu = tf.layers.dense(hidden1,ACTION_NUM,activation = tf.nn.tanh)
            self.sig = tf.layers.dense(hidden1, 1, activation = tf.nn.softplus)
            self.v=tf.layers.dense(hidden1,1)
        with tf.variable_scope("old_brain"):
            old_hidden1 = tf.layers.dense(self.input,HIDDEN_LAYERE,activation = tf.nn.leaky_relu)
            self.old_mu = tf.layers.dense(old_hidden1,ACTION_NUM,activation = tf.nn.tanh)
            self.old_sig = tf.layers.dense(old_hidden1, 1, activation = tf.nn.softplus)
            self.old_v=tf.layers.dense(old_hidden1,1)
        with tf.variable_scope("store_brain"):
            store_hidden1 = tf.layers.dense(self.input,HIDDEN_LAYERE,activation = tf.nn.leaky_relu)
            self.store_mu = tf.layers.dense(store_hidden1,ACTION_NUM,activation = tf.nn.tanh)
            self.store_sig = tf.layers.dense(store_hidden1,1,activation = tf.nn.softplus)
            self.store_v = tf.layers.dense(store_hidden1,1)

        with tf.name_scope("input_train"):
            self.reward=tf.placeholder(dtype=tf.float32,shape=(None,1))
            self.action=tf.placeholder(dtype=tf.float32,shape=(None,ACTION_NUM))

        #define loss function
        with tf.name_scope("advantage"):
            advantage = self.reward-self.v  

        #distributions
        with tf.name_scope("distrobutions"):
            action_dist_new = tf.distributions.Normal(loc = self.mu, scale = self.sig, allow_nan_stats = False)
            action_dist_old = tf.distributions.Normal(self.old_mu, self.old_sig, allow_nan_stats = False)

        #define policy loss
        with tf.name_scope("probs"):
            self.prob_new = action_dist_new.prob(self.action) + 1e-10
            self.prob_old = action_dist_old.prob(self.action) + 1e-10
        with tf.name_scope("r_theta"):
            r_theta = tf.div(self.prob_new , tf.stop_gradient(self.prob_old))
            action_theta = tf.reduce_mean(r_theta, axis=1, keepdims=True)
        with tf.name_scope("r_clip"):
            r_clip = tf.clip_by_value(action_theta,1-EPSIRON,1+EPSIRON)
        with tf.name_scope("CPI"):
            advantage_cpi = tf.multiply(action_theta , tf.stop_gradient(advantage))
        with tf.name_scope("CLIP"):
            advantage_clip = tf.multiply(r_clip , tf.stop_gradient(advantage))
        with tf.name_scope("policy_loss"):
            self.policy_loss = tf.minimum(advantage_clip , advantage_cpi)
            # self.policy_loss = advantage_clip

        #define value loss
        with tf.name_scope("value_loss"):
            self.value_loss = tf.square(advantage)

        #define entropy
        with tf.name_scope("entropy"):
            self.entropy = action_dist_new.entropy()

        #output loss function
        with tf.name_scope("loss"):
            self.loss=tf.reduce_mean(-self.policy_loss+LOSS_V*self.value_loss+LOSS_ENTROPY*self.entropy)

        #update parameters
        with tf.name_scope("optimizer"):
            self.opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
            self.minimize = self.opt.minimize(self.loss)

        #get trainable parameters
        with tf.name_scope("get_prams"):
            self.weight_param = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES, scope="current_brain")
            self.old_weight_param = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES, scope="old_brain")
            self.store_weight_param = tf.get_collection(key = tf.GraphKeys.TRAINABLE_VARIABLES, scope="store_brain")

        #new prarameter assign to old parameter 
        with tf.name_scope("insert"):
            self.store = [g_p.assign(l_p) for l_p,g_p in zip(self.weight_param,self.store_weight_param)]
            self.insert = [g_p.assign(l_p) for l_p,g_p in zip(self.store_weight_param,self.old_weight_param)]

        with tf.name_scope('summary'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar("entropy",tf.reduce_mean(self.entropy))
            tf.summary.scalar("advantage",tf.reduce_mean(tf.abs(advantage)))
            tf.summary.scalar("r_theta",tf.reduce_mean(r_theta))
            tf.summary.scalar("r_clip",tf.reduce_mean(r_clip))
            tf.summary.scalar("CPI", tf.reduce_mean(advantage_cpi))
            tf.summary.scalar("CLIP", tf.reduce_mean(tf.abs(advantage_clip)))
            tf.summary.scalar("policy_loss", tf.reduce_mean(self.policy_loss))
            tf.summary.scalar("value_loss", tf.reduce_mean(self.value_loss))
            tf.summary.scalar("sigma", tf.reduce_mean(self.sig))
            tf.summary.scalar("mu", tf.reduce_mean(self.mu))
            self.merged = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter(SUMMARY_PATH, SESS.graph)

    def predict(self,state):
        with tf.name_scope("predict"):
            state=np.array(state).reshape(-1,STATE_NUM)
            feed_dict={self.input:state}
            mu,sig,v=SESS.run([self.mu,self.sig,self.v],feed_dict)
        return mu.reshape(-1,ACTION_NUM),mu.reshape(-1),v.reshape(-1)


    #preprocessing memory data [observation, action, R,done,next_observation],state_mask
    #make t 
    def update(self,memory):
        length=len(memory)
        self.train_num += 1
       
        s_ = np.array([memory[j][0] for j in range(length)]).reshape(-1,STATE_NUM)
        a_ = np.vstack([[memory[j][1] for j in range(length)]])
        R_ = np.array([memory[j][2] for j in range(length)]).reshape(-1,1)
       
        d_ = np.array([memory[j][3] for j in range(length)]).reshape(-1,1)
        s_mask = np.array([memory[j][5] for j in range(length)]).reshape(-1,1)
        _s = np.array([memory[j][4] for j in range(length)]).reshape(-1,STATE_NUM)
        _, _, v = self.predict(_s)
        R = (np.where(d_,0,1) * v.reshape(-1,1)) * s_mask+R_


        #update params
        feed_dict={self.input:s_, self.action:a_, self.reward:R}
        SESS.run(self.store)
        try:
            summary, loss, _ = SESS.run([self.merged, self.loss, self.minimize],feed_dict)
        except:
            print(self.memory[0],self.memory[1],self.memory[2])
            assert False
        SESS.run(self.insert)

        self.writer.add_summary(summary, global_step = self.train_num)

        return loss



class ppo_agent:
    def __init__(self,brain):
        self.brain=brain
        self.memory=[]


    #get action without random
    def action(self,state):
        mu,sig,v = self.brain.predict(state)
        mu = mu.reshape(-1)
        count=0
        while True:
            try:
                action=np.random.multivariate_normal(mu,sig*np.eye(ACTION_NUM))
            except:
                print("error happend")
                return random(-1,1,size=ACTION_NUM)
            if sum(action>=-1)==ACTION_NUM and sum(action<=1)==ACTION_NUM:
                return action
            count+=1
            if  count>=10:
                return random(-1,1,size=ACTION_NUM)


    #get action with random
    def greedy_action(self,state):
        if frame >= EPS_STEPS:  
            eps = EPS_END
        else:
            eps = EPS_START + frame* (EPS_END - EPS_START) / EPS_STEPS  

        if np.random.random() <= eps:
            return random(-1,1,size=ACTION_NUM)
        else:
            return self.action(state)


    #push observation and action, reward, done, next observation
    def update(self,memory):
        if len(memory)<ADVANTAGE+1:
            R = sum([memory[j][2]*(GAMMA**j) for j in range(len(memory))])
            self.memory.append([memory[0][0],memory[0][1],R,memory[0][3],memory[0][4],GAMMA**ADVANTAGE])
            return 
        R = sum([memory[j][2]*(GAMMA**j) for j in range(ADVANTAGE+1)])
        self.memory.append([memory[0][0],memory[0][1],R,memory[0][3],memory[0][4],GAMMA**ADVANTAGE])

        for i in range(1,len(memory)-ADVANTAGE):
            R = ((R-memory[i-1][2])/GAMMA) + memory[i+ADVANTAGE][2]*(GAMMA**(ADVANTAGE-1))
            self.memory.append([memory[i][0],memory[i][1],R,memory[i+ADVANTAGE][3],memory[i][4],GAMMA**ADVANTAGE])
            
        for i in range(ADVANTAGE-1):
            R=((R-memory[len(memory)-ADVANTAGE+i][2])/GAMMA)
            self.memory.append([memory[i][0],memory[i][1],R,True,memory[i][4],GAMMA**(ADVANTAGE-i)])
        loss = self.brain.update(self.memory)
        # log=[memory[j][2] for j in range(len(memory))]
        # print(log)
        self.memory=[]
        return loss



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
        global total_trial
        global saver
        total_trial += 1
        self.total_trial+=1

        step=0
        if self.thread_type == "test":
            self.env.render(mode = "human")
        observation=self.env.reset()
        # if self.total_trial%100==0:
        #     print(SESS.run(self.agent.brain.weight_param))

        while True:
            step+=1
            frame+=1
            if self.thread_type=="train":
                action=self.agent.greedy_action(observation)
            elif self.thread_type=="test":
                action=self.agent.action(observation)
            
            next_observation,reward,done,_=self.env.step(action)

            self.memory.append([observation,action,reward,done,next_observation])

            observation=next_observation

            if step % 1000 == 0:
                break

            if step % 100 == 0:
                loss = self.agent.update(self.memory)
                self.memory = self.memory[-ADVANTAGE:]
                trial_rate = step // 100
                print("Thread:",self.name," Thread_trials:",self.total_trial,"progress",trial_rate,"/2","reward:","{:.8f}".format(reward)[:7],"loss:","{:.8f}".format(loss)[:7]," total_trial:",total_trial)
            
            if self.thread_type == "test":
                self.env.render(mode = 0)

            if done:
                break

        self.leaning_memory=np.hstack((self.leaning_memory[1:],step))

        if total_trial%(WORKER_NUM*5)==0 and args.save:
            saver.save(SESS,MODEL_SAVE_PATH)
            print("saved")
            total_trial+=1
            with open(MODEL_SAVE_PATH,"w") as f:
                f.write(str(total_trial))
        else:
            pass
        loss = self.agent.update(self.memory)
        if loss ==None:
            loss = 0
        self.memory = []

        # print("Thread:",self.name," Thread_trials:",self.total_trial,"progress","2 /2","reward:","{:.8f}".format(reward)[:7],"loss:","{:.8f}".format(loss)[:7]," total_trial:",total_trial)
        print("Thread:",self.name," Thread_trials:","{:8}".format(self.total_trial),"step","{:4}".format(step),"reward:","{:.8f}".format(reward)[:7],"loss:","{:.8f}".format(loss)[:7]," total_trial:",total_trial)
 

    

def main(args):
    global saver
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

    if args.show:
        global isLearned
        isLearned = True
        thread = []

    if args.load:
        global total_trial
        with open(MODEL_SAVE_PATH,"r") as f:
            total_trial=int(f.read())
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

if __name__=="__main__":


    SESS = tf.Session()
    saver = None
    frame = 0
    isLearned = False
    total_trial = 0
    
    main(args)

print("end")