import roboschool
import gym
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import threading
from time import sleep
from gym import wrappers
# 基本的な環境の使用方法
# env = gym.make('RoboschoolAnt-v1')
# env.reset()
# print(env.observation_space)
# print(env.action_space)
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.high)
# print(env.action_space.low)
# while True:
#     ob,r,_,done=env.step(env.action_space.sample())
#     print(r)
    # env.render()


"""
今回の実装：連続行動空間において行動決定を行うモデルの作成
今回用いるモデル：'RoboschoolAnt-v1'
a3c.pyを改良して行う
変更点
・入力部分の変更(入力：２８次元、出力：８次元)
・ネットワークの出力部分を行動確率分布の平均と偏差を出力するように変更
・行動選択部分の変更
ネットワークの出力：行動確率（平均（８次元）mu、分散（一次元））sig、状態価値（一次元）v
ネットワークの損失関数部分
agentのaction部分
Workerの報酬評価部分
環境から与えられるrewardは重心の進み具合と思われる
これの差を報酬として与えることにする
さらに一エピソードを１０００ステップと設定しておく→1000ステップたつと強制的に終了
そして１００ステップごとに学習を行うことにする
"""
parser=ArgumentParser(description="Reiforcement training with A3C",add_help=True)
parser.add_argument("--model_path",type=str,required=True,help="model path. required")
parser.add_argument("--env_name",default="RoboschoolAnt-v1",help="environment name. default is CartPole-v0")
parser.add_argument("--save",action="store_true",default=False,help="save command")
parser.add_argument("--load",action="store_true",default=False,help="load command")
args=parser.parse_args()

#define constatns

sample_env=gym.make(args.env_name)

#video
video_path="./video"

WORKER_NUM=8
ADVANTAGE=10
ENV_NAME=args.env_name
STATE_NUM=28
ACTION_NUM=len(sample_env.action_space.high)
ACTION_SPACE=[(x,y) for x,y in zip(sample_env.action_space.low,sample_env.action_space.high)]
#[(-1,1),(-1,1),(-1,1),(-1,1)]
GREEDY_EPS=0.1
GAMMA=0.99
LEARNING_RATE=0.001
RMS_DECAY=0.99
LOSS_V=0.5
LOSS_ENTROPY=0.02
HIDDEN_LAYERE1=100
OUTPUT_FILEPATH=args.model_path

def random(min,max,size):
    return np.random.uniform(size=size)*(max-min)+min


class brain:
    def __init__(self,name,parameter_server):
        self.name=name
        self.parameter_server=parameter_server
        self.build_model()

    def build_model(self):
        with tf.variable_scope(self.name):
            self.input=tf.placeholder(dtype=tf.float32,shape=[None,STATE_NUM])
            hidden1=tf.layers.dense(self.input,HIDDEN_LAYERE1,activation=tf.nn.leaky_relu)
            self.mu=tf.layers.dense(hidden1,ACTION_NUM)
            self.theta=tf.layers.dense(hidden1,1)
            self.v=tf.layers.dense(hidden1,1)

        self.reward=tf.placeholder(dtype=tf.float32,shape=(None,1))
        self.action=tf.placeholder(dtype=tf.float32,shape=(None,ACTION_NUM))


        advantage=self.reward-self.v
        self.sig=1/(1+tf.exp(-self.theta))
        self.prob=tf.exp(-tf.square(self.action-self.mu)/(2*tf.square(self.sig)))/self.sig
        self.log_prob=tf.log(self.prob+1e-10)
        self.policy_loss=-tf.reduce_sum(tf.stop_gradient(advantage)*self.log_prob,axis=1,keepdims=True)

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
        mu,sig,v=SESS.run([self.mu,self.sig,self.v],feed_dict)
        return mu.reshape(-1),sig.reshape(-1),v.reshape(-1)

    def update_parameter(self):
        feed_dict={self.input:self.s_, self.action:self.a_, self.reward:self.R}
        # print("::::::::",feed_dict,"::::::::")
        SESS.run(self.update_parameter_server,feed_dict)
        return SESS.run(self.loss,feed_dict)

    def pull_parameter(self):
        SESS.run(self.pull_parameter_server)

    def push_parameter(self):
        SESS.run(self.push_parameter_server)
    #preprocessing memory data [observation, action, R,done,next_observation],state_mask
    #make t 
    def make_train_table(self,memory):
        length=len(memory)
       
        self.s_=np.array([memory[j][0] for j in range(length)]).reshape(-1,STATE_NUM)
        self.a_=np.vstack((memory[j][1] for j in range(length))).reshape(-1,ACTION_NUM)
        self.R_=np.array([memory[j][2] for j in range(length)]).reshape(-1,1)
       
        self.d_=np.array([memory[j][3] for j in range(length)]).reshape(-1,1)
        s_mask=np.array([memory[j][5] for j in range(length)]).reshape(-1,1)
        _s=np.array([memory[j][4] for j in range(length)]).reshape(-1,STATE_NUM)
        _,_, v=self.predict(_s)
        self.R=(np.where(self.d_,0,1)*v.reshape(-1,1))*s_mask+self.R_
        # print([memory[j][2] for j in range(length)])


class Parameter_server:
    def __init__(self):
        print("************model initialization****************")
        self.build_model()
        self.weight_param=tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES,scope="server_model")
        self.optimizer=tf.train.RMSPropOptimizer(LEARNING_RATE,RMS_DECAY)
    def build_model(self):
        with tf.variable_scope("server_model"):
            self.input=tf.placeholder(dtype=tf.float32,shape=[None,STATE_NUM])
            hidden1=tf.layers.dense(self.input,HIDDEN_LAYERE1,activation=tf.nn.leaky_relu)
            self.mu=tf.layers.dense(hidden1,ACTION_NUM)
            self.theta=tf.layers.dense(hidden1,1)
            self.v=tf.layers.dense(hidden1,1)
class agent:
    def __init__(self,name,parameter_server):
        self.brain=brain(name,parameter_server)
        self.memory=[]

    #get action without random
    def action(self,state):
        mu,sig,v = self.brain.predict(state)
        count=0
        while True:
            try:
                action=np.random.multivariate_normal(mu,sig*np.eye(ACTION_NUM))
            except:
                print(mu,sig)
                action=random(-1,1,size=ACTION_NUM)
            if sum(action>=-1)==ACTION_NUM and sum(action<=1)==ACTION_NUM:
                break
            count+=1
            if  count>=10:
                action=random(-1,1,size=ACTION_NUM)
                break
        return action

    #get action with random
    def greedy_action(self,state):
        if np.random.random() <= GREEDY_EPS:
            return random(-1,1,size=ACTION_NUM)
        else:
            action=self.action(state)
            return action

    def pull_parameter_server(self):
        self.brain.pull_parameter()

    #push observation and action, reward, done, next observation
    def push_advantage_reward(self,memory):
        R = sum([memory[j][2]*(GAMMA**j) for j in range(ADVANTAGE+1)])
        self.memory.append([memory[0][0],memory[0][1],R,memory[0][3],memory[0][4],GAMMA**ADVANTAGE])

        for i in range(1,len(memory)-ADVANTAGE):
            R = ((R-memory[i-1][2])/GAMMA) + memory[i+ADVANTAGE][2]*(GAMMA**(ADVANTAGE-1))
            self.memory.append([memory[i][0],memory[i][1],R,memory[i+ADVANTAGE][3],memory[i][4],GAMMA**ADVANTAGE])
            
        for i in range(ADVANTAGE-1):
            R=((R-memory[len(memory)-ADVANTAGE+i][2])/GAMMA)
            self.memory.append([memory[i][0],memory[i][1],R,True,memory[i][4],GAMMA**(ADVANTAGE-i)])
        self.brain.make_train_table(self.memory)
        # log=[memory[j][2] for j in range(len(memory))]
        # print(log)
        self.memory=[]

    def finish_leaning(self):
        self.brain.push_parameter()
    
    def train(self):
        return self.brain.update_parameter()


class Worker:
    def __init__(self,thread_type,thread_name,parameter_server):
        self.thread_type=thread_type
        self.name=thread_name
        self.agent=agent(thread_name,parameter_server)
        self.parameter_server=parameter_server
        self.env=gym.make(ENV_NAME)
        if self.thread_type=="test":
            self.env=wrappers.Monitor(self.env, video_path, force=True)
        self.loss_memory=np.zeros(10)
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
        self.total_trial+=1
        step=0
        observation=self.env.reset()
        global isLearned
        global frame
        self.agent.pull_parameter_server()

        # if self.total_trial%100==0:
        #     print(SESS.run(self.agent.brain.weight_param))
        distance=0
        while True:
            step+=1
            frame+=1
            if self.thread_type=="train":
                action=self.agent.greedy_action(observation)
            elif self.thread_type=="test":
                self.env.render()
                action=self.agent.action(observation)
                sleep(0.02)
            next_observation,next_distance,done,_=self.env.step(action)

            reward=next_distance-distance
            
            self.memory.append([observation,action,reward,done,next_observation])

            observation=next_observation
            distance=next_distance

            if step%100==0:
                self.agent.push_advantage_reward(self.memory)
                self.loss_memory=np.hstack((self.loss_memory[1:],self.agent.train()))
                self.memory=[]
                print("Thread:",self.name," Thread_trials:",self.total_trial," score:",step,"-",distance," loss:",self.loss_memory.mean()," total_step:",frame)
            if step==1000:
                break
            
        self.leaning_memory=np.hstack((self.leaning_memory[1:],step)) 
        #when finish learning
        if self.total_trial>=100:
            isLearned=True
            sleep(3)
            # self.agent.finish_leaning()
        else:
            # self.agent.push_advantage_reward(self.memory)
            # self.agent.train()
            # self.memory=[]
            pass


def main(args):
    #make thread
    with tf.device("/cpu:0"):
        parameter_server=Parameter_server()
        thread=[]
        for i in range(WORKER_NUM):
            thread_name="local_thread"+str(i)
            thread.append(Worker(thread_type="train",thread_name=thread_name,parameter_server=parameter_server))
    
    COORD=tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    if args.load:
        ckpt = tf.train.get_checkpoint_state('./trained_model')
        if ckpt:
            saver.restore(SESS,args.model_path)

    runnning_thread=[]
    for worker in thread:
        job=lambda: worker.run_thread()
        t=threading.Thread(target=job)
        t.start()
        runnning_thread.append(t)
    COORD.join(runnning_thread)
    test=Worker(thread_type="test",thread_name="test_thread",parameter_server=parameter_server)
    test.run_thread()
    if args.save:
        saver.save(SESS,args.model_path)

if __name__=="__main__":   
    SESS=tf.Session()

    frame=0
    isLearned=False
    main(args)

print("end")


