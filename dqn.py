import tensorflow as tf
import numpy as np
import gym
from time import sleep


def huber_loss(prediction, label, delta=0.5):
    error = label - prediction
    cond = tf.abs(error) < delta

    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(cond, squared_loss, linear_loss)

class DQNagent:
    #initialization
    def __init__(self,environmentname="model/model0"):
        self.env=gym.make("CartPole-v0")

        self.iteration=1
        self.batch_size=10000
        self.minibatch_size=32
        self.restore_rate=1
        self.state_table=[]
        self.next_table=[]
        self.action_table=[]
        self.reward_table=[]
        self.done_table=[]

        self.learning_rate=0.001
        self.modelname=environmentname
        self.epsiron=0.3
        self.action=[0,1]
        self.gamma=0.95

        self.sleep_time=0.01

        #model initialization
        self.init_model()


    def happen(self,rate):
        if np.random.random()<rate:
            return True
        else:
            return False

    def init_model(self):

        #placeholder
        self.X=tf.placeholder(dtype=tf.float32,shape=[None,4],name="input")
        hidden1=tf.layers.dense(self.X,100,activation=tf.nn.relu,name="hidden1")
        hidden2=tf.layers.dense(hidden1,20,activation=tf.nn.relu,name="hidden2")
        self.Out=tf.layers.dense(hidden2,2,name="output")

        #loss:特殊な関数を用いる
        self.t=tf.placeholder(dtype=tf.float32,shape=[None,2],name="teacher")

        self.loss=huber_loss(self.t,self.Out)
        self.loss_mean=tf.reduce_mean(self.loss)
        #optimizer
        self.optimizer=tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)

        #training operation
        self.trainig_op=self.optimizer.minimize(self.loss)

        #saver
        self.saver=tf.train.Saver()

        #session
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def q_value(self,state):
        return self.sess.run(self.Out,feed_dict={self.X:state})

    #update model
    def model_learn(self,state,action,next_state,reward,done):
        next_value_max=np.array(list(map(max,self.q_value(next_state)))).reshape(-1,1)
        new_state_value=self.sess.run(self.Out,feed_dict={self.X:state})
        for i in range(self.minibatch_size):
            if done[i]:
                new_state_value[i][action[i]]=reward[i][0]
            else:
                new_state_value[i][action[i]]=reward[i][0]+self.gamma*next_value_max[i][0]
        self.sess.run(self.trainig_op,feed_dict={self.X:state,self.t:new_state_value})

    #select with epsiron greedy policy
    def select_action(self,state,rate):
        #self.actionに登録された行動の中から一定確率でランダムに選択する
        if self.happen(rate):
            return np.random.choice(self.action)
        else:
            return self.prediction(state)

    def prediction(self,state):
        predict=self.sess.run(self.Out,feed_dict={self.X:state})
        return self.action[np.argmax(predict)]


    def add_experience(self,state,action,next_state,reward,done):
        if len(self.state_table)<self.batch_size:
            self.state_table.append(state)
            self.next_table.append(next_state)
            self.action_table.append(action)
            self.reward_table.append(reward)
            self.done_table.append(done)

    def experience_reply(self):
        index=np.random.permutation(self.batch_size)
        self.state_table=np.array(self.state_table).reshape(-1,4)
        self.next_table=np.array(self.next_table).reshape(-1,4)
        self.action_table=np.array(self.action_table).reshape(-1,1)
        self.reward_table=np.array(self.reward_table).reshape(-1,1)
        self.done_table=np.array(self.done_table)

        done_list=self.done_table[index][:self.minibatch_size]
        state_list=self.state_table[index][:self.minibatch_size]
        action_list=self.action_table[index][:self.minibatch_size]
        next_state_list=self.next_table[index][:self.minibatch_size]
        reward_list=self.reward_table[index][:self.minibatch_size]

        self.model_learn(state_list,action_list,next_state_list,reward_list,done_list)

        index=np.random.permutation(self.batch_size)
        self.state_table=self.state_table[index][self.minibatch_size:].tolist()
        self.next_table=self.next_table[index][self.minibatch_size:].tolist()
        self.action_table=self.action_table[index][self.minibatch_size:].reshape(-1).tolist()
        self.reward_table=self.reward_table[index][self.minibatch_size:].reshape(-1).tolist()
        self.done_table=self.done_table[index][self.minibatch_size:].tolist()


    def load_model(self,model_path):
        self.modelname=model_path
        self.saver.restore(self.sess,self.modelname)

    def save_model(self):
        self.saver.save(self.sess,self.modelname)

    #normalization for CartPole
    def normalization(self,state):
        ans=[0,0,0,0]
        ans[0]=state[0]
        ans[1]=state[1]
        ans[2]=state[2]
        ans[3]=state[3]
        return ans

    def serch_env(self):
        step_time=0
        observation=self.normalization(self.env.reset())
        action=self.select_action(np.array(observation).reshape(1,4),self.epsiron)
        while True:
            rate=self.epsiron*np.random.random()
            step_time+=1
            next_observation,_reward,done,info=self.env.step(action)
            next_observation=self.normalization(next_observation)
            #give a reward
            reward=0
            if done:
                if step_time>=195:
                    reward+=1.0
                else:
                    reward+=-1.0
            else:
                reward+=0.01
            #save experience with self.restore_rate
            if self.happen(self.restore_rate):
                self.add_experience(observation,action,next_observation,reward,done)

            if done:
                break
            observation=next_observation
            action=self.select_action(np.array(observation).reshape(1,4),rate)
        return step_time

    def show(self):
        step_time=0
        observation=self.normalization(self.env.reset())
        action=self.prediction(np.array(observation).reshape(1,4))
        self.env.render()
        sleep(self.sleep_time)
        while True:
            step_time+=1
            next_observation,reward,done,info=self.env.step(action)
            if done:
                break
            observation=self.normalization(next_observation)
            action=self.prediction(np.array(observation).reshape(1,4))
            self.env.render()
            sleep(self.sleep_time)
        return step_time



    def learn(self):
        count=0
        t=0
        for i in range(self.iteration):
            while True:
                count+=1
                t+=self.serch_env()
                if len(self.state_table)>=self.batch_size:
                    break
            self.experience_reply()
        return t/count

agent=DQNagent()
count=1
tcount=0
while True:
    meantimes=agent.learn()
    print("Epoch:",count,"  Mean time steps:",meantimes)
    t=agent.show()
    agent.learning_rate*=(count/(count+1))**(1/2)
    agent.epsiron*=(count/(count+1))**(1/2)
    if t>195:
        tcount+=1
    else:
        tcount=0
    count+=1
    if tcount>=10:
        break
    print("Time Step:",t,"  Learning Rate:",agent.learning_rate)
print(count,"times")
agent.save_model()