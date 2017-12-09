#coding: utf-8
import numpy as np
import tensorflow as tf 
import os
from grid_world import GridWorld
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

class DQN():
    def __init__(self, env, name):
        self.name = name
        self.env = env
        ################## 网络结构 ##############
        self.scalar_input =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.image = tf.reshape(self.scalar_input,shape=[-1,84,84,3])
        self.conv1 = tf.contrib.layers.convolution2d(inputs=self.image, num_outputs=32, 
            kernel_size=[8,8], stride=[4,4], padding='VALID', biases_initializer=None)
        self.conv2 = tf.contrib.layers.convolution2d(inputs=self.conv1, num_outputs=64, 
            kernel_size=[4,4], stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = tf.contrib.layers.convolution2d(inputs=self.conv2, num_outputs=64, 
            kernel_size=[3,3], stride=[1,1], padding='VALID', biases_initializer=None)
        self.conv4 = tf.contrib.layers.convolution2d(inputs=self.conv3, num_outputs=512, 
            kernel_size=[7,7], stride=[1,1], padding='VALID', biases_initializer=None)

        self.conv4_1, self.conv4_2 = tf.split(self.conv4, 2, 3)
        self.flat_1 = tf.contrib.layers.flatten(self.conv4_1)
        self.flat_2 = tf.contrib.layers.flatten(self.conv4_2)
        self.advantage_weight = tf.Variable(tf.random_normal([256, self.env.action_num]), name="advantage_weight")
        self.value_weight = tf.Variable(tf.random_normal([256, 1]), name="value_weight")
        self.advantage = tf.matmul(self.flat_1, self.advantage_weight)
        self.value = tf.matmul(self.flat_2, self.value_weight)
        
        self.q_value = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, reduction_indices=1, keep_dims=True))
        self.action = tf.argmax(self.q_value, 1)

        # self.actions 是一维的，类似于[1,1,2,3,1,0.....]
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        # self.target_q 是一维的，是self.actions 中每个动作对应的q值
        self.target_q = tf.placeholder(shape=[None], dtype=tf.float32)
        # self.q_value_of_action 是一维的，是self.actions 中每个动作对应的预测q值
        self.actions_onehot = tf.one_hot(self.actions, self.env.action_num, dtype=tf.float32)
        self.q_value_on_action = tf.reduce_sum(tf.multiply(self.q_value, self.actions_onehot), reduction_indices=1)
        
        self.loss = tf.reduce_mean(tf.square(self.target_q - self.q_value_on_action))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.00001)
        self.update_model = self.trainer.minimize(self.loss)

class ExperienceBuffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self, experience):
        num_to_remove = len(self.buffer) + len(experience) - self.buffer_size
        if num_to_remove > 0:
            for i in range(num_to_remove):
                self.buffer.pop(0)
        self.buffer.extend(experience)
            
    def sample(self, size):
        indices = np.random.choice(np.arange(len(self.buffer)), size=size)
        return [self.buffer[index] for index in indices]

     
def get_update_target_ops(variables, tau):
    main_DQN_vars = variables[0:len(variables)//2]
    target_DQN_vars = variables[len(variables)//2:]
    ops = []
    for i in range(len(target_DQN_vars)):
        ops.append(target_DQN_vars[i].assign(
            main_DQN_vars[i].value() * tau + (1 - tau) * target_DQN_vars[i].value()
            ))
    return ops

def update_target(ops, sess):
    for op in ops:
        sess.run(op)

########################## 超参数 #####################
batch_size = 32

#How often to perform a training step.
update_freq = 4
gamma = .99
random_upper_bound = 1
random_lower_bound = 0.1
annealing_steps = 10000
random_threshold = random_upper_bound
drop_step = (random_upper_bound - random_lower_bound) / annealing_steps

#How many episodes of game environment to train network with.
num_episodes = 10000
#How many steps of random actions before training begins.
pre_train_steps = 10000
#The max allowed length for one episode.
max_episode_length = 50
load_model = False 
path = "./dqn"
#Rate to update target network toward primary network
tau = 0.001

########################## 训练 #####################
env = GridWorld(size=5)
tf.reset_default_graph()
main_DQN = DQN(env, "main")
target_DQN = DQN(env, "target")

init_op = tf.global_variables_initializer()
update_target_ops_1 = get_update_target_ops(tf.trainable_variables(), 1)
update_target_ops_2 = get_update_target_ops(tf.trainable_variables(), tau)

saver = tf.train.Saver()
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    sess.run(init_op)
    
    #Set the target network to be equal to the main network.
    update_target(update_target_ops_2, sess)
    global_experience_buffer = ExperienceBuffer()
    
    #create list to contain total rewards per episode
    total_reward_list = []
    steps = 0
    for i in range(num_episodes + 1):
        episode_buffer = ExperienceBuffer()
        observation = env.reset()
        observation = np.reshape(observation, [21168])
        done = False
        total_reward_in_episode = 0
        steps_in_episode = 0
        while steps_in_episode < max_episode_length:
            # 积累样本
            steps_in_episode += 1
            steps += 1
            if np.random.rand(1) < random_threshold or steps < pre_train_steps:
                action = np.random.randint(0, 4)
            else:
                action = sess.run(main_DQN.action, feed_dict={main_DQN.scalar_input: [observation]})[0]
            new_observation, reward, done = env.step(action)
            new_observation = np.reshape(new_observation, [21168])
            #Save the experience to episode buffer
            episode_buffer.add([[observation, action, reward, new_observation, done]])
            
            total_reward_in_episode += reward
            observation = new_observation
            # 积累了 pre_train_steps/max_episode_length 次经历的pre_train_steps个样本之后才会第一次开始衰减随机门限，进行第一次训练。
            if steps >= pre_train_steps:
                if random_threshold > random_lower_bound:
                    random_threshold -= drop_step
                if steps % (update_freq) == 0:
                    train_batch = global_experience_buffer.sample(batch_size)
                    #Below we perform the Double-DQN update to the target Q-values
                    observations = np.vstack([record[0] for record in train_batch])
                    actions = np.array([record[1] for record in train_batch])
                    instant_rewards = np.array([record[2] for record in train_batch])
                    new_observations = np.vstack([record[3] for record in train_batch])
                    
                    action = sess.run(main_DQN.action, feed_dict={main_DQN.scalar_input: new_observations})
                    q_value, value, advantage = sess.run([target_DQN.q_value, target_DQN.value, target_DQN.advantage], feed_dict={target_DQN.scalar_input: new_observations})
                    labels = instant_rewards + gamma * q_value[range(batch_size) ,action]
                    # print(q_value, advantage)
                    # Update the network with our target values.
                    _ = sess.run(main_DQN.update_model, feed_dict={
                        main_DQN.scalar_input: observations,
                        main_DQN.target_q: labels,
                        main_DQN.actions: actions})

                    #update the target network towards the main network.
                    update_target(update_target_ops_2, sess)
        
        #Get all experiences from this episode
        global_experience_buffer.add(episode_buffer.buffer)
        total_reward_list.append(total_reward_in_episode)
        
        if i>0 and i % 25 == 0:
            print('episode',i,', average reward of last 25 episode', np.mean(total_reward_list[-25:]))
        #Periodically save the model.
        if i>0 and i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.cptk')
            print("Saved Model")
    saver.save(sess,path+'/model-'+str(i)+'.cptk')
