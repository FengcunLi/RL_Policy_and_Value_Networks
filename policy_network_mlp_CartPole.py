#!coding: utf-8
import numpy as np 
import gym 
import tensorflow as tf 
env = gym.make("CartPole-v0")

################################# 网络结构 #####################
hidden_n = 50
batch_size = 25
learning_rate = 1e-1
input_dim = 4
output_dim = 1
gamma = 0.99

observation_placeholder = tf.placeholder(tf.float32, [None, input_dim], name="observation_placeholder")
w_1 = tf.get_variable("w_1", shape=[input_dim, hidden_n], initializer=tf.contrib.layers.xavier_initializer())
layer_1 = tf.nn.relu(tf.matmul(observation_placeholder, w_1))
w_2 = tf.get_variable("w_2", shape=[hidden_n, output_dim], initializer=tf.contrib.layers.xavier_initializer())
probability = tf.nn.sigmoid(tf.matmul(layer_1, w_2))

################################# 批优化 #####################
'''
深度强化学习的训练也是采用 batch training，不逐个“样本”的更新参数，而累计 batch_size 个回合再更新参数，防止
单一回合随机扰动噪声对模型训练带来的不利影响。
不像 CNN 那样一个 batch 样本 feed 进模型，直接产生一个梯度平均值就可以进行一次参数的更新，
这里需要进行 batch_size 次经历从而得到 batch_size 个回合，需要存储每回合中的平均梯度，并进行平均，最终更新一次参数。
'''
adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
w_1_grad = tf.placeholder(tf.float32, name="batch_grad_1")
w_2_grad = tf.placeholder(tf.float32, name="batch_grad_2")
batch_grad = [w_1_grad, w_2_grad]
update = adam.apply_gradients(zip(batch_grad, tf.trainable_variables()))

################################# 损失 #####################
'''
在 CartPole 问题中，每一次得到的奖励和这次奖励之前的所有动作都有关，也就说之前的所有动作导致了这次奖励的得到，
为了计算一个动作带来的奖励我们要计算这个动作之后全部奖励的折扣和。
我们倒推求解每一个动作带来的奖励。在 CartPole 任务中除了导致任务失败的那次动作之外，所有动作的即时奖励都是 1。
一个动作带来的奖励是后一时间步动作带来的奖励的折扣加上这个动作的即时奖励。
'''
def discount_reward(r):
    d_r = np.zeros_like(r)
    d_r[-1] = r[-1]
    for time in range(len(r)-1)[::-1]:
        d_r[time] =  d_r[time+1] * gamma + r[time]
    return d_r

# 模型做出的动作的反动作
opposite_action = tf.placeholder(tf.float32, [None, 1], name="opposite_action")
# 动作带来的奖励
advantage = tf.placeholder(tf.float32, name="reward_signal")
'''
模型得到这份奖励的概率：
opposite_action=1 时，即 action=0，1- probability
opposite_action=0 时，即 action=1，probability
综合上面两个式子，模型得到这份奖励的概率：
opposite_action * (opposite_action - probability) + (1 - opposite_action) * (opposite_action + probability)
'''
# 模型得到这份奖励的概率的对数：
log_prob = tf.log(opposite_action * (opposite_action - probability) 
    + (1 - opposite_action) * (opposite_action + probability))
loss = -tf.reduce_mean(log_prob * advantage)
gradients = tf.gradients(loss, tf.trainable_variables())

################################# 训练 #####################
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    observation = env.reset()
    xs, ys, rewards = [], [], []
    reward_sum = 0
    episode = 1
    total_episodes = 10000

    grad_buffer = sess.run(tf.trainable_variables())
    for i, grad in enumerate(grad_buffer):
        grad_buffer[i] = grad * 0

    while episode <= total_episodes:
        x = np.reshape(observation, [1, input_dim])
        prob = sess.run(probability, feed_dict={observation_placeholder: x})
        action = 1 if np.random.uniform() < prob else 0
        xs.append(x)
        ys.append(1 - action)
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        rewards.append(reward)
        if done:
            ################################# 构建一个回合的样本，计算该回合中所有样本上的梯度和，并加入到此批次的梯度buffer中 #####################
            episode += 1
            xs_per_episode = np.vstack(xs)
            ys_per_episode = np.vstack(ys)
            rewards_per_episode = np.vstack(rewards)
            xs, ys, rewards = [], [], []
            d_r = discount_reward(rewards_per_episode)
            d_r -= np.mean(d_r)
            d_r /= np.std(d_r)
            grad_per_episode = sess.run(gradients, feed_dict={observation_placeholder: xs_per_episode,
                opposite_action: ys_per_episode,
                advantage: d_r})
            for i, grad in enumerate(grad_per_episode):
                grad_buffer[i] += grad

            if episode % batch_size == 0:
                sess.run(update, feed_dict={w_1_grad: grad_buffer[0], w_2_grad: grad_buffer[1]})

                for i, grad in enumerate(grad_buffer):
                    grad_buffer[i] = grad * 0

                print("Average reward for episode %d : %.2f" %(episode, reward_sum/batch_size))

                if reward_sum/batch_size >= 200:
                    print("Task solved in %d episodes" % episode)
                    break
                reward_sum =0
            observation = env.reset()
