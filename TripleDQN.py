"""
基于教程上的3DQN修改
"""
import numpy as np
import gym
from collections import namedtuple
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from IPython import display

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))

device = 'cuda'
ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 300
BATCH_SIZE = 32
CAPACITY = 2000


def show_state(env, step=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Step: %d %s" % (step, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())


def seed_torch(seed):
    # with open('seed.txt', 'w') as f:
    #     f.write(str(seed))
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ReplayMemory:

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state, action, state_next, reward)

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):  # todo:网络结构可修改

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3_adv = nn.Linear(n_mid, n_out)
        self.fc3_v = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))

        adv = self.fc3_adv(h2)
        val = self.fc3_v(h2).expand(-1, adv.size(1))

        output = val + adv - adv.mean(1, keepdim=True).expand(-1, adv.size(1))

        return output


class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions

        self.memory = ReplayMemory(CAPACITY)

        n_in, n_mid, n_out = num_states, 32, num_actions  # todo:网络结构实现部分可修改
        self.main_q_network = Net(n_in, n_mid, n_out).to(device)
        self.target_q_network = Net(n_in, n_mid, n_out).to(device)

        self.optimizer = optim.Adam(
            self.main_q_network.parameters(), lr=0.0001)  # todo:神经网络学习率可修改

    def replay(self):

        if len(self.memory) < BATCH_SIZE:
            return

        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = \
            self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()
        self.update_main_q_network()

    def decide_action(self, state, episode):
        epsilon = 0.5 * (1 / (episode + 1))  # todo:贪婪系数计算方法可修改

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)  # 非目标Q函数更新

        else:
            action = torch.LongTensor(
                [[random.randrange(self.num_actions)]]).to(device)

        return action

    def make_minibatch(self):

        transitions = self.memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None]).to(device)  # 有些经验使游戏结束，无next_state，不能用

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):

        self.main_q_network.eval()
        self.target_q_network.eval()

        self.state_action_values = self.main_q_network(
            self.state_batch).gather(1, self.action_batch)  # 用于选择的主状态动作Q值

        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None,
                                                    self.batch.next_state))).to(device)  # 下个状态是否结束的flag
        next_state_values = torch.zeros(BATCH_SIZE).to(device)

        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor).to(device)

        a_m[non_final_mask] = self.main_q_network(
            self.non_final_next_states).detach().max(1)[1]

        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        next_state_values[non_final_mask] = self.target_q_network(
            self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values
        return expected_state_action_values

    def update_main_q_network(self):

        self.main_q_network.train()

        loss = F.smooth_l1_loss(self.state_action_values,
                                self.expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        self.brain.update_target_q_network()


class Environment:
    """
    环境类
    """
    def __init__(self, seed):
        seed_torch(seed)
        self.env = gym.make(ENV).unwrapped  # gym环境随机种子
        self.env.seed(seed)  # gym环境随机种子
        num_states = self.env.observation_space.shape[0]  # 状态数
        num_actions = self.env.action_space.n  # 动作数
        self.agent = Agent(num_states, num_actions)
        print('ob_space:', self.env.observation_space)
        print('ac_space:', self.env.action_space)

    def run(self):
        step_record = []

        for episode in range(NUM_EPISODES):

            observation = self.env.reset()  # 环境初始化

            state = observation
            state = torch.from_numpy(state).type(
                torch.FloatTensor)
            state = torch.unsqueeze(state, 0).to(device)

            for step in range(MAX_STEPS):
                # show_state(self.env, step + 1, ENV)  # 展示
                action = self.agent.get_action(state, episode)

                observation_next, _, done, _ = self.env.step(
                    action.item())

                if done:
                    state_next = None

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])

                    else:
                        reward = torch.FloatTensor([1.0])

                    step_record.append(step + 1)
                else:
                    reward = torch.FloatTensor([0.0])
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0).to(device)

                self.agent.memorize(state, action, state_next, reward)

                self.agent.update_q_function()

                state = state_next

                if done or (step == MAX_STEPS - 1):
                    # print('%d Episode: Finished after %d' % (
                    #     episode, step + 1))  # 输出提示
                    if step == MAX_STEPS - 1:
                        step_record.append(step + 1)

                    if episode % 2 == 0:  # todo:target网络更新间隔可修改
                        self.agent.update_target_q_function()
                    break

        return step_record