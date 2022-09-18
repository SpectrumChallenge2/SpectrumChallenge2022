from utility.environment_interface import EnvironmentInterface
import numpy as np
import itertools
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.categorical import Categorical
from pathlib import Path

class Agent:
    def __init__(self, environment, discount_factor, dnn_learning_rate):
        super(Agent, self).__init__()

        self._env = environment
        self._discount_factor = discount_factor
        self._dnn_learning_rate = dnn_learning_rate
        self._num_freq_channel = 4
        self._max_num_unit_packet = 2
        self._sta_list = list(range(0, 8))
        self._observation = np.zeros(self._num_freq_channel)
        self._num_freq_channel_combination = 2 ** self._num_freq_channel - 1
        self._num_action = self._num_freq_channel_combination * self._max_num_unit_packet + 1
        self._freq_channel_combination = [np.where(np.flip(np.array(x)))[0].tolist()
                                          for x in itertools.product((0, 1), repeat=self._num_freq_channel)][1:]
        self._actor = Actor(input_size=self._num_freq_channel, output_size=self._num_action)
        self._critic = Critic(input_size=self._num_freq_channel, output_size=1)
        self._eps = 1e-25
        self._total_param = list(self._actor.parameters()) + list(self._critic.parameters())
        self._optimizer = torch.optim.Adam(params=self._total_param, lr=self._dnn_learning_rate)
        self._model_path = str(Path(__file__).parent.resolve() / 'my_model')

    def set_init(self):
        initial_action = {'type': 'sensing'}
        observation_dict = self._env.step(initial_action)
        self._observation = self.convert_observation_dict_to_arr(observation_dict['observation'])

    def train(self, num_episode, run_time):
        self._env.disable_video_logging()
        self._env.disable_text_logging()
        print("Train start")
        for episode in range(num_episode):
            self._env.start_simulation(time_us=run_time)
            self.set_init()
            loss_avg, reward_avg = self.train_episode()
            print(f"{episode} episode => loss_avg: {loss_avg}, reward_avg: {reward_avg}")
        print("finish")
        self.save_model()

    def test(self, run_time: int):
        # self._env.disable_video_logging()
        # self._env.disable_text_logging()
        self._env.enable_video_logging()
        self._env.enable_text_logging()
        self._env.start_simulation(time_us=run_time)
        self.set_init()
        self.load_model()
        while True:
            observation = torch.Tensor(self._observation).type(torch.float32)
            logits, dist, prob = self._actor(observation)
            action_index = torch.argmax(prob)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:
                break
            self._observation = self.convert_observation_dict_to_arr(observation_dict['observation'])
            print(f"{self._env.get_score()            }\r", end='', flush=True)

    def train_episode(self):
        loss_lst = []
        reward_lst = []
        while True:
            observation = torch.Tensor(self._observation).type(torch.float32)
            logits, dist, probs = self._actor(observation)
            action_index = torch.argmax(probs)
            action_dict = self.convert_action_index_to_dict(action_index)
            observation_dict = self._env.step(action_dict)
            if observation_dict == 0:       # finish episode
                loss_avg = np.mean(loss_lst)
                reward_avg = np.mean(reward_lst)
                return loss_avg, reward_avg
            else:
                reward = torch.Tensor([observation_dict['reward']]).type(torch.float32)
                next_observation = self.convert_observation_dict_to_arr(observation_dict['observation'])
                next_observation = torch.Tensor(next_observation).type(torch.float32)

                # compute targets
                with torch.no_grad():
                    td_target = reward + self._discount_factor * self._critic(next_observation)
                    td_error = td_target - self._critic(observation)

                # compute actor loss
                prob = probs[action_index]
                actor_loss = -torch.log(prob + self._eps) * td_error
                actor_loss = actor_loss.mean()
                # compute critic loss
                v = self._critic(observation)
                critic_loss = F.mse_loss(v, td_target)

                # backpropagation
                loss = actor_loss + critic_loss
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                loss_lst.append(loss.item())
                reward_lst.append(reward.item())

    def convert_observation_dict_to_arr(self, observation):
        observation_type = observation['type']
        observation_arr = np.zeros(self._num_freq_channel)
        if observation_type == 'sensing':
            is_sensed = observation['sensed_freq_channel']
            observation_arr[is_sensed] = 1
        elif observation_type == 'tx_data_packet':
            observation_arr[:] = 2
            success_freq_channel_list = observation['success_freq_channel']
            observation_arr[success_freq_channel_list] = 3
        return observation_arr

    def convert_action_index_to_dict(self, action_index):
        if action_index == 0:
            action_dict = {'type': 'sensing'}
        else:
            num_unit_packet = int((action_index - 1) // self._num_freq_channel_combination + 1)
            freq_channel_combination_index = (action_index - 1) % self._num_freq_channel_combination
            freq_channel_list = self._freq_channel_combination[freq_channel_combination_index]
            sta_allocation_dict = {}
            for freq_channel in freq_channel_list:
                freq_channel = int(freq_channel)
                sta = random.choice(self._sta_list)
                sta_allocation_dict[freq_channel] = sta
            action_dict = {'type': 'tx_data_packet', 'sta_allocation_dict': sta_allocation_dict,
                           'num_unit_packet': num_unit_packet}
        return action_dict

    def save_model(self):
        save_dict = {
            "Actor": self._actor.state_dict(),
            "Critic": self._critic.state_dict()
        }
        torch.save(save_dict, self._model_path)

    def load_model(self):
        saved_dict = torch.load(self._model_path)
        self._actor.load_state_dict(saved_dict["Actor"])
        self._critic.load_state_dict(saved_dict["Critic"])


class Actor(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, num_hidden_layer: int = 2):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_hidden_layer = num_hidden_layer
        self._act = nn.ReLU()
        self._input_layer = nn.Linear(self._input_size, self._hidden_size)
        self._hidden_layer = nn.Linear(self._hidden_size, self._hidden_size)
        self._output_layer = nn.Linear(self._hidden_size, self._output_size)

    def forward(self, x):
        x = self._input_layer(x)
        x = self._act(x)
        for _ in range(self._num_hidden_layer):
            x = self._hidden_layer(x)
            x = self._act(x)
        x = self._output_layer(x)
        logit = x
        dist = Categorical(logits=logit)
        prob = dist.probs
        return logit, dist, prob


class Critic(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128, num_hidden_layer: int = 2):
        super().__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size
        self._num_hidden_layer = num_hidden_layer
        self._act = nn.ReLU()
        self._input_layer = nn.Linear(self._input_size, self._hidden_size)
        self._hidden_layer = nn.Linear(self._hidden_size, self._hidden_size)
        self._output_layer = nn.Linear(self._hidden_size, self._output_size)

    def forward(self, x):
        x = self._input_layer(x)
        x = self._act(x)
        for _ in range(self._num_hidden_layer):
            x = self._hidden_layer(x)
            x = self._act(x)
        x = self._output_layer(x)
        return x


if __name__ == "__main__":
    env = EnvironmentInterface()
    env.connect()
    agent = Agent(environment=env, discount_factor=0.9, dnn_learning_rate=0.000001)

    # When submitting, the training part should be excluded,
    # and it should be submitted in a form that can be evaluated by loading the trained model like agent.test()
    agent.train(num_episode=1000, run_time=100000)
    agent.test(500000)
