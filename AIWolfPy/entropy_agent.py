#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.special import softmax
from model import Model
from protocol_parser import is_protocol_correct

import aiwolfpy
import aiwolfpy.contentbuilder as cb

myname = 'BIU wolfs'

WEREWOLF_ROLE = "WEREWOLF"
POSSESSED_ROLE = "POSSESSED"


class EntropyOutlierAgent(object):
    def __init__(self, agent_name, history_size=10000, n=3, natural_language=False):
        # myname
        self.myname = agent_name
        if natural_language:
            self.model = Model(n, is_protocol_correct)
        else:
            self.model = Model(n)
        self.day = 1
        self.message_queue = []

    def getName(self):
        return self.myname

    # new game (no return)
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting
        self.my_idx = self.base_info['agentIdx']
        self.my_idx_str = self._idx2str(self.my_idx)
        self.agents = [agent_idx for agent_idx in base_info['statusMap'].keys() if agent_idx != base_info['agentIdx']]
        self.agents2idx = {agent_idx: i for i, agent_idx in enumerate(self.agents)}
        self.is_evil = base_info['myRole'] == WEREWOLF_ROLE or base_info['myRole'] == POSSESSED_ROLE
        self.save_agents = []

        print(self.agents, self.agents2idx)

    # new information (no return)
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        if not diff_data.empty:
            self.diff_data = pd.concat([self.diff_data, diff_data])
        print(diff_data)

    # Start of the day (no return)
    def dayStart(self):
        self._reset_diff_data()
        self.agent2entropy = np.zeros(len(self.agents2idx))
        self.agent2boldness = np.zeros(len(self.agents2idx))
        self.agent2hate = np.zeros(len(self.agents2idx))
        return None

    # conversation actions: require a properly formatted
    # protocol string as the return.
    def talk(self):
        if self.message_queue:
            return self.message_queue.pop()
        if self.day == 1:
            return cb.skip()
        response = self.model.generate(max_length=100, best_of_k=100)
        if not response:
            return cb.over()
        return response

    def whisper(self):
        return cb.attack(self.attack())

    # targetted actions: Require the id of the target
    # agent as the return
    def vote(self):
        self._update_state()
        if self.is_evil:
            probabilities = (1 - self.outlier_probabilities).dot(self.agent2boldness)
            return np.random.choice(self.agent2entropy, p=probabilities)
        else:
            return np.random.choice(self.agent2entropy, p=self.outlier_probabilities)

    def attack(self):
        if not hasattr(self, 'outlier_probabilities'):
            return self.base_info['agentIdx']
        probabilities = (1 - self.outlier_probabilities).dot(self.agent2boldness).dot(self.agent2hate)
        return np.random.choice(self.agent2entropy, p=probabilities)

    def divine(self):
        if not hasattr(self, 'outlier_probabilities'):
            return self.base_info['agentIdx']
        idx = self.outlier_probabilities.argmax()
        return self._idx2agent(idx)

    def guard(self):
        scores = np.dot(1 - self.outlier_probabilities, self.agent2boldness)
        idx = scores.argmax()
        return self._idx2agent(idx)

    # Finish (no return)
    def finish(self):
        return None

    def _update_state(self):
        # fit model
        self.day = int(self.diff_data['day'].max()) if not self.diff_data.empty else 1
        sentence_list = EntropyOutlierAgent._text_from_diff_data(self.diff_data)  # unite all text
        self.model.fit(sentence_list, self.day)
        # joint all text
        text = "".join(sentence_list)
        for agent_idx in self._unsave_agents():
            if agent_idx in self.save_agents:
                continue
            # agent index
            idx = self.agents2idx[agent_idx]
            idx_str = self._idx2str(idx)

            # get agent sentences
            df_agent = self.diff_data[self.diff_data['agent'] == float(agent_idx)]
            agent_sentences = EntropyOutlierAgent._text_from_diff_data(df_agent)
            if not agent_sentences:  # ignore agents with no sentences (will always be me)
                continue

            # update entropy
            self.agent2entropy[idx] = self.model.entropy(agent_sentences)

            # update boldness
            # TODO: add whisper
            self.agent2boldness[idx] = text.count(f"Agent[{idx_str}]")

            # update hate
            self.agent2hate[idx] = sum([sentence.count(self.my_idx_str) for sentence in agent_sentences])

        # update probabilities
        self.outlier_probabilities = softmax(self.agent2entropy)

        # normalize boldness
        boldness_sum = self.agent2boldness.sum()
        if boldness_sum != 0:
            self.agent2boldness /= boldness_sum

        # normalize hate
        hate_sum = self.agent2hate.sum()
        if boldness_sum != 0:
            self.agent2hate /= hate_sum

    def _unsave_agents(self):
        return (agent_idx for agent_idx in self.agents if agent_idx not in self.save_agents)

    def _idx2agent(self, idx):
        return self.agents[idx]

    def _idx2str(self, idx):
        if type(idx) == int:
            idx = str(idx)
        return str(idx) if idx >= '10' else f"0{idx}"

    def _reset_diff_data(self):
        self.diff_data = pd.DataFrame({'day': [], 'type': [], 'idx': [], 'turn': [], 'agent': [], 'text': []})

    # utils functions
    @staticmethod
    def _text_from_diff_data(diff_data):
        return diff_data[diff_data['type'] == 'talk']['text'].to_list()


agent = EntropyOutlierAgent(myname)

# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
