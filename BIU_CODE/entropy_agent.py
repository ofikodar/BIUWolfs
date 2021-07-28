#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.special import softmax
from model import Model
# from protocol_parser import is_protocol_correct

import aiwolfpy
import aiwolfpy.contentbuilder as cb

myname = 'BIU wolfs'

WEREWOLF_ROLE = "WEREWOLF"
POSSESSED_ROLE = "POSSESSED"
WHISPER_REQUESTS = 'WHISPER'
VILLAGER_ROLE = "VILLAGER"
HUMAN_SPECIES = "HUMAN"

dummy_sentences = ["Depending on what kind of questions everyone asks, I can assign the role of the person who asked.",
                   "I don't know who to vote because I'm crazy.",
                   "I'm not sure who to vote for.",
                   "'ll keep the right to silence in this round",
                   "We will wait for the vote and see who is right."
                   "I think the vote will speak for itself.",
                   "I won’t tell you"]
# cb.comingout(1, VILLAGER_ROLE),
# cb.request(cb.guard(1)),
# cb.estimate(1, VILLAGER_ROLE),
# cb.identified(1, HUMAN_SPECIES)

class EntropyOutlierAgent(object):
    def __init__(self, agent_name, n=3, natural_language=False):
        # myname
        self.myname = agent_name
        self.natural_language = natural_language
        self.n = n
        self._reset_agent()

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

    # new information (no return)
    def update(self, base_info, diff_data, request):
        # print(diff_data)  # TODO: del it
        self.base_info = base_info
        if not diff_data.empty:
            self.diff_data = pd.concat([self.diff_data, diff_data])
            # add executed and dead agents to the save list (you cant die twice)
            df_execute = diff_data[(diff_data['type'] == 'execute') & (diff_data['type'] == 'dead')]
            if not df_execute.empty:
                self.save_agents += df_execute['agent'].to_list()

    # Start of the day (no return)
    def dayStart(self):
        # if "".join(EntropyOutlierAgent._text_from_diff_data(self.diff_data))
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
        # if self.day == 1:
        #     return cb.skip()
        response = self.model.generate(max_length=100, best_of_k=100)
        if not response:
            return cb.over()
        return response

    def whisper(self):
        return cb.attack(int(self.attack()))

    # targetted actions: Require the id of the target
    # agent as the return
    def vote(self):
        self._update_state()
        if self.is_evil:
            probabilities = (1 - self.outlier_probabilities)
            if np.isin(False, self.agent2boldness == 0).item():
                probabilities *= self.agent2boldness
            return np.random.choice(self.agent2entropy, p=softmax(probabilities))
        else:
            return np.random.choice(self.agent2entropy, p=self.outlier_probabilities)

    def attack(self):
        if not hasattr(self, 'outlier_probabilities'):
            return self.base_info['agentIdx']
        probabilities = (1 - self.outlier_probabilities)
        if np.isin(False, self.agent2boldness == 0).item():
            probabilities *= self.agent2boldness
        if np.isin(False, self.agent2hate == 0).item():
            probabilities *= self.agent2hate
        return np.random.choice(self.agent2entropy, p=softmax(probabilities))

    def divine(self):
        if not hasattr(self, 'outlier_probabilities'):
            return self.my_idx
        idx = self.outlier_probabilities.argmax()
        return self._idx2agent(idx)

    def guard(self):
        scores = (1 - self.outlier_probabilities)
        if np.isin(False, self.agent2boldness == 0).item():
            scores *= self.agent2boldness
        idx = scores.argmax()
        return self._idx2agent(idx)

    # Finish (no return)
    def finish(self):
        # reset for next game
        self._reset_agent()
        return None

    def _update_state(self):
        # fit model
        self.day = int(self.diff_data['day'].max()) if not self.diff_data.empty else 1
        sentence_list = EntropyOutlierAgent._text_from_diff_data(self.diff_data)
        self.model.fit(sentence_list, self.day)
        # joint all text
        text = "".join(sentence_list)

        # join all whispers
        if self.is_evil:
            whispers = EntropyOutlierAgent._text_from_diff_data(self.diff_data, text_type="whisper")
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
            self.agent2boldness[idx] = text.count(f"Agent[{idx_str}]")
            if self.is_evil:
                self.agent2boldness[idx] += whispers.count(f"Agent[{idx_str}]")

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
        if hate_sum != 0:
            self.agent2hate /= hate_sum
            # print(self.agent2hate)
            # print(hate_sum)

    # utils functions #

    def _unsave_agents(self):
        return (agent_idx for agent_idx in self.agents if agent_idx not in self.save_agents)

    def _idx2agent(self, idx):
        return self.agents[idx]

    def _idx2str(self, idx):
        if type(idx) == int:
            idx = str(idx)
        return str(idx) if idx >= '10' else f"0{idx}"

    def _reset_agent(self):
        if self.natural_language:
            # self.model = Model(self.n, is_protocol_correct)
            self.model = Model(self.n)
        else:
            self.model = Model(self.n)
        self.day = 1
        self.message_queue = [] + dummy_sentences

    def _reset_diff_data(self):
        self.diff_data = pd.DataFrame({'day': [], 'type': [], 'idx': [], 'turn': [], 'agent': [], 'text': []})

    @staticmethod
    def _text_from_diff_data(diff_data, text_type='talk'):
        return diff_data[diff_data['type'] == text_type]['text'].to_list()


agent = EntropyOutlierAgent(myname)

# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
