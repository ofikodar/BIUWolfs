#!/usr/bin/env python
from __future__ import print_function, division

import numpy as np
import pandas as pd
from scipy.special import softmax
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk import ngrams

# This sample script connects to the AIWolf server, but
# does not do anything else. It will choose itself as the
# target for any actions requested by the server, (voting,
# attacking ,etc) forcing the server to choose a random target.

import aiwolfpy
import aiwolfpy.contentbuilder as cb

START_TOKEN = "<s>"
END_TOKEN = "</s>"

myname = 'sample_python'


class SampleAgent(object):
    def __init__(self, agent_name, history_size=10000, n=2):
        # myname
        self.myname = agent_name
        self.model = MLE(n)
        self.n = n
        self.history = [""] * history_size
        self.last_index = 0

        # Index(['day', 'type', 'idx', 'turn', 'agent', 'text'], dtype='object')
        self.diff_data = pd.DataFrame({'day': [], 'type': [], 'idx': [], 'turn': [], 'agent': [], 'text': []})

    def getName(self):
        return self.myname

    # new game (no return)
    def initialize(self, base_info, diff_data, game_setting):
        self.base_info = base_info
        self.game_setting = game_setting
        self.agents = list(base_info['statusMap'].keys())
        print(base_info)
        # print(diff_data)

    # new information (no return)
    def update(self, base_info, diff_data, request):
        self.base_info = base_info
        if not diff_data.empty:
            self.diff_data = pd.concat([diff_data, self.diff_data])

    # Start of the day (no return)
    def dayStart(self):
        return None

    # conversation actions: require a properly formatted
    # protocol string as the return.
    def talk(self):
        return cb.over()

    def whisper(self):
        return cb.over()

    # targetted actions: Require the id of the target
    # agent as the return
    def vote(self):
        text = SampleAgent._text_from_diff_data(self.diff_data)  # unite all text
        train, vocab = padded_everygram_pipeline(self.n, text)
        self.model.fit(train, vocab)

        # get entropy per agent
        agent2entropy = {}
        for agent_idx in self.agents:
            agent_sentences = SampleAgent._text_from_diff_data(self.diff_data[self.diff_data['agent'] == float(agent_idx)])
            if not agent_sentences:  # ignore agents with no sentences (will always be me)
                continue
            agent2entropy[agent_idx] = sum([self.model.entropy(sentence) for sentence in agent_sentences])
        probabilities = softmax(list(agent2entropy.values()))
        idx = np.random.choice(list(agent2entropy.keys()), p=probabilities)
        return idx

    def attack(self):
        return self.base_info['agentIdx']

    def divine(self):
        return self.base_info['agentIdx']

    def guard(self):
        return self.base_info['agentIdx']

    # Finish (no return)
    def finish(self):
        return None

    # utils functions
    @staticmethod
    def _text_from_diff_data(diff_data):
        return diff_data[diff_data['type'] == 'talk']['text'].to_list()


agent = SampleAgent(myname)

# run
if __name__ == '__main__':
    aiwolfpy.connect_parse(agent)
