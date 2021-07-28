agents = []  # TODO

# units
subject = agents + ['UNSPEC' + 'ANY']
roles = ['VILLAGER', 'SEER', 'MEDIUM', 'BODYGUARD', 'WEREWOLF', 'POSSESSED', 'ANY']
species = ['HUMAN', 'WEREWOLF', 'ANY']

# sentences
knowledge = ['ESTIMATE', 'COMINGOUT']
actions = ['DIVINATION', 'GUARD', 'VOTE', 'ATTACK']
past_actions = ['DIVINED', 'IDENTIFIED', 'GUARDED', 'VOTED', 'ATTACKED']
agreement = ['AGREE', 'DISAGREE']

token2sentences = {
    'KNOW': knowledge,
    'ACTIONS': actions,
    'PAST': past_actions,
    'AGREE': agreement
}

# sentences = sum(token2sentences.values())

# operators
actions_operators = ['REQUEST', 'INQUIRE']
# for consistency operators type with only one operator are a list too
reasoning_operators = ['BECAUSE']
time_operators = ['DAY']
logic_operators = ['NOT', 'AND', 'OR', 'XOR']

token2operators = {
    # there are actions sentences and operators
    # to separate them we add OP to the operators token
    'ACTIONS_OP': actions,
    'REASON': reasoning_operators,
    'TIME': time_operators,
    'LOGIC': logic_operators
}
# operators = sum(token2operators.values())

# shapes
# TODO
print(sum(list(token2sentences.values())))