l = ["knowledge",
     "actions",
     "past_actions1",
     "past_actions2",
     "agreement",
     "flow",
     "unsubject_sentences",
     "sentences",
     "nested_sentences",
     "actions_operators",
     "reasoning_operators",
     "time_operators",
     "not_operator",
     "logic_operators",
     "non_operator",
     "unsubject_operators",
     "operators"]


def form(role):
    return f"""
    @staticmethod
    def {role}(s):
        return TestTransformer.format("{role}", s)\n
    """


for r in l:
    print(form(r))
