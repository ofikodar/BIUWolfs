from lark import Transformer
from parser.parser import talk_parser

class TestTransformer(Transformer):

    @staticmethod
    def format(node, value):
        return f"{node}: {value}"

    @staticmethod
    def number(n):
        (n,) = n
        return float(n)

    @staticmethod
    def string(s):
        (s,) = s
        return s[1:-1]

    @staticmethod
    def subject(s):
        return TestTransformer.format("subject", s)

    @staticmethod
    def target(s):
        return TestTransformer.format("target", s)

    @staticmethod
    def role(s):
        return TestTransformer.format("role", s)

    @staticmethod
    def specie(s):
        return TestTransformer.format("specie", s)

    @staticmethod
    def talk_number(s):
        return TestTransformer.format("talk_number", s)

    @staticmethod
    def day_number(s):
        return TestTransformer.format("day_number", s)

    @staticmethod
    def agent(s):
        return TestTransformer.format("agent", s)

    @staticmethod
    def knowledge(s):
        return TestTransformer.format("knowledge", s)

    @staticmethod
    def actions(s):
        return TestTransformer.format("actions", s)

    @staticmethod
    def past_actions1(s):
        return TestTransformer.format("past_actions1", s)

    @staticmethod
    def past_actions2(s):
        return TestTransformer.format("past_actions2", s)

    @staticmethod
    def agreement(s):
        return TestTransformer.format("agreement", s)

    @staticmethod
    def flow(s):
        return TestTransformer.format("flow", s)

    @staticmethod
    def unsubject_sentences(s):
        return TestTransformer.format("unsubject_sentences", s)

    @staticmethod
    def sentences(s):
        return TestTransformer.format("sentences", s)

    @staticmethod
    def nested_sentences(s):
        return TestTransformer.format("nested_sentences", s)

    @staticmethod
    def actions_operators(s):
        return TestTransformer.format("actions_operators", s)

    @staticmethod
    def reasoning_operators(s):
        return TestTransformer.format("reasoning_operators", s)

    @staticmethod
    def time_operators(s):
        return TestTransformer.format("time_operators", s)

    @staticmethod
    def not_operator(s):
        return TestTransformer.format("not_operator", s)

    @staticmethod
    def logic_operators(s):
        return TestTransformer.format("logic_operators", s)

    @staticmethod
    def non_operator(s):
        return TestTransformer.format("non_operator", s)

    @staticmethod
    def unsubject_operators(s):
        return TestTransformer.format("unsubject_operators", s)

    @staticmethod
    def operators(s):
        return TestTransformer.format("operators", s)



examples = [
    # "Agent1 DIVINATION Agent1",
    "REQUEST Agent2 (DIVINATION Agent3)",
    "asd",
    "Agent2 Agent3",
    "REQUEST Agent2 (Agent2 Agent3)"

]
i = 0
print(talk_parser.parse(examples[i]).pretty())
print(TestTransformer().transform(talk_parser.parse(examples[i])).replace("\\", "").replace("\'", ""))
for exmp in examples:
    print(is_protocol_correct(exmp))