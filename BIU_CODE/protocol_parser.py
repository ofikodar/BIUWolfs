from lark import Lark
from lark.exceptions import UnexpectedInput


with open("grammar.txt") as grammar:
    talk_parser = Lark(grammar, start='operators')


def is_protocol_correct(text):
    try:
        talk_parser.parse(text)
    except UnexpectedInput:
        return False
    return True


