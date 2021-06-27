from lark import Lark
from lark import Transformer


class TreeToJson(Transformer):
    def string(self, s):
        (s,) = s
        return s[1:-1]
    def number(self, n):
        (n,) = n
        return float(n)

    list = list
    pair = tuple
    dict = dict

    null = lambda self, _: None
    true = lambda self, _: True
    false = lambda self, _: False

grammer = open("del2.txt")
json_parser = Lark(grammer, start='value')

text = '{"key": ["item0", "item1", 3.14]}'
tree = json_parser.parse(text)
transformer = TreeToJson()
print(transformer.transform(tree))
