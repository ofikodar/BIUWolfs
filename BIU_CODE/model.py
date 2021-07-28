from operator import itemgetter
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

START_TOKEN = "<s>"
END_TOKEN = "</s>"


class Model:
    def __init__(self, n, validator=None):
        self.model = MLE(n)
        self.n = n
        self.validator = validator

    def fit(self, sentence_list, weight):
        sentence_list = sentence_list * weight
        train, vocab = padded_everygram_pipeline(self.n, sentence_list)
        self.model.fit(train, vocab)

    def entropy(self, sentence_list):
        return sum([self.model.entropy(sentence) for sentence in sentence_list])

    def generate(self, max_length=100, best_of_k=100, random_seed=42):
        gen = (self._generate_sentence(max_length, random_seed + seed) for seed in range(best_of_k))
        if self.validator:
            # list for size validation
            gen2entropy = [(text, self.entropy(text)) for text in gen if self.validator(text)]
            if not gen2entropy:
                return None
        else:
            gen2entropy = ((text, self.entropy(text)) for text in gen)
        return min(gen2entropy, key=itemgetter(1))[0]

    def _generate_sentence(self, max_length=100, random_seed=42):
        generated_tokens = []

        assert max_length > 0, "The `length` must be more than 0."
        for idx, token in enumerate(self.model.generate(max_length, random_seed=random_seed)):
            if token == "<s>":
                continue
            if token == "</s>":
                break
            generated_tokens.append(token)
        return " ".join(generated_tokens)
