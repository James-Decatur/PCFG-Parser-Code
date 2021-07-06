"""
CKY algorithm from the "Natural Language Processing" course by Michael Collins
https://class.coursera.org/nlangp-001/class
"""

import sys
from sys import stdin, stderr
from time import time
from json import dumps

from collections import defaultdict
from pprint import pprint

from pcfg import PCFG
from tokenizer import PennTreebankTokenizer


def argmax(lst):
    return max(lst) if lst else (0.0, None)

def backtrace(back, bp):
    if len(back) == 4:
        return [back[0], back[1]]
    if len(back) == 6:
        return [back[0], backtrace(bp[back[3], back[4], back[1]], bp), backtrace(bp[back[4], back[5], back[2]], bp)]

def CKY(pcfg, norm_words):
    sentence = [i[0] for i in norm_words] #Just cuts it down to the words in the sentence
    n = len(sentence) # Gets the length of a sentence

    chart = defaultdict(float)
    bp = defaultdict(tuple)

    for k in range(n):
        for C in pcfg.N:
            chart[k, k+1, C] = pcfg.q1[C, norm_words[k][0]] # Whenever we have recognized a parse tree that spans all words between min and max and whose root node is labeled with C, we set the entry chart[min][max][C] to true.
            bp[k, k+1, C] = (C, norm_words[k][1], k, k+1)

    for l in range(2, n+1):
        for e in range(l-2, -1, -1):
            for C in pcfg.N:
                best = 0.0

                for rule in pcfg.binary_rules[C]:
                    for s in range(e+1, l):
                        t1 = chart[e, s, rule[0]]
                        t2 = chart[s, l, rule[1]]
                        candidate = t1 * t2 * pcfg.q2[C, rule[0], rule[1]]
                        if candidate > best:
                            best = candidate
                            chart[e, l, C] = best
                            bp[e, l, C] = (C, rule[0], rule[1], e, s, l)

    return  backtrace(bp[0, n, "S"], bp)


class Parser:
    def __init__(self, pcfg):
        self.pcfg = pcfg
        self.tokenizer = PennTreebankTokenizer()

    def parse(self, sentence):
        words = self.tokenizer.tokenize(sentence)
        norm_words = []
        for word in words:
            norm_words.append((self.pcfg.norm_word(word), word))
        tree = CKY(self.pcfg, norm_words)

        if tree != None:
            tree[0] = tree[0].split("|")[0]
            return tree
        return None

def display_tree(tree):
    pprint(tree)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("usage: python3 parser.py GRAMMAR")
        exit()

    start = time()
    grammar_file = sys.argv[1]
    print("Loading grammar from " + grammar_file + " ...", file=stderr)
    pcfg = PCFG()
    pcfg.load_model(grammar_file)
    parser = Parser(pcfg)
    error_lst = []

    print("Parsing sentences ...", file=stderr)
    for sentence in stdin:
        tree = parser.parse(sentence)
        if tree != None:
            print(dumps(tree))
        else:
            print([])
            error_lst.append(1)

    print("Time: (%.2f)s\n" % (time() - start), file=stderr)
    #print(error_lst.count(1), 'errors raised')
