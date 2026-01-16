#!/usr/bin/env python3
import unittest, random, sys, copy, argparse, inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np
import os
import traceback

# Import student submission
import submission
import sacrebleu
import nltk

import sys
import torch
import torch.nn as nn
import torch.nn.utils
from typing import Dict
from docopt import docopt
from vocab import Vocab
from nltk.translate.bleu_score import corpus_bleu

args = {
            'TEST_OUTPUT_FILE': './submission/test_outputs.txt',
            'TEST_GOLD_FILE': './chr_en_data/test.en'
        }

def bleu(args: Dict[str, str]):
    """ computes belu score
    @param args (Dict): args for file path details
    """

    # test_data_out = submission.read_corpus(args['TEST_OUTPUT_FILE'], source='tgt')
    # test_data_gold = submission.read_corpus(args['TEST_GOLD_FILE'], source='tgt')
    # min_len = min(len(test_data_out), len(test_data_gold))

    # bleu_score = corpus_bleu([[ref] for ref in test_data_gold[:min_len]],
    #                          [hyp for hyp in test_data_out[:min_len]])
    # print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    f = open(args['TEST_OUTPUT_FILE'], "r", encoding='utf8') #change path to submission
    hyps = []
    for sent in f:
      hyps.append(sent[:-1])     # gets rid of the end \n characters
    f.close()

    f = open(args['TEST_GOLD_FILE'], "r", encoding='utf8') #change to our local path
    refs = []
    for sent in f:
      refs.append(sent[:-1])
    f.close()
    bleu_score = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu_score.score


score = bleu(args)
print(f"Corpus BLEU: {score:.2f}")
