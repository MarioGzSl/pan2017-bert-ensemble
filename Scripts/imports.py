import xml.etree.ElementTree as ET
import pandas as pd
import torch
import os
import numpy as np

from nltk.tokenize import TweetTokenizer

from nltk import ngrams

import re

import random

from gensim.models import KeyedVectors

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import tqdm

import copy

from sklearn.svm import LinearSVC


