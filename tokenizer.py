
import sys,io,os,glob,nltk
from nltk.collocations import *
from collections import OrderedDict
from Bigram_Utils import Bigram

path =sys.argv[1]

bigram_generator = Bigram()

bigram_generator(path)

