import numpy as np
import tensorflow as tf
from model import SRCNN
from nnconfig import  srcnn_flags
import pprint
import os




pp = pprint.PrettyPrinter()

def main():
    flag = srcnn_flags()
    srcnn = SRCNN(flag)
    srcnn.train()
if __name__ == '__main__':
    main()