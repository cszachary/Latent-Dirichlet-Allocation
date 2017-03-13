# main function for lda

import numpy as np
from lda import *
from datapreprocess import *

if __name__ == '__main__':
	corpusFileName = "data/corpus.txt"
	trainFileName = "data/train.txt"
	testFileName = "data/test.txt"

	dictionary = genDict(corpusFileName)
	id2word = transformIDToWord(dictionary)
	trainDocs = transformWordToID(trainFileName, dictionary)
	testDocs = transformWordToID(testFileName, dictionary)


	lda = LDA(k = 20, alpha = 2.5, beta = 0.01, docs = trainDocs, voca = len(dictionary))
	lda.learning(iteration = 200, vocabulary = id2word)
	lda.inference(testDocs, iteration = 100)
