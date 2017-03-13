# data preprocess
import numpy as np

def readFile(filename):
	f = open(filename, "r")
	lines = f.readlines()
	f.close()
	return lines

def genDict(filename):
	# generate global dict by corpus
	lines = readFile(filename)
	cnt = 0
	dictionary = dict()
	for line in lines:
		l = line.strip("\n").split(" ")
		for w in l:
			if(not dictionary.has_key(w)):
				dictionary[w] = cnt
				cnt = cnt + 1
	return dictionary

def transformWordToID(filename, dictionary):
	lines = readFile(filename)
	# transform each document' each word into dictionary id
	docs = []
	for line in lines:
		doc = []
		l = line.strip("\n").split(" ")
		for w in l:
			doc.append(dictionary[w])
		docs.append(doc)
	return docs

def transformIDToWord(dictionary):
	return dict((v,k) for k,v in dictionary.iteritems())

