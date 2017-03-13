# Latent Dirichlet Allocation + Gibbs Sampling

import numpy as np
import random
import matplotlib.pyplot as plt

class LDA:
	def __init__(self, k, alpha, beta, docs, voca):
		'''
		k: number of topics
		alpha: hyper parameter of Dirichlet Distribution
		beta: hyper parameter of Dirichlet Distribution
		docs: documents to perform LDA eg.(doc_examples, doc_length)
		voca: vocabulary of the corpus
		'''
		self.k = k
		self.alpha = alpha
		self.beta = beta
		self.docs = docs
		self.voca = voca

		# temp variable to store sampling process
		self.p = np.zeros(self.k)
		# word count in each document' each topic
		self.num_mz = np.zeros((len(self.docs), self.k))
		# the sum of each word count in each documnent
		self.num_m_sum = np.zeros(len(self.docs)) 
		# word count in each topic' each vocabulary
		self.num_zw = np.zeros((self.k, self.voca))
		# the sum of each word count in each topic
		self.num_z_sum = np.zeros(self.k)
		# topic of each document's each word
		self.z_m_w = []

		for m, doc in enumerate(docs):
			# the topic of each word in this document
			z_w = []
			self.num_m_sum[m] += len(doc)
			for t in doc:
				z = np.random.randint(0, self.k)
				z_w.append(z)
				self.num_mz[m, z] += 1
				self.num_zw[z, t] += 1
				self.num_z_sum[z] += 1
			self.z_m_w.append(np.array(z_w))
		self.theta = np.zeros((len(self.docs), self.k))
		self.phi = np.zeros((self.k, self.voca))

	def sampling(self, i, j):
		'''perform gibbs sampling'''
		# the topic of the jth word of the ith documnent 
		topic = self.z_m_w[i][j]
		word = self.docs[i][j]
		self.num_zw[topic, word] -= 1
		self.num_mz[i, topic] -= 1
		self.num_z_sum[topic] -= 1
		self.num_m_sum[i] -= 1

		Beta = self.voca * self.beta
		Alpha = self.k * self.alpha

		self.p = (self.num_mz[i] + self.alpha)/(self.num_m_sum[i] + Alpha) * \
				 (self.num_zw[:, word] + self.beta)/(self.num_z_sum + Beta)
		for t in xrange(1, self.k):
			self.p[t] += self.p[t-1]
		u = random.uniform(0, self.p[self.k - 1])
		# generate new topic
		for t in xrange(self.k):
			if(self.p[t] > u):
				break

		self.num_zw[t, word] += 1
		self.num_z_sum[t] += 1
		self.num_mz[i, t] += 1
		self.num_m_sum[i] += 1

		return t

	def calculateTheta(self):
		# calculate doc-topic distribution
		self.theta = (self.num_mz + self.alpha)/(self.num_m_sum[:, np.newaxis] + self.k * self.alpha)

	def calculatePhi(self):
		# calculate topic-word distribution
		self.phi = (self.num_zw + self.beta)/(self.num_z_sum[:, np.newaxis] + self.voca * self.beta)

	def perplexity(self):
		N = 0
		log_perplexity = 0
		self.calculateTheta()
		self.calculatePhi()
		for m, doc in enumerate(self.docs):
			for w in doc:
				log_perplexity -= np.log(np.inner(self.phi[:, w], self.theta[m,:]))
			N += len(doc)
		return np.exp(log_perplexity/N)

	def learning(self, iteration, vocabulary):
		print("-------------Training-------------------")
		# save history perplexity to plot
		history_perplexity = []
		init_perplexity = self.perplexity()
		history_perplexity.append(init_perplexity)
		print("---initial perplexity = %f" %(init_perplexity))
		for x in xrange(iteration):
			for i in xrange(len(self.docs)):
				for j in xrange(len(self.docs[i])):
					topic = self.sampling(i, j)
					self.z_m_w[i][j] = topic
			perp = self.perplexity()
			history_perplexity.append(perp)
			print("---%dth iteration perplexity = %f" %(x + 1, perp))
		# save the final parameters
		self.calculateTheta()
		self.calculatePhi()
		self.output_word_topic_distribution(vocabulary)
		plotIterationPerplexityPicture(iteration, history_perplexity, mode = "learning")
		

	def output_word_topic_distribution(self, vocabulary):
		zcount = np.zeros(self.k, dtype = int)
		wcount = [dict() for k in range(self.k)]
		for wlist, zlist in zip(self.docs, self.z_m_w):
			for w, z in zip(wlist, zlist):
				zcount[z] += 1
				if w in wcount[z]:
					wcount[z][w] += 1
				else:
					wcount[z][w] = 1

		phi = self.phi
		for k in range(self.k):
			print("\n--- topic: %d (%d words)" %(k, zcount[k]))
			for w in np.argsort(-phi[k])[:20]:
				print(" %s: %f(%d)" %(vocabulary[w], phi[k,w], wcount[k].get(w, 0)))

	def inference(self, docs_t, iteration):
		# inference new document's topic
		# keep phi fix
		print("\n-------------inference-------------------")
		p = np.zeros(self.k)
		num_mz_t = np.zeros((len(docs_t), self.k))
		num_m_sum_t = np.zeros(len(docs_t))
		num_zw_t = np.zeros((self.k, self.voca))
		num_z_sum_t = np.zeros(self.k)
		z_m_w_t = []
		for m, doc in enumerate(docs_t):
			z_w_t = []
			num_m_sum_t[m] += len(doc)
			for t in doc:
				z = np.random.randint(0, self.k)
				z_w_t.append(z)
				num_zw_t[z, t] += 1
				num_mz_t[m, z] += 1
				num_z_sum_t[z] += 1
			z_m_w_t.append(np.array(z_w_t))
		theta = np.zeros((len(docs_t), self.k))
		alpha = self.alpha
		k = self.k
		phi = self.phi
		Alpha = alpha * k

		history_perplexity = []
		init_perplexity = inferencePerplexity(docs_t, num_mz_t, num_m_sum_t, alpha, k, phi)
		history_perplexity.append(init_perplexity)
		print("initial perplexity = %f" %(init_perplexity))
		for x in xrange(iteration):
			for i in xrange(len(docs_t)):
				for j in xrange(len(docs_t[i])):
					topic = inferenceSampling(i, j, p, num_mz_t, num_zw_t, num_m_sum_t, num_z_sum_t, alpha, k, Alpha, phi, z_m_w_t, docs_t)
					z_m_w_t[i][j] = topic
			perp = inferencePerplexity(docs_t, num_mz_t, num_m_sum_t, alpha, k, phi)
			history_perplexity.append(perp)
			print("--%dth iteration perplexity = %f" %(x + 1, perp))
		theta = calculateTheta(num_mz_t, num_m_sum_t, alpha, k)
		output_document_topic_distribution(docs_t, k, theta)
		plotIterationPerplexityPicture(iteration, history_perplexity, mode = "inference")

def inferenceSampling(i, j, p, num_mz_t, num_zw_t, num_m_sum_t, num_z_sum_t, alpha, k, Alpha, phi, z_m_w_t, docs_t):
	topic = z_m_w_t[i][j]
	word = docs_t[i][j]
	num_zw_t[topic, word] -= 1
	num_mz_t[i, topic] -= 1
	num_z_sum_t[topic] -= 1
	num_m_sum_t[i] -= 1
			
	p = (num_mz_t[i] + alpha)/(num_m_sum_t[i] + Alpha) * phi[:, word]
	for t in xrange(1, k):
		p[t] += p[t-1]
	u = random.uniform(0, p[k - 1])
	for t in xrange(k):
		if(p[t] > u):
			break
	num_zw_t[t, word] += 1
	num_z_sum_t[t] += 1
	num_mz_t[i, t] += 1
	num_m_sum_t[i] += 1

	return t

def inferencePerplexity(docs_t, num_mz_t, num_m_sum_t, alpha, k, phi):
	N = 0
	log_perplexity = 0
	theta = calculateTheta(num_mz_t, num_m_sum_t, alpha, k)
	for m, doc in enumerate(docs_t):
		for w in doc:
			log_perplexity -= np.log(np.inner(phi[:, w], theta[m,:]))
		N += len(doc)
	return np.exp(log_perplexity/N)

def calculateTheta(num_mz_t, num_m_sum_t, alpha, k):
	theta = (num_mz_t + alpha)/(num_m_sum_t[:, np.newaxis] + alpha * k)
	return theta

def output_document_topic_distribution(docs_t, k, theta):
	print("\ndocument-topic distribution (M-by-K dimension)")
	for i in xrange(len(docs_t)):
		print("\n--- %dth document topic: " %(i + 1))
		for j in xrange(k):
			print("%dth topic: %f" %(j + 1, theta[i, j]))

def plotIterationPerplexityPicture(iteration, history_perplexity, mode):
	plt.figure()
	plt.xlabel("Iteration")
	plt.ylabel("Perplexity")
	plt.grid(True)
	if(mode == "learning"):
		plt.title("Learning perplexity curve")
		plt.plot(range(iteration + 1), history_perplexity, '-*r', linewidth = 2)
		plt.savefig("learn.png")
	else:
		plt.title("Inference perplexity curve")
		plt.plot(range(iteration + 1), history_perplexity, '-*k', linewidth = 2)
		plt.savefig("inference.png")