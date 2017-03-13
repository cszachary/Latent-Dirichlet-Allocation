# Latent-Dirichlet-Allocation
LDA implement with Gibbs Sampling
LDA implement with Variational Inference will be added in the future
## Dependencies
- Python 2.7(Better on Linux)
- numpy
- matplotlib

## Getting Started
- Dataset
I choose 1000 users' query lists from Sougou corpus as our dataset, taking each user's query list as a document, which has been cut by Jieba tokenizer. I choose 80% of the data as our training set, and leave the rest 20% for testing.
- lda.py
A LDA class is defined in this file and use the Gibbs sampling to train the LDA model.
Besides, we use the perplexity to determine whether the model converged.
- datapreprocess.py
Some useful functions to preprocess the data, eg. transforming the word to id.
- main.py
To run the model just type:
'''
python main.py
'''
- results
Pictures of the training and inference process.
![](https://github.com/cszachary/Latent-Dirichlet-Allocation/blob/master/pic/learn.png)
![](https://github.com/cszachary/Latent-Dirichlet-Allocation/blob/master/pic/inference.png)
- **zachary zhang 2017/03/13**
