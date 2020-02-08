# nlp_project

* **Action Step**:
  1. Research on possible datasets
  2. Research on possible algorithms 
  3. Implement Extractive Summary on news
  4. Implement Abstractive Summary on news
  5. Implement Sentiment Analysis on news 

## Tools 

* **Possible Datasets**
  1. [CNN and Daily Mail] (https://github.com/abisee/cnn-dailymail)
  2. [Amazon Fine Food reviews] (https://www.kaggle.com/snap/amazon-fine-food-reviews#Reviews.csv)
  3. [XSum Dataset] (https://github.com/EdinburghNLP/XSum)
 
* **Possible Libraries**
  1. [SpaCy] (https://spacy.io/)
  2. [keras] (https://www.tensorflow.org/guide/keras)
  3. [Gensim] (https://radimrehurek.com/gensim/)
  4. [PyTorch] (https://pytorch.org/) 

## Algorithms 

In the field of text summarization, there are mainly two ways to do so, namely Extractive and Abstractive Approach.

### 1. Extractive Approach 

* Retrieve the most relevant sentences or words from the input text and combine them together as the summary. Therefore, the output summary may come from the exact wordings from the input text.

#### TextRank

TextRank is based on the famous PageRank algorithm which has been deployed in the Google search enigne. The algorithm is built on the idea of majority voting, "The webpage which is linked by other webpages is important, but the webpage which is linked by many other linked webpages is more important".  The links between the webpages can be expressed by a matrix , and the matrix will be converted as a transition probability matrix.  By using the iteration formula, the weights of each node (webpage) can be determined accordingly.

In TextRank, we will regard each sentences in the documents as the node. For the matrix formualtion, the sentence similarity will be used in filling the entries in the matrix. After carrying out the iterative process, sentence with highest weighting will be the summary.  The sentence similarity can be computed in many different ways, in the original paper, it uses the overlapping words frequency between two sentences as the similarity.  Also, the similarity can be obtained by word embedding method as well.

Reference paper: 
[TextRank: Brining Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf)

#### Latent Semantic Analysis 

#### Unsupervised learning with skip-thought vectors 

### 2. Abstractive Approach 
