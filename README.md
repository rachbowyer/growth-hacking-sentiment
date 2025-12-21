# Growth Hacking sentiment

This repo contains my solution to the Manning Live Project ["Sentiment Analysis and Natural Language Processing for Marketing"](https://www.manning.com/liveproject/sentiment-analysis-and-natural-language-processing-for-marketing)

The project dates from 2020 and is based around the then state-of-the-art BERT series of LLMs. 


## Setting up the Python environment

The code currently works with Python version 3.13.9. I use miniconda and pip to create the environment.
The C/data science heavy packages come from conda. The rest from pip.

To create from scratch use:

      conda create -n growth-hacking-sentiment python=3.13 numpy pandas
      conda activate growth-hacking-sentiment
      conda install pytorch pytorch-cuda=12.4 -c pytorch -c nvidia
      conda install scipy matplotlib
      conda install scikit-learn
      conda install tabulate
      conda install jupyter
      pip install -r requirements.txt

Alternatively, to run on Ubuntu and Nvdia CUDA, the conda environment can be
created from the environment.yml file

    conda env create -f environment.yml

## Getting the data

The project uses a dataset of reviews of video games from Amazon. It can be downloaded from:

    https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Video_Games_5.json.gz

It should be unzipped and should be placed in the `growth-hacking-sentiment/data` directory.


## Part 1

In Part 1 two datasets are created. The first, the small data set, contains 4500 items. It was balanced

| Rating | Frequency |
|--------|-----------|
| 1      | 1500      |
| 2      | 500       |
| 3      | 500       |
| 4      | 500       |
| 5      | 1500      |

using Random under sampling.

The second dataset contains 100k items.


## Part 2

In Part 2 a simple dictionary based sentiment classifier is created. No training set is needed so it
is applied to the whole of the small data set.

Correlations between the sentiment score from the classifier and the human "rating" of the game are found.

 * A simple contingency table is created.
 * Chi squared measure of independence
 * Spearman's ρ. This looks at the rank correlations

 The conclusion is that there is a loose correlation between the sentiment score and the human rating.

 An investigation reveals that context of words is a challenge for the dictionary based classifier. For example, the word "limited" is scored as having negative sentiment. However, in one review it is used in the phrase "limited edition" - something that is viewed as positive.


## Part 3

In Part 3 the accuracy of the dictionary based classifier is measured.

Both the human ratings and the model sentiment are converted into classes {negative, neutral, positive}.  A rating of 1 is viewed as negatives, ratings of 2, 3, 4 are viewed as neutral and a rating of 5 is viewed as positive. This gives 3 balanced ratings classes.

Sentiment of < -2 is viewed as negative, sentiment > 2 is viewed as positive, otherwise sentiment is viewed as neutral.

The confusion matrix is calculated. Also the precision, recall and F1 metrics. Precision, recall and F1 are calculated slightly different in multi-class problems compared to binary problems.

First, they are calculated for each sentiment categoryns individually. Then the "macro average" is calculated which is the average of the individual values. And finally a weighted average is used to take into account class imbalances. In this case, the macro and weighted averages are the same as there are no class imbalances.


### Part 4

In Part 4, various Small Language Models (SLMs) are fine tuned/trained to classify sentiment.

| Model | Weighted F1 |
|-------|-------------|
| Dictionary model with negation handling | 30% |
| Pre-trained DistilBERT classifier from Hugging Faces Transformer Library | 62% |
| DistilBERT with classification head trained and 512 byte sequences| 72.5%  |
| RoBERTa with classification head trained and 512 byte sequences| 76% |
| Fine tuned RoBERTa with classification head trained | 74% |

The instructors' model answer used DistilBERT with a trained classification head and had an F1 score of 68%. In the model answer fine tuning the DistilBERT model did not improve classification accuracy.



## Part 5

In Part 5, the data is analysed to identify key phrases associated with positive and negative sentiment.

The first step is to use the model from Part 4 to classify game reviews into 3 corpora based on sentiment: negative, neutral and positive. Then bigrams are extract from each corpus and ranked by log likelihood. This enables key phrases to be found in each corpus, hopefully allowing the identification of features of successful video games. Although many of the bigrams were generic e.g ('great', 'game'), some bigrams do identify features of games. The data tells us reliability, good graphics, compelling game play and competitive pricing are all key ingredients of a successful game.


### Positive sentiment

Features of games that are viewed favourably:

* works perfectly
* good price
* graphics amazing
* great fun


### Negative sentiment

Features of games that are viewed less favourably:

* stopped working
* game boring
* bad graphics
* poorly made



## Reflections on the project

I have really enjoyed the project and learnt a lot. It might be felt that in an era of LLMs, the project is out of date. I disagree. Small language models (SLM), such as RoBERTa are still widely used in practice as they are cheap and easy to train and run.

Having said that, there seemed to be a number of methodological flaws in the project. This might be down to the available tech at the time, access to data or simplifications made for pedagogical purposes.

### Labelling the data

The project relies on a key assumption: that the rating of a game reflects the sentiment of the review. In particular, games rated 1 were viewed as negative reviews. Games rated 5, as positive, and games rated 2, 3 and 4 as neutral. It might be that some people who leave a negative review rate a game 2 stars and some people who leave a neutral view also rate the game as 2 stars. So although the assumption is probably correct for 1 and 5 star game reviews, it might be incorrect for other reviews.

It is also worth mentioning that the labels were not created by humans reading the reviews and deciding the sentiment. They were generated by the human who wrote the review and recorded their sentiment. So a British person, for example, might have hated a game, rated it 1 stars, but merely put in the review "It was ok". Not knowing the reliability of the labels means that there could be flaws with both the training of the SLMs and assessing the model's accuracy.


### Training the models

The instructors' solution appears to leak data in training the classification head as the test (hold out) data set was used as part of the training process to decide when to stop training.

Like the instructors, I found that unsupervised fine tuning of the SLM (I used RoBERTa and they used DistilBERT) did not improve the metrics, but supervised training of the classification head was effective. Unfortunately, no explanation was offered as to why fine tuning can help improve performance in some cases and why in this case it did not appear to help.

My understanding is that the main rationale for unsupervised fine tuning is to help a model understand the domain better - for example law or maths - where a specific vocabulary is used. And although gamers do have their own vocabulary, the poor quality of the writing in a lot of the reviews probably blunted any advantages.

I think it would be interesting to explore a supervised training approach that trained the classification head and the upper neuron layers, whilst leaving the base layers frozen. Unfortunately this would need to be done in PyTorch or similar, rather than Simple Transformers, and hence was probably out of scope for the project.


### Finding the key phrases

The model that is trained earlier is used to create 3 corpora (negative, neutral, positive). It is not explained why the model is used instead of the rating of the game given by the reviewer. It might be the goal of the model is to "average" out outliers, a bit like linear regression. Or the model might have been used for pedagogical purposes.

Had the project required the reviews to be parsed into sentences, then the model could have been used to determine the sentiment of individual sentences. This would have helped to avoid the problem of positive sentences appearing in poor reviews and vice-versa.

When the log likelihood for corpora was calculated, no guidance was given as to the reference corpus. I initially concatenated all 3 corpora together to create a reference corpus. This was the same approach the authors took in their model answer. But, this might be problematic as the positive corpus was far larger than the other two copora and hence it would dominate the reference corpus. I tried an alternative approach of calculating the log likelihood for just the positive and negatives corpora using the neutral corpus as a reference. However, this did not appear to provide much improvement.

The project required doing a scatter plots of log likelihood against reference frequency, but did not provide any tasks that made use of the plot. The purpose of the plots might have been to verify the keyness functionality worked correctly.

The approach of using bigrams with stop words filtered and keyness has a number of limitations. The bigrams do not capture the full context so positive phrases such as "great game" appear often in the negative corpus. Also, keyness looks at the syntactic form of a word rather than the semantic meaning. This means wadding through a lot of "noise" to find actionable insights.


