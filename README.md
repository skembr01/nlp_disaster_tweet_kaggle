## Problem and Data

This challenge on Kaggle is a Natural Language Processing (NLP) problem analyzing the text in tweets from Twitter. This analysis involves a binary classification to determine if a tweet is pertaining to a disaster (or an emergency) or not. For the challenge, data were provided as csv files and were split to training data, test data, and a sample submission was presented. 

In the training data there were the columns: id, text, location, keyword, and target. In the test data there were the columns: id, text, location, and keyword. The overall shape of the training data is (7613, 5) and the shape of the test data is (3263, 4). 

## EDA

By looking at the info of the data, I saw that there were NaN's placed in both the keyword and location columns for both the training and the test data. As I utilized both columns in my modeling, I decided to convert the NaN into a '' so the NLP would not be affected. It is of note that there were no NaN's in the text column of either data frame. 

Then, I utilized altair to chart the number of rows per class in the training data. This was followed up by charts illustrating the number of null values for both location and keyword columns with training data. After this, I added values for keywords at the beginning of each text value. Further, I added each location value at the end of each text value for both train and test values.

After this, I set X to be the training text column and y to be the training target column. Additionally X_test was set to test text option.

Further, I used LabelEncoder on the y labels so as to prepare for modeling.

## NLP Preprocessing

When performing NLP, the text needs to be preprocessed to improve training performance. As an example consider three variations of an English word: 'thoughtfulness', 'thôüghtfulnéss', and <thoughtfulness/>. While an English speaker will likely be able to understand the word at hand in each of these three variations, the first is the simplest as it is the typical spelling and does not require any extra steps for comprehension besides basic reading. A machine learning model or neural network will operate in much the same way. It derives meaning from the word and even further at the root of the word i.e. 'thoughtfullness' has meaning from 'thoughtful' and then even at the term 'thought.' 

Thus, I began to clean the words to their useful form. To do this, I borrowed upon a past NLP project I had completed, but also as a refresher utilized the article listed below. Functions were created and applied to the text in both training and test dataset so in order to: remove urls, remove punctuation, remove accents on characters, remove html markings, and to make every character lower case.

Then, I utilized the nltk and spacy packages to remove stopwords from the tweets. Stopwords are words such as 'an', 'a', 'or', 'that' which are quite common in the English language but which provide little semantic meaning. Given the low meaning associated with these words, they are removed so the model can focus on more meaningful content. NLTK and spacy each have their own list of English stopwords and I removed words from tweets that were in either list. 

Lastly, I lemmatized each word in every tweet. Lemmatizing is the process of using contextual analysis of words to remove endings of words and return the lemma of a word or its root, base meaning. For example, consider the word 'saw.' If it were used as a noun (the tool) then the lemma of 'saw' would remain 'saw' as there is no base of the word which still captures its meaning. However, if 'saw' were utilized as a verb, then its lemma would become 'see' as at its core, 'saw' is a form of 'see' and this captures its base meaning. Here we can see lemmatizing can return different values depending on the context a word is used in. Hence, this can be more powerful than stemming and is why I chose to utilize it. Further, the process itself helps the model in its performance as the model can still learn contextual meaning without having to deal with certain ends of words and overall reduce the vocabulary.

Also, I plotted the 25 most frequent words for both training and test data utilizing nltk FrequencyDistribution.

Data preprocessing source: https://towardsdatascience.com/nlp-preprocessing-with-nltk-3c04ee00edc0

Lemmatizing source: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html

## Vectorization and Tf dataset

Once we have completed data cleaning, we then need to vectorize data for input into our neural network. Vectorization is the idea of representing text as numerical vectors. These vectors are data that a neural network can understand and work with. Given the conversion from text to vector, the embedding is crucial to ensure each vector contains meaning similar to the word itself. 

In order to do this I utilized the tensorflow method TextVectorization. Here, I set the output mode for each value in the vectors to be integers. Then, the vectorizer was 'adapted' to the values in X. This means that the vectorizer utilized the vocabulary found in my cleaned text values in order to operate with. Then, once the vectorizer was built on the vocabulary, I created the numerical vectorized version of the training and test text.

After the word embedding (vectorization), the data needs to be prepared for input into the model. Building upon my second source below, I created a tf Dataset from the vectorized X and y values. Then, the dataset was shuffled for each iteration of the model so that the order of the data did not leak into the model. Further, I cached and prefetched the data so as to improve moodel performance. This allows the gpu to load the data in parallel with the modeling. Finally, I separated the dataset into a training and validation set for modeling purposes.

Sources: 

https://www.tensorflow.org/api_docs/python/tf/keras/layers/TextVectorization

https://www.tensorflow.org/guide/keras/preprocessing_layers


## Modeling

Utilizing keras, I then began to create a sequential model. I began by creating an embedding layer which randomly initializes weights to map the integers and their words to certain vectors. Through backpropagation, these embedding weights get trained so that similarities between words will be recognized.

I initially explored a LSTM layer for my RNN; however, based on the lecture I decided to utilize a gated recurrent unit (GRU) layer. After exploring hyperparameter tuning, I eventually optimized at 64 units in the layer. Further, I made this GRU to be a part of a bidirectional layer. Based upon the third source below, this enables the model at any point x to train on data that came before x, but also to train on data upcoming in x. Additionally after this birdirectional GRU, I utilized two dense layers, both with relu activation and with 64 and 8 units, respectively. Finally, given the binary classification, I utilized a dense layer with 1 unit and sigmoid activation function.

After defining the layers, I then created a learning rate scheduler, which lowers exponentially each epoch after the first epoch. Then, I created an early stopper which starts after the 5th epoch (this was chosen given my initial high learning rate, so that the model can deepen as learning rate lowers through epochs). With these, it was time to compile the model and hear I chose the adam optimizer with an initial learning rate of 0.001 and the learning rate scheduler. Further, the model's loss was determined using binary crossentropy. Lastly, the metrics used in the model were accuracy and area under the ROC curve which is a measure of both the true positive rate and the false positive rate of the classifier.

Then, I fit the model over 15 epochs with a batch size of 8 and here using callbacks, early stopping and learning rate scheduler were utilized. This model was fitted with the history module so that visualizations of the model could be created. Here, I utilized pyplot to measure the accuracy on training and validation data as well as the loss on training and validation data. These visualizations do indicate that overfitting has occurred. However, time did not permit for me to improve model performance past this level.



Source: https://www.tensorflow.org/text/tutorials/text_classification_rnn

https://www.tensorflow.org/text/guide/word_embeddings#:~:text=The%20Embedding%20layer%20takes%20the,batch%2C%20sequence%2C%20embedding)%20.

https://blog.paperspace.com/bidirectional-rnn-keras/

https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/

## Prediction and Model Submission

In order to create predictions, each value in the vectorized X_test which altered the shape of the array for each vector in the data, so that the model could make predictions on it. Then, these were appended to a list and were transformed to a pd dataframe in the manner of the sample submission. Then, utilizing the kaggle package, the data was submitted to Kaggle.

## Conclusion

In conclusion, this perhaps took me more time than is necessary for a mini-project. However, I did enjoy the project and it was very educational. We first learned the value of cleaning text and especially lemmatizing. Then, we learned the importance of word embedding. It was noteworthy that tfidf with random forest classifier achieved an accuracy nearly equivalent (or better) than a RNN with a classic count vectorizer. This could imply an improved NN architecture is needed (reasonable assertion) or that the tfidf word embedding just provides more context which adds to the value of NLP classification (also reasonable). Yet, regardless, each model was able to achieve adequate performance for a mini-project of this nature, an accuracy of 0.79773.

Highest NN score: 0.79773

Highest tfidf RF score: 0.78669
