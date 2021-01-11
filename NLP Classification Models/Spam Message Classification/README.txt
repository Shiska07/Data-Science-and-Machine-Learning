This program uses Natural Language Processing to classify a message as 'spam' or 'ham'.

- It uses the text in the message to extract length of the message and saves it as a new attribute.
- Text preprocessing is done by removing punctuations and stopwords.
- CountVectorizer is used to create a 'Bag of Words' model transformer.
- The transformer is applied on the entire dataset to obtain a word count matrix for eah message data. 
- TfidfTransformer is used to transform binary data into Tfidf values. 
- This data is used to train a Multinomial Naive Bayes model, which is used for prediction of the test dataset.