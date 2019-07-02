#natural language processing

#importing librabries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
data = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t')

#cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
filtered_review = []

for i in range(0,len(data)):
    review =re.sub('[^a-zA-Z]', ' ' , data['Review'][i])                                       #removing word except alphabets
    review = review.lower()                                                                  #coverting to lower case to all
    review = review.split()                                                                        #dividing into list on bases of space
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]    #for removing preposotion and usless words and applyins stemmin that isconverting to single tense
    review = ' '.join(review)                                  #assemble list back tosiring with space
    filtered_review.append(review)


#forming a model of word
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)                                                      #cleaning text canbe done by countvectorizor
X=cv.fit_transform(filtered_review).toarray()
y=data.iloc[:,-1].values

#trainig and test set spliting
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y=train_test_split(X,y,test_size=0.2,random_state=0)


#fitting trainig set to learning model
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(train_X,train_y)

#predicting the test result
y_pred= classifier.predict(test_X)

#evaluating the result of model
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(test_y, y_pred)

score=accuracy_score(test_y, y_pred)*100


