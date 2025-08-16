import pandas as pd
import sklearn
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer


#models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC



# Metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix




df = pd.read_csv(r"Restaurant_Reviews.tsv", delimiter = "\t")


import re       #regular expression
import nltk     # natural langage tool kit
nltk.download("stopwords")


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # stemming & PorterStemmer is used to bring words to their root words





a = stopwords.words("english")




ps = PorterStemmer()
    #porter stemmer, lanchaster stemming, snowball stemming
    # all of these are used for stemming and we are using portstemmer



comments = []


#using above concepts to generate 
for i in range(len(df)):
    reviews = re.sub("[^a-zA-Z]"," ", df["Review"][i]) 
    #to remove digits, puntuations, symbols, emojis etc from reviews
    #the purpose here is to replace all the digits, puntuations, symbols, emojis etc to be replaced by a space
    #re.cun(pattern, replace, string)
    #sub() replaces all parts of the string that match the given pattern with replacement.
    #in pattern, "^" denotes "NOT" hence include the values in range a-z and A-Z only and replace everything else with the replacement provided
    # in replace it is blank because we are not replacing anything
    #String = dataframe Review Liked column is passed to apply filter and replacement on it
    # spaces are vaid input hence carefully use them in here. 
    #if "" is used instead of " " then all the values(digits, puntuations, symbols, emojis etc) will be replaced by no space and reviews will be concatinated 

    reviews = reviews.lower()
    reviews = reviews.split()
    
    stop_words = set(stopwords.words("english"))
    
    d = [ps.stem(word) for word in reviews if word not in stop_words]
    #creating list comprehension to create a new list "d" containing only the words from reviews that are not in the list of English stopwords provided by NLTK.
    #d = [i for i in reviews if not i in (stopwords.words("english"))]
    #to save time convert it into set
    #d = [i for i in reviews if not i in set(stopwords.words("english"))]
    #ps.stem is taking each word from d and reducing it to it's stemmed form

    li = " ".join(d) 
    #ps.stem(j) → takes each word j from list d and reduces it to its root form.\
    #Example: "loved" → "love", "playing" → "play".
    #This helps group similar words so the model treats them as the same feature.
    #li = " ".join(rw)
    #Turns the list of stemmed words rw into a single sentence-like string separated by spaces.
    #Example: ["love", "food", "place"] → "love food place".
    #This is necessary because later steps (like CountVectorizer) expect a full sentence, not a list of words.
    
    comments.append(li)
    #appending changes
    
#bag of words

#converting words into binary
cv = CountVectorizer(max_features=1500)
#max_features=1500 means: When creating the Bag of Words representation, only keep the top 1,500 most frequent words in the dataset.
x = cv.fit_transform(comments).toarray() #its dimension is comments X root values

#saving x in dataframe
df2 = pd.DataFrame(data=x)
df2.to_excel(r'y.xlsx')

#processing all dimensions are not necessary if any column have only one or two apperances

#now to define answers
y = df.iloc[:,1].values





x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)
#y will have only one collumn hence it's dimension is (200,)




models = {
    "Knn": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p = 2),
    "MNB": MultinomialNB(),
    "SVC": LinearSVC()
    }

results = []

for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, prec, rec, f1))
    print(f"{name} -> ACC: {acc:.4f}, PREC: {prec:.4f}, REC:{rec:4f}, F1: {f1:.4f}")

#. → start of format specification
#4 → number of decimal places to show
#f → format as a floating-point number

best_model = sorted(results, key=lambda x: x[3], reverse=True)[0]
#sorting all the elements usinf sorted() acc to all 4 (acc,prec,rec,f1) parameters usnig x[3] 
#in descending order by enabling reverse=True
# and picking first one using [0] 

print(best_model[0])

KNeighborsClassifier().get_params()
MultinomialNB().get_params()
LinearSVC().get_params()

#hyperparameter tuning for all models
param = {
    "KNN":{
        "n_neighbors":[2,3,7,9,11,15],
        "metric": "minkowski",
        "p":[1,2]
            },
    "MNB":{
        "alpha": [0.1, 0.5, 1.0, 2.0]
        },
    "SVC":{
        "C": [0.01, 0.1, 1, 10],   
        "max_iter": [1000, 5000, 10000]       
        }
    }


best_model_name = best_model[0]

if best_model_name in param:
    print(f"\nRunning Hyperparameter Tuning for {best_model_name}...")
    model_class = type(models[best_model_name])
    grid = GridSearchCV(model_class(), param[best_model_name], cv=5, scoring="f1",)
    grid.fit(x_train, y_train)
    print("Best Parameters:", grid.best_params_)
    print("Best Cross-Validated F1:", grid.best_score_)

    best_model_tuned = grid.best_estimator_
    y_pred_tuned = best_model_tuned.predict(x_test)

    print("Tuned F1:", f1_score(y_test, y_pred_tuned))
else:
    print(f"No parameter grid defined for {best_model_name}.")

#KNN algo


#classifier = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p = 2)
#it is creating a K-Nearest Neighbors (KNN) classifier with specific parameters:
# n_neighbors=5 → The classifier will look at the 5 nearest training data points (neighbors) when making a prediction.
#metric="minkowski" → The distance metric used to measure closeness between points.
#p=2 → With Minkowski distance, p=2 means it will use Euclidean distance.
#If p=1, it would use Manhattan distance instead.
#📌 In short:
#This is setting up a KNN model that predicts a sample’s label based on the most common label among its 5 closest points, using Euclidean distance for measuring closeness.



# to test the prediction is correct or not we will use confusion metrix
cfm = confusion_matrix(y_test, best_model_tuned.predict(x_test))
print(cfm)
