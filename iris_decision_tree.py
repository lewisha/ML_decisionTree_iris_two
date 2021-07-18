from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier, export_graphviz



def decision_iris():
    
    # 1）get iris data
    iris = load_iris()

    # 2）split
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=22)

    # 3）estimator
    estimator = DecisionTreeClassifier(criterion="entropy")
    estimator.fit(x_train, y_train)

    # 4）Model Evaluation
    # Method One
    y_predict = estimator.predict(x_test)
    print("y_predict:\n", y_predict)
    print(y_test == y_predict)

    # Method two
    score = estimator.score(x_test, y_test)
    print("The accuracy is：\n", score)

    # visualization
    export_graphviz(estimator, out_file="iris_tree.dot", feature_names=iris.feature_names)

    return None

if __name__ == "__main__":
   
    decision_iris()
