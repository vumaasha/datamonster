
from datetime import datetime

def print_log(msg):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")
    print(time_now+ "\t" + msg)


from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def validate_model(X, Y, classifier, split_generator=lambda Y: StratifiedKFold(Y, n_folds=3)):
    for train_index, test_index in split_generator(Y):
        X_train, Y_train = X.iloc[train_index], Y.iloc[train_index]
        X_test, Y_test = X.iloc[test_index], Y.iloc[test_index]
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        print(classification_report(Y_test, Y_test))
        print(confusion_matrix(Y_test, Y_test))
        print('='*80)