import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from scipy import stats


def read_file(infile, is_train=True):
    data = pd.read_csv(infile, header=0, index_col=0)
    data = data.drop(['username', 'course_id'], axis=1)
    res = {}
    res['y'] = data.label.values
    res['X'] = data.drop('label', axis=1).values
    return res


def do_split(X, y):
    return train_test_split(X, y, test_size=0.2)


def shuffle(X, y):
    idx = range(len(X))
    np.random.shuffle(idx)
    return X[idx], y[idx]


def normalize(tr_X):
    mean = tr_X.mean(0)
    std = tr_X.std(0)
    assert((std != 0).all())
    tr_X = (tr_X - mean) / std
    return tr_X


def do_test(clf, X, y):
    pred = clf.predict_proba(X)[:, 1].ravel()
    return roc_auc_score(y, pred)


def do_train_grid(X, y, params=None):
    if not params:
        params = {'n_estimators': [50, 100], 'max_depth': range(12, 35),
                  'min_samples_leaf': range(70, 150, 6),
                  'max_features': ['auto', 'sqrt']}
    clf = GridSearchCV(GradientBoostingClassifier(), params, scoring=do_test,
                       n_jobs=15, verbose=1, cv=4)
    clf.fit(X, y)
    return clf


def do_train_rand(X, y, params=None, n_iter=50):
    if not params:
        params = {'n_estimators': stats.randint(40, 90),
                  'max_depth': stats.randint(20, 40),
                  'min_samples_leaf': stats.randint(80, 110),
                  'max_features': ['auto', 'sqrt']}
    clf = RandomizedSearchCV(GradientBoostingClassifier(), params,
                             n_iter=n_iter, scoring=do_test, n_jobs=15,
                             verbose=1, cv=4)
    clf.fit(X, y)
    return clf


def save_result(clf, frm):
    hist = clf.grid_scores_
    container = {i: [] for i in hist[0][0].keys()}
    container['valid_auc'] = []
    for i in hist:
        for k, v in i[0].items():
            container[k].append(v)
        container['valid_auc'].append(i[1])
    hist = pd.DataFrame.from_dict(container).sort('valid_auc', ascending=False)
    hist.to_csv('{}_history.csv'.format(frm), index=False)


def main():
    data = read_file('/home/jq401/kddcup/gradboost/train.csv')
    print 'finished reading data'
    X, y = shuffle(data['X'], data['y'])
    # grid search
    clf = do_train_grid(X, y)
    print 'finished grid search'
    auc = clf.best_score_
    print 'valid auc = {}'.format(auc)
    print 'best params is {}'.format(clf.best_params_)
    save_result(clf, 'grid')
    print 'done saving'
    # random search
    params = clf.best_params_
    for k, v in params.items():
        if k != 'max_features':
            params[k] = stats.randint(v / 2, v * 2)
    clf = do_train_rand(X, y, n_iter=128)
    print 'finished random search'
    auc = clf.best_score_
    print 'valid auc = {}'.format(auc)
    print 'best params is {}'.format(clf.best_params_)
    save_result(clf, 'rand')
    print 'done saving'


if __name__ == '__main__':
    main()
