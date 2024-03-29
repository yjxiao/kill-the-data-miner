import pandas as pd
import graphlab as gl
from scipy import stats
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import sys


def read_and_join(in_path=os.curdir, side_info=None):
    """Read in csv files and join tables by enrollment_id
    Args:
        in_path (str): path to the directory containing the csv files
        side_info (str): file name of side information (eg. aggregated
                         counts), assumed to be in the same directory.
    Returns:
        joined (DataFrame): joined table
    """
    fn_enroll = os.path.join(in_path, 'enrollment_train.csv')
    fn_truth = os.path.join(in_path, 'truth_train.csv')
    enrollment = pd.read_csv(fn_enroll, header=0, index_col=0)
    truth = pd.read_csv(fn_truth, header=None, names=['label'], index_col=0)
    joined = enrollment.join(truth, how='inner')
    if side_info and os.path.exists(os.path.join(in_path, side_info)):
        fn_side = os.path.join(in_path, side_info)
        side = pd.read_csv(fn_side, header=0, index_col=0)
        joined = joined.join(side, how='inner')
    return joined


def do_train_rand(train, valid, params=None, max_models=32):
    """Do randomized hyper-parameter search
    Args:
        train (SFrame): training set
        valid (SFrame): validataion set
        params (dict): parameters for random search
        max_models (int): maximum number of models to run
    Returns:
        res (SFrame): table of choices of parameters sorted by valid RMSE
    """
    if not params:
        params = {'user_id': ['username'], 'item_id': ['course_id'],
                  'target': ['label'], 'binary_target': [True],
                  'num_factors': stats.randint(4, 128),
                  'regularization': stats.expon(scale=1e-4),
                  'linear_regularization': stats.expon(scale=1e-7)}
    try:
        job = gl.toolkits.model_parameter_search \
                         .random_search.create((train, valid),
                                               gl.recommender.
                                               factorization_recommender.create,
                                               params, max_models=max_models)
        res = job.get_results()
        res = res.sort('validation_rmse')
        print 'Best params for random search are: {}'.format(res[0])
        res.save('rand_search.csv', format='csv')
    except:
        print job.get_metrics()
        res = None
    return res


def do_train_grid(train, valid, params=None):
    """Do grid search
    Args:
        train (SFrame): training set
        valid (SFrame): validataion set
        params (dict): parameters for grid search
    Returns:
        res (SFrame): table of choices of parameters sorted by valid RMSE
    """
    if not params:
        params = {'user_id': ['username'], 'item_id': ['course_id'],
                  'target': ['label'], 'binary_target': [True],
                  'num_factors': [2 ** i for i in range(3, 7)],
                  'regularization': [10 ** i for i in range(-7, -2)],
                  'linear_regularization': [10 ** i for i in range(-10, -5)]}
    try:
        job = gl.toolkits.model_parameter_search \
                         .grid_search.create((train, valid),
                                             gl.recommender.
                                             factorization_recommender.create,
                                             params)
        res = job.get_results()
        res = res.sort('validation_rmse')
        print 'Best params for grid search are: {}'.format(res[0])
        res.save('grid_search.csv', format='csv')
    except:
        print job.get_metrics()
        res = None
    return res


def do_train_single(train, valid=None, params=None):
    """Do single training process with given parameters and optionally
       test the result on validation set
    Args:
        train (SFrame): training set
        valid (SFrame): validataion set, set to None if no need to test
        params (dict): parameters for running model
    Returns:
        model (recommender obj): model fitted
    """
    if not params:
        params = {'user_id': 'username', 'item_id': 'course_id',
                  'target': 'label', 'binary_target': True,
                  'num_factors': 16,
                  'regularization': 0.0002,
                  'linear_regularization': 1e-6, 'max_iterations': 80}
    try:
        model = gl.recommender.factorization_recommender.create(train,
                                                                **params)
        model.save('model')
        if valid:
            print model.evaluate_rmse(valid, target='label')
            y = np.array(valid['label'])
            pred = np.array(model.predict(valid))
            print 'auc = {}'.format(roc_auc_score(y, pred))
    except:
        pass
    return model


def save_coefs_pred(model, data):
    """Save coefficients and predicted values to file"""
    coefs = model.get('coefficients')
    for k, v in coefs.items():
        if k != 'intercept':
            v.unpack('factors').save('{}_factors'.format(k), format='csv')
    temp = data.copy()
    temp['pred'] = np.array(model.predict(gl.SFrame(data)))
    temp.to_csv('pred.csv', index_label='enrollment_id')


def main(args):
    # check args, args[0] should be a directory containing csv files for train
    if len(args) > 0 and os.path.exists(str(args[0])):
        data = read_and_join(str(args[0]), side_info='aggregated_counts.csv')
    else:
        data = read_and_join(side_info='aggregated_counts.csv')
    train, valid = gl.recommender \
                     .util.random_split_by_user(gl.SFrame(data),
                                                user_id='username',
                                                item_id='course_id',
                                                item_test_proportion=0.2,
                                                max_num_users=None,
                                                random_seed=2345)
    # initialized params for randome search based on grid search results
    res = do_train_grid(train, valid)
    if not res:
        raise ValueError('No results returned from grid search')
    params_needed = ['user_id', 'item_id', 'target', 'binary_target',
                     'num_factors', 'regularization', 'linear_regularization']
    rand_params = {i: res[0][i] for i in params_needed}
    rand_params['num_factors'] = stats.randint(rand_params['num_factors'] / 2,
                                               rand_params['num_factors'] * 2)
    rand_params['regularization'] = stats \
        .expon(scale=rand_params['regularization'])
    rand_params['linear_regularization'] = stats \
        .expon(scale=rand_params['linear_regularization'])
    res = do_train_rand(train, valid, params=rand_params, max_models=96)
    if not res:
        raise ValueError('No results returned from random search')
    # test parameters on another random split
    train, valid = gl.recommender \
                     .util.random_split_by_user(gl.SFrame(data),
                                                user_id='username',
                                                item_id='course_id',
                                                item_test_proportion=0.2,
                                                max_num_users=None,
                                                random_seed=9876)
    params = {i: res[0][i] for i in params_needed}
    do_train_single(train, valid, params=params)
    # train final model and save coefs
    params = {i: res[0][i] for i in params_needed}
    params['max_iterations'] = 80
    model = do_train_single(gl.SFrame(data), params=params)
    save_coefs_pred(model, data)


if __name__ == '__main__':
    main(sys.argv[1:])
