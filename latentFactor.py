import pandas as pd
import graphlab as gl
from scipy import stats
import os
import sys


def read_and_join(in_path=os.curdir):
    fn_enroll = os.path.join(in_path, 'enrollment_train.csv')
    fn_truth = os.path.join(in_path, 'truth_train.csv')
    enrollment = pd.read_csv(fn_enroll, header=0, index_col=0)
    truth = pd.read_csv(fn_truth, header=None, names=['label'], index_col=0)
    joined = enrollment.join(truth, how='inner')
    return joined


def do_train_rand(train, valid):
    params = {'user_id': ['username'], 'item_id': ['course_id'],
              'target': ['label'], 'binary_target': [True],
              'num_factors': stats.randint(4, 512),
              'regularization': stats.expon(scale=1.0/100000),
              'linear_regularization': stats.expon(scale=1.0/100000000)}
    try:
        job = gl.toolkits.model_parameter_search \
                         .random_search.create((train, valid),
                                               gl.recommender.
                                               factorization_recommender.create,
                                               params)
        res = job.get_results()
        res = res.sort('validation_rmse')
        print res[0]
        res.save('train_history.csv', format='csv')
    except:
        print job.get_metrics()
    return job


def do_train_grid(train, valid):
    params = {'user_id': ['username'], 'item_id': ['course_id'],
              'target': ['label'], 'binary_target': [True],
              'num_factors': [2 ** i for i in range(3, 7)],
              'regularization': [10 ** i for i in range(-6, -1)],
              'linear_regularization': [10 ** i for i in range(-10, -5)]}
    try:
        job = gl.toolkits.model_parameter_search \
                         .grid_search.create((train, valid),
                                             gl.recommender.
                                             factorization_recommender.create,
                                             params)
        res = job.get_results()
        res = res.sort('validation_rmse')
        print res[0]
        res.save('train_history.csv', format='csv')
    except:
        print job.get_metrics()
    return job


def main(args):
    if len(args) > 0 and os.path.exists(str(args[0])):
        data = gl.SFrame(read_and_join(str(args[0])))
    else:
        data = gl.SFrame(read_and_join())
    train, valid = gl.recommender \
                     .util.random_split_by_user(data,
                                                user_id='username',
                                                item_id='course_id',
                                                item_test_proportion=0.2,
                                                max_num_users=None,
                                                random_seed=2345)

    do_train_grid(train, valid)


if __name__ == '__main__':
    main(sys.argv[1:])
