import pandas as pd
import graphlab as gl
import os
import sys


def read_and_join(in_path=os.curdir):
    fn_enroll = os.path.join(in_path, 'enrollment_train.csv')
    fn_truth = os.path.join(in_path, 'truth_train.csv')
    enrollment = pd.read_csv(fn_enroll, header=0, index_col=0)
    truth = pd.read_csv(fn_truth, header=None, names=['label'], index_col=0)
    joined = enrollment.join(truth, how='inner')
    return joined


def do_train(df):
    train, valid = gl.recommender \
                     .util.random_split_by_user(df,
                                                user_id='username',
                                                item_id='course_id',
                                                item_test_proportion=0.2,
                                                max_num_users=None,
                                                random_seed=2345)
    params = {'user_id': ['username'], 'item_id': ['course_id'],
              'target': ['label'], 'binary_target': [True],
              'num_factors': [2 ** i for i in range(3, 9)],
              'regularization': [10 ** i for i in range(-11, -2)],
              'linear_regularization': [10 ** i for i in range(-13, -4)]}
    job = gl.toolkits.model_parameter_search \
                     .grid_search.create((train, valid),
                                         gl.recommender.
                                         factorization_recommender.create,
                                         params)
    print job.get_results()
    print job.get_best_params()
    return job


def main(args):
    if len(args) > 0 and os.path.exists(str(args[0])):
        data = gl.SFrame(read_and_join(str(args[0])))
    else:
        data = gl.SFrame(read_and_join())
    job = do_train(data)
    with open('best_params.log', 'wb') as f:
        f.write(str(job))


if __name__ == '__main__':
    main(sys.argv[1:])
