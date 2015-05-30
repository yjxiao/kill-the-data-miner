import pandas as pd
import os
import sys


def aggregate_naive(df):
    """naive way of aggregating counts for one enrollment_id
       Includes average counts, average of first order diff, and days
    """
    if isinstance(df, pd.Series):
        days_span = 1
        days_present = 1
        avg = df.drop('date')
        temp = pd.Series([df.date - pd.DateOffset()] + [0] * 9, index=df.index)
        avg_delta = pd.DataFrame([temp, df]).diff().mean().add_suffix('_delta')
    else:
        days_span = (df.date.iloc[-1] - df.date.iloc[0]).days + 1
        days_present = df.shape[0]
        avg = df.drop('date', axis=1).mean()
        avg_delta = df.diff().mean().add_suffix('_delta')
    res = avg.append(avg_delta).append(pd.Series([days_span, days_present],
                                                 index=['days_span',
                                                        'days_present']))
    res.date_delta = res.date_delta.days + res.date_delta.hours / 24.0
    return res


def read_and_aggr(in_path=os.curdir):
    """read counts file and aggregate to one vector for each enrollment_id"""
    fn = os.path.join(in_path, 'event_counts_nz.csv')
    df = pd.read_csv(fn, header=0, index_col=0, parse_dates=['date'])
    res = []
    l = len(df.index.unique())
    for i, e in enumerate(df.index.unique()):
        res.append(aggregate_naive(df.loc[e]))
        print 'finished {} of {}\r'.format(i+1, l),
    print 'finished {} of {}'.format(l, l)
    return pd.DataFrame(res, index=df.index.unique())


def main(args):
    if len(args) > 0 and os.path.exists(str(args[0])):
        data = read_and_aggr(str(args[0]))
    else:
        data = read_and_aggr()
    data.to_csv('aggregated_counts.csv', index_label='enrollment_id')
    print 'done'


if __name__ == '__main__':
    main(sys.argv[1:])
