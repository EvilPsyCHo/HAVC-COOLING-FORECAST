# coding:utf8
# @Time    : 19-7-8 下午4:07
# @Author  : evilpsycho
# @Mail    : evilpsycho42@gmail.com
import argparse
from chinese_calendar import is_holiday
import pandas as pd
import pickle


o1 = ['o1_temp', 'o1_cold_h24',
      'weekday', 'is_holiday', 'month', 'hour', 'dry_temp']

o2 = ['o2_temp', 'o2_cold_h24',
      'weekday', 'is_holiday', 'month', 'hour', 'dry_temp']

cw = ['cw_temp', 'cw_cold_h24',
      'weekday', 'is_holiday', 'month', 'hour', 'dry_temp']

ce = ['ce_temp', 'ce_cold_h24',
      'weekday', 'is_holiday', 'month', 'hour', 'dry_temp']


def add_feature(x):
    x['is_holiday'] = x['时间'].apply(is_holiday).astype(int)
    x['month'] = x['时间'].dt.month
    x['hour'] = x['时间'].dt.hour
    x['weekday'] = x['时间'].dt.weekday
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', help='data file path', type=str)
    parser.add_argument('-r', '--result', help='result file path', type=str)
    args = parser.parse_args()
    data = pd.read_csv(args.file, parse_dates=['时间'])
    data = add_feature(data)

    o1_model = pickle.load(open('./model/o1.pkl', 'rb'))
    o2_model = pickle.load(open('./model/o2.pkl', 'rb'))
    cw_model = pickle.load(open('./model/cw.pkl', 'rb'))
    ce_model = pickle.load(open('./model/ce.pkl', 'rb'))

    if 'o1_temp' not in data.columns:
        data['o1_temp'] = 24.5
        data['o2_temp'] = 24.5
        data['cw_temp'] = 26.
        data['ce_temp'] = 26.

    o1_preds = o1_model.predict(data[o1])
    o2_preds = o2_model.predict(data[o2])
    cw_preds = cw_model.predict(data[cw])
    ce_preds = ce_model.predict(data[ce])

    result = pd.DataFrame({
        '时间': data['时间'].values,
        'o1': o1_preds,
        'o2': o2_preds,
        'cw': cw_preds,
        'ce': ce_preds,
        'total': o1_preds + o2_preds + cw_preds + ce_preds,
    })

    result.to_csv(args.result, index=None)
    print(f'success, result save in {args.result}')