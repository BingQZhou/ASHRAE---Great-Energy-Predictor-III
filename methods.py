import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def check_dependency(df, ref_col, other_col):
    """
    This method takes in a dataframe and two column names. Then it runs
    a permutation test and returns a p value on whether the two columns are
    dependent.
    """
    #observed value
    gpA = df.loc[df[ref_col].isnull(), other_col]
    gpB = df.loc[df[ref_col].notnull(), other_col]
    obs = ks_2samp(gpA, gpB).statistic

    #permutation
    copy = df.copy()
    perm_results = []
    for i in range(100):
        copy[ref_col] = df[ref_col].sample(frac = 1, replace = False).reset_index(drop = True)
        gpA = copy.loc[copy[ref_col].isnull(), other_col]
        gpB = copy.loc[copy[ref_col].notnull(), other_col]
        perm_results.append(ks_2samp(gpA, gpB).statistic)
    pval = np.mean(np.array(perm_results) >= obs)
    return pval

def fill_floor_count(row, dict):
    """
    This method fill nans in floor_count.
    """
    if np.isnan(row.loc['floor_count']):
        return dict[row.loc['site_id']]
    return row.loc['floor_count']

def fill_year_built(row, dict):
    """
    This method fill nans in year_built.
    """
    if np.isnan(row.loc['year_built']):
        return dict[row.loc['site_id']]
    return row.loc['year_built']

def select_with_lin(lin_reg, all_combined, y):
    """
    This method selects the best feature according to the linear model.
    """
    r_sqr = {}
    for feat in all_combined.columns:
        X = all_combined[feat]
        lin_reg.fit(np.array(X).reshape(-1, 1), y)
        r_sqr[feat] = lin_reg.score(all_combined[[feat]], y)
    best_feat = max(r_sqr, key = r_sqr.get)
    return (best_feat, r_sqr[best_feat])

def feat_engi_test(test_train, weather_train, building_meta):
    """
    This method takes in three dataframes and conduct feature selection and
    engineering.
    """
    copy = test_train.copy()
    weather_building = weather_train.merge(building_meta, on = 'site_id', how = 'left')
    all_combined = copy.merge(weather_building, on = ['building_id', 'timestamp'], how = 'left')
    all_combined = all_combined.dropna()
    #test_meter = all_combined['meter_reading']
    X_y = all_combined[['meter_reading', 'air_temperature', 'square_feet', 'sea_level_pressure', 'wind_direction', 'dew_temperature']].dropna()
    X = X_y[['air_temperature', 'square_feet', 'sea_level_pressure', 'wind_direction', 'dew_temperature']]
    test_meter = X_y['meter_reading']

    return X, test_meter

def process_test(test, weather, building):
    test = test.reset_index(drop = True)
    test['timestamp'] = pd.to_datetime(test['timestamp'], format = "%Y-%m-%d %H:%M:%S")
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], format = "%Y-%m-%d %H:%M:%S")
    return test, weather

def tree_reg_perf(X_train, y_train, X_test, y_test):
    result = []
    for i in range(21, 30):
        dtr = DecisionTreeRegressor(max_depth = i)
        dtr.fit(X_train, y_train)

        #train_err
        preds = dtr.predict(X_train)
        train_rmse = np.sqrt(np.mean((preds - y_train)**2))

        #test_err
        preds = dtr.predict(X_test)
        test_rmse = np.sqrt(np.mean((preds - y_test)**2))

        result.append([i, train_rmse, test_rmse])
    result = pd.DataFrame(result).set_index(0)
    result.columns = ['train_err', 'test_err']
    return result

# function for reducing df size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
