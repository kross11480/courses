import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

def update_unknown_to_nan(df, feat_info):
    '''convert unknowns in each feature as given in feat_info to np.nan
        - E.g. in CAMEO_INTL_2015, -1 and X denote missing or unknown values and are replaced with np.nan
    
    Args:
        df: data frame to be cleaned
        feat_info : feature information
        
    Returns:
        df : data frame where all unknowns are converted to np.nan
    '''
    feat_info_l = feat_info[~feat_info['attribute'].isin(['CAMEO_DEUG_2015', 'CAMEO_DEU_2015','CAMEO_INTL_2015'])]    
    for col in feat_info_l.attribute.tolist():
        if col in df.columns.values:
            missing_unknown_vals = eval(
                feat_info.loc[feat_info.attribute == col,'missing_or_unknown'].values[0])
            di = dict.fromkeys(missing_unknown_vals, np.nan)
            df[col].replace(di, inplace = True)
    
    di = {-1:np.nan, 'X': np.nan}
    df['CAMEO_DEUG_2015'].replace(di, inplace = True)
    di = {-1:np.nan, 'XX': np.nan}
    df['CAMEO_INTL_2015'].replace(di, inplace = True)
    return df


def plot_miss_val(df, top_n = 20):
    '''plot percentage of missing values for each feature in data frame.
        - the plot is sorted from highest missing percentage to lowest.
        - show top_n features with highest percentage of missing data.
     
    
    Args:
        df: data frame whose features are to be visualized for percentage of missing data
        top_n: the number of features to be visualized in plot
        
    Returns:
    '''
    df_numnull = df.isnull().sum()
    #print (mailout_train_numnull.sort_values(ascending=False).to_string())
    df_null_percent = df_numnull / len(df) * 100
    (df_null_percent.sort_values(ascending=False)[:top_n].plot(kind='bar', figsize=(20,5), fontsize=12))
    plt.ylabel('Percentage of Missing/Unknown Data')
    return

def plot_compare_feat(column, df_cust, df_pop):
    '''for a feature show its distribution both in general population dataset and customer dataset
        - gives an idea if feature is helpful in model
    
    Args:
        column: feature(s) to be visualized as customer and population distribution
        df_cust: data frame containing customer data
        df_pop: data frame containing population data
        
    Returns:
    '''
    sns.set(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(figsize=(12,4), ncols=2)
    sns.countplot(x=column, data=df_cust, ax=ax1, palette="dark")
    ax1.set_xlabel('Value')
    ax1.set_title(f'Customer Distribution ')
    sns.countplot(x=column, data=df_pop, ax=ax2, palette="dark")
    ax2.set_xlabel('Value')
    ax2.set_title(f'Population Distribution ')
    fig.suptitle(f'Feature: {column}')
    return

def get_decade(x):
    '''for a feature PRAEGENDE_JUGENDJAHRE obtain the code for dummy var
        - used as lambda function
        - helps in obtaining dummy variable PRAEGENDE_JUGENDJAHRE_DEKADE
    Args:
        x: value of feature
        
    Returns: return value (DEKADE) given x as key
    '''
    decade_dict = {'1': 1, '2': 1, '3': 2, '4': 2, '5': 3, '6': 3,'7': 3,'8': 4,
                   '9': 4,'10': 5, '11': 5, '12': 5, '13': 5, '14': 6, '15': 6}
    if not np.isnan(x):
        key = str(int(x))
        return decade_dict[key] if key in decade_dict else np.nan

def get_movement(x):
    '''for a feature PRAEGENDE_JUGENDJAHRE obtain the code for dummy var
        - used as lambda function
        - helps in obtaining dummy variable PRAEGENDE_JUGENDJAHRE_BEWEGUNG
    Args:
        x: value of feature
        
    Returns: return value (movement) given x as key
    '''
    decade_dict = {'1': 1, '2': 0, '3': 1, '4': 0, '5': 1, '6': 0,'7': 0,'8': 1,
                   '9': 0,'10': 1, '11': 0, '12': 1, '13': 0, '14': 1, '15': 0}
    if not np.isnan(x):
        key = str(int(x))
        return decade_dict[key] if key in decade_dict else np.nan


def get_standard(x):
    '''for a feature CAMEO_INTL_2015_STANDARD obtain the code for dummy var
        - used as lambda function
        - helps in obtaining dummy variable CAMEO_INTL_2015_STANDARD
    Args:
        x: value of feature
        
    Returns: return value (standard) given x as key
    '''
    if pd.isnull(x):
        return np.nan
    standard_dict =  {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    return standard_dict[x[-2]]
    
def get_phase(x):
    '''for a feature CAMEO_INTL_2015_STANDARD obtain the code for dummy var
        - used as lambda function
        - helps in obtaining dummy variable CAMEO_INTL_2015_LEBENSPHASE
    Args:
        x: value of feature
        
    Returns: return value (life phase)
    '''
    if pd.isnull(x):
        return np.nan 
    standard_dict =  {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5}
    return standard_dict[x[-1]]

def preprocessing(df):
    '''handles the features of object data type
        - drop CAMEO_DEU_2015 similar value in another feature
        - for EINGEFUEGT_AM extract and replace with year instead of exact date
        - for OST_WEST_KZ replace with binary variables
        - drop LNR
        - for PRAEGENDE_JUGENDJAHRE feature create dummy variables decade and movement
        - for CAMEO_INTL_2015 feature create dummy variables standard and life phase
        - convert D19_LETZTER_KAUF_BRANCHE feature values to categorical codes
    
    Args:
        df: input dataframe
        
    Returns:
        df: data frame where object types are preprocessed
    '''
    
    if 'CAMEO_DEU_2015' in df.columns.values:
        df.drop([ 'CAMEO_DEU_2015'], axis=1, inplace=True)
    df["EINGEFUEGT_AM"] = pd.to_datetime(df["EINGEFUEGT_AM"], format='%Y/%m/%d %H:%M')
    df["EINGEFUEGT_AM"] = df["EINGEFUEGT_AM"].dt.year
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map({'O':1,'W':2})
    key = 'LNR'
    if key in df.columns.values:
        df.drop(key, axis=1, inplace=True)
        
    key = 'PRAEGENDE_JUGENDJAHRE'
    if key in df.columns.values:
        df['PRAEGENDE_JUGENDJAHRE_DEKADE'] = df[key].apply(get_decade)
        df['PRAEGENDE_JUGENDJAHRE_BEWEGUNG'] = df[key].apply(get_movement)
        df.drop('PRAEGENDE_JUGENDJAHRE', axis=1, inplace=True)
    
    key = 'CAMEO_INTL_2015'
    if key in df.columns.values:
        df['CAMEO_INTL_2015_STANDARD'] = df['CAMEO_INTL_2015'].apply(get_standard) 
        df['CAMEO_INTL_2015_LEBENSPHASE'] = df['CAMEO_INTL_2015'].apply(get_phase)
        df.drop('CAMEO_INTL_2015', axis=1, inplace=True)
    df.D19_LETZTER_KAUF_BRANCHE = pd.Categorical(df.D19_LETZTER_KAUF_BRANCHE)
    df['D19_LETZTER_KAUF_BRANCHE'] = df['D19_LETZTER_KAUF_BRANCHE'].cat.codes
    df.CAMEO_DEUG_2015 = pd.Categorical(df.CAMEO_DEUG_2015)
    return df
    
def preprocess_pipeline(df_in, cols):
    '''performs the following steps
        - imputation: trying out both most frequent and fill with zero to replace missing vals
        - encoding: encode features in categorical attribure list (cols)
        - scaling: using standard scaler
    Args:
        df_in: input dataframe
        cols: contains variables which are categorical in nature and should be encoded with dummy vars
        
    Returns:
        scaler: scaler is returned, as in unsupervised learning inverse transform is done.
        df_impute_encoded_scaled: data frame where input data frame is imputed, encoded, and scaled
    '''
    
    #impute missing values    
    imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
    #imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    df_i = imputer.fit_transform(df_in)
    df_i = pd.DataFrame(df_i, columns=df_in.columns)
    
    # one-hot encdoding
    df_e = pd.get_dummies(df_i, columns=cols) #, columns=categorical_attr_list
    
    #scaling
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_e)#.astype(float)
    df_impute_encoded_scaled = pd.DataFrame(data=scaled, index=df_e.index, columns=df_e.columns) 

    return scaler, df_impute_encoded_scaled

def drop_col(df, cols_drop):
    ''' drop certain columns from dataframe
        - used to drop columns with high percentage of missing data
    Args:
        df: input dataframe
        cols_drop: list of features whose corresponding columns in data frame is dropped.
        
    Returns:
        df_opt: dataframe after dropping features.
    '''
    for col in cols_drop:
        if col not in df.columns.values:
            return
    print('dataset shape before dropping', df.shape)
    df_opt = df.drop(columns=cols_drop, inplace = False)
    print('dataset shape after dropping', df_opt.shape)
    return df_opt