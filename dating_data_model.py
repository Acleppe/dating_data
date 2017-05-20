import numpy as np
import pandas as pd
import xgboost as xgb

def load_data():
    """
    Load dating database, recast numeric columns into float.
    INPUT: None
    OUTPUT: DataFrame
    """
    df = pd.read_csv('pandas_dating_demo_df_anon.csv')
    df.drop('ID', inplace=True, axis=1)
    # correct dtypes
    for col in df.columns:
        if df[col].dtype.type not in [np.int64, np.float64]:
            if df.loc[2, col].split('.')[0].isnumeric():
                df[col] = df[col].astype(float)
    return df


def make_dummies(df):
    """
    Make dummy columns from categorical variables.
    INPUT: DF
    OUTPUT: DF with dummy cols concatenated.
    """
    for col in ['Hair', 'Attitude', 'Politics', 'Divorced', 'Income', 'Kids', 'Second_Date']:
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
        df.drop(col, inplace=True, axis=1)
        # df.drop(['Second_Date_No', 'Kids_No', 'Divorced_No'], inplace=True, axis=1)
    return df


def bin_age(df):
    """
    Age has a range over double that of any other numeric column, causing incorrect amount of 'information gain' with each split in a tree-based model.  Thus, must bin (into quintiles) to give fair balance in splitting.
    INPUT: DF
    OUTPUT: DF with Age now binned into quintiles, each roughly having equal membership (qcut)
    """
    df = pd.concat([df, pd.get_dummies(pd.qcut(df['Age'], 5), prefix='Age_Bins')], axis=1)
    age = df.pop('Age')
    return df, age


def feat_importances(df, mod):
    for idx, itm in enumerate(sorted(list(zip(mod.feature_importances_, df.columns.tolist())), reverse=True), 1):
        print("{r}. {feat}: {imp}".format(r=idx, feat=itm[1], imp=itm[0]))


if __name__ == '__main__':
    df = load_data()
    df = make_dummies(df)

    y = df.pop('Like_This_Person?').values
    X = df.values

    # Intentionally overfit the model b/c I only want to know feature_importances for existing data, not attempting to actually makes predictions (yet!)
    mod = xgb.XGBClassifier(n_estimators=500, learning_rate=.5, max_depth=4)
    mod.fit(X, y)

    # feats = list(zip(mod.feature_importances_, df.columns.tolist()))
    # sorted(feats, reverse=True)
    # df['Age'] = age

# yy = np.array([1 if x == 'Yes' else 0 for x in y])
# sns.jointplot(x=df['Intellectual_Connection'].values, y=yy, kind='reg', ratio=10)
# plt.xlabel()
# plt.ylabel()
# plt.show()



# Feature Importance
# [(0.60714287, 'Phys_Chemistry'),
#  (0.090225562, 'Humor'),
#  (0.067669176, 'Attraction'),
#  (0.052631579, 'Intellectual_Connection'),
#  (0.046992481, 'Politics_Left'),
#  (0.037593983, 'Second_Date_Yes'),
#  (0.031954888, 'Hair_Brunette'),
#  (0.02631579, 'Height(in.)'),
#  (0.02631579, 'Attitude_Positive'),
#  (0.0075187972, 'Income_Low'),
#  (0.0037593986, 'Hair_Blonde'),
#  (0.0018796993, 'Age_Bins_(27.6, 32.4]'),
#  (0.0, 'Politics_Right'),
#  (0.0, 'Politics_Independent'),
#  (0.0, 'Kids_Yes'),
#  (0.0, 'Income_Medium'),
#  (0.0, 'Income_High'),
#  (0.0, 'Hair_Red'),
#  (0.0, 'Divorced_Yes'),
#  (0.0, 'Attitude_Neutral'),
#  (0.0, 'Attitude_Negative'),
#  (0.0, 'Attitude_Complainer'),
#  (0.0, 'Age_Bins_[21, 25]'),
#  (0.0, 'Age_Bins_(35.2, 45]'),
#  (0.0, 'Age_Bins_(32.4, 35.2]'),
#  (0.0, 'Age_Bins_(25, 27.6]')]
