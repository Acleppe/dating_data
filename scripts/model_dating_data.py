import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

def load_data():
    """
    Load dating database, recast numeric columns into float.
    INPUT: None
    OUTPUT: DataFrame
    """
    df = pd.read_csv('../data/pandas_dating_demo_df_anon.csv')
    df.drop('ID', inplace=True, axis=1)

    ans = int(input("Would you like to include the 'Chemistry' variable or not? \
    \n 1. Yes \
    \n 2. No \
    \n Please enter the number of your choice: "))

    if ans == 2:
        df.drop('Chemistry', inplace=True, axis=1)

    # correct dtypes if needed -- sometimes the CSV has a weird conversion issue.
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
    """
    Print the relative feature importances of the model.
    INPUT: df, model (fitted)
    OUTPUT: Console output of feature importances in descending order
    """
    print("")
    for idx, itm in enumerate(sorted(list(zip(mod.feature_importances_, df.columns.tolist())), reverse=True), 1):
        print("{r}. {feat}: {imp:.1f}%".format(r=idx, feat=itm[1], imp=itm[0] * 100))


def plot_pairs(df, col1, col2):
    """
    Plot any two features against the binary version of the "Like this Person?" target column, with a simple regression line and no confidence intervals (confidence intervals aren't valid with a binary dependent variable in an OLS linear regression).  Purpose is simple to gauge general trend b/w vars and target.
    INPUT: df, two columns of interest
    OUTPUT: scatter matrix
    """
    sns.pairplot(data=df, x_vars=[col1, col2], y_vars=['Like_Binary'], kind='reg', size=6, plot_kws=dict(ci=0))
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_voters(dfv):
    """
    Take Voters DF and plot it in a basic bar chart, segmented by party.
    INPUT: Voter DF
    OUTPUT: Bar plot
    """
    dfv = pd.read_csv('../data/voters.csv')
    fig = plt.figure(figsize=(10, 10))
    sns.barplot(data=dfv, x='County', y='Voters', hue='Party', saturation=.7, palette='muted')
    plt.title("Registered Voters by Party per County in Colorado, May 2017")
    plt.xlabel("County")
    plt.ylabel("Voters")
    plt.tight_layout()
    plt.show()
    plt.close()


def get_logit_coef(df, col):
    """
    Get coefficient from logit model.
    INPUT: DF, variable of choice to regress on
    OUTPUT: coefficient (float)
    """

    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(df[col].values.reshape(-1, 1), df['Like_Binary'].values.ravel())
    return clf.coef_


def plot_attitude_counts(df):
    """
    Group by Attitude then Income, so we get the breakdown of each Income for every type of Attitude.
    INPUT: df
    OUTPUT: Bar plot of the Attitude and Income breakdowns.
    """
    dfa = pd.read_csv('../data/pandas_dating_demo_df_anon.csv')
    # Index into grouped df with 'Age' just to return a single column.
    dfa = dfa.groupby(['Attitude', 'Income']).count()['Age']
    plot_dict = dict()
    for outter, inner in dfa.index:
        plot_dict["{out}_{inn}".format(out=outter, inn=inner)] = dfa.loc[outter, inner]

    l, m, h = 0, 6, 12
    fig = plt.figure(figsize=(10, 10))
    delta = 0
    for tude in ['Positive', 'Neutral', 'Negative', 'Complainer']:
        plt.bar(np.array([l, m, h])+delta, [plot_dict.get(tude+'_Low', 0), plot_dict.get(tude+'_Medium', 0), plot_dict.get(tude+'_High', 0)], alpha=.8, label=tude)
        delta += 1
    plt.xticks([1.5, 7.5, 13.5], ['Low', 'Med', 'High'])
    plt.yticks([n for n in range(10)])
    plt.title("Income Grouped by Attitude")
    plt.xlabel("Income")
    plt.ylabel("Number of People")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()
    plt.close()


def make_voter_df():
    """
    Statistics taken from May 2017 Voter registration information.  Defunct function because replaced with a csv.  Didn't delete because I might use in another plot or some such...
    INPUT: None
    OUTPUT: DF of voter registration infor for plotting.
    """
    voters = {
        'Boulder': {'D': 96024, 'R': 36628, 'I': 77453},
        'Broomfield': {'D': 13200, 'R': 12583, 'I': 16738},
        'Denver': {'D': 196418, 'R': 58246, 'I': 136332},
        'Arapahoe': {'D': 125447, 'R': 107519, 'I': 129540},
        'Larimer': {'D': 63789, 'R': 72790, 'I': 86850},
        'Weld': {'D': 38411, 'R': 63204, 'I': 59652},
        'Adams': {'D': 86898, 'R': 62006, 'I': 89643},
        'Jefferson': {'D': 117494, 'R': 117197, 'I': 142994}
    }
    return pd.DataFrame(voters)


if __name__ == '__main__':
    df = load_data()
    df = make_dummies(df)
    df, age = bin_age(df)
    for col in [c for c in df.columns if 'Hair' in c]:
        df.drop(col, inplace=True, axis=1)
    y = df.pop('Like_This_Person?').values
    X = df.values

    ## Intentionally overfit the model b/c I only want to know feature_importances for existing data, not attempting to actually makes predictions (yet!)
    mod = xgb.XGBClassifier(n_estimators=500, learning_rate=.5, max_depth=4)
    mod.fit(X, y)
    feat_importances(df, mod)

    ## Add Binary version of target col back for plotting Logit and EDA
    df['Like_Binary'] = np.array([1 if x == 'Yes' else 0 for x in y])
    df['Age'] = age
    # get_logit_coef(df, 'Age')
    # plot_pairs(df)
    # plot_voters(dfv)
