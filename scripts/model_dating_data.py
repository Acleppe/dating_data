import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

def load_data():
    """
    Load dating database, recast numeric columns into float.
    INPUT: None
    OUTPUT: DataFrame
    """
    df = pd.read_csv('../data/jp_dating_df_anon.csv')
    df.drop(['ID', 'Be_Friends?'], inplace=True, axis=1)

    ans = int(input("Would you like to include the 'Chemistry' variable or not? \
    \n 1. Yes \
    \n 2. No \
    \n Please enter the number of your choice: "))

    if ans == 2:
        df.drop('Chemistry', inplace=True, axis=1)

    # correct dtypes if needed -- sometimes the CSV has a weird dtype issue.
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


def bin_height(df):
    """
    I modified some rankings in the database and now Height is overwhelmingly dominating the importances.  I think there's something wonky going on, so I'm binning it just for peace of mind to see if there's any clarity.  Sigh.
    INPUT: DF
    OUTPUT: DF with Height now binned into quintiles, each roughly having equal membership (qcut)
    """
    df = pd.concat([df, pd.get_dummies(pd.qcut(df['Height(in.)'], 5), prefix='Ht_Bins')], axis=1)
    height = df.pop('Height(in.)')
    return df, height


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


def plot_heights(df):
    """
    Plot heights of people in the database, with people I "liked" enough to date being highlighted by a slightly larger and brighter blue circle.  Midpoint of the range of heights is marked by a dashed line.
    INPUTE: df
    OUTPUT: plot of heights
    """
    hue = 'DarkBlue'
    like = mlines.Line2D([], [], color='SkyBlue', markersize=8, marker='o', ls='', lw=0, mec=hue, mew=1, label='Liked')
    no_like = mlines.Line2D([], [], color=hue, markersize=8, marker='o', ls='', lw=0, mec=hue, mew=1, label="Didn't Like")
    mask = df['Like_This_Person?'] == 'Yes'
    colors = ['SkyBlue' if itm else hue for itm in mask]
    sizes = [85 if itm else 45 for itm in mask]

    fig = plt.figure(figsize=(10,8))
    hline = plt.axhline(67, ls=':', lw=1, color='k', label='Midpoint')
    plt.plot(df['Height(in.)'], color=hue, lw=2.5)
    plt.scatter(np.arange(df.shape[0]), df['Height(in.)'], s=sizes, c=colors, marker='o', linewidths=1, edgecolor=hue, zorder=5)
    plt.yticks([x for x in range(60, 75)],
        [str(divmod(x, 12)[0]) + ', ' + str(divmod(x, 12)[1]) for x in range(60, 75)])
    plt.title("Height of Everyone in the Database")
    plt.xlabel("Person")
    plt.ylabel("Height (ft, in)")
    plt.legend(handles=[like, no_like, hline], labels=[like.get_label(), no_like.get_label(), hline.get_label()], loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()


def plot_venn():
    """
    Plot basic Venn diagram of the intersection of people I "liked" romantically and people I felt would be a good friend.
    INPUT: None
    OUTPUT: Venn diagram
    """
    like = mlines.Line2D([], [], color='r', markersize=8, marker='o', ls='', lw=0, alpha=.5, label='Liked')
    friend = mlines.Line2D([], [], color='b', markersize=8, marker='o', ls='', lw=0, alpha=.5, label="Friends")
    # Radii set to approximate relative size of each group
    like_circle = patches.Circle((0.35, 0.5), 0.25, color='r', fill=True, alpha=.5)
    friend_circle = patches.Circle((0.65, 0.5), 0.29, color='b', fill=True, alpha=.5)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, aspect='equal')
    for p in [like_circle, friend_circle]:
        ax.add_patch(p)

    for center in zip([like_circle.center, friend_circle.center, (.5, .5)], ['21', '25', '16'], ['right', 'left', 'center'], ['k', 'k', 'w']):
        ax.annotate(s=center[1], xy=center[0], size=25, ha=center[2], color=center[3])

    plt.xticks([])
    plt.yticks([])
    plt.title("Venn Diagram of People Liked and Just Friends")
    plt.legend(handles=[like, friend], labels=[like.get_label(), friend.get_label()])
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


def get_logit_coef(df, col, targ='Like_This_Person?'):
    """
    Get coefficient from logit model.
    INPUT: DF, variable of choice to regress on
    OUTPUT: coefficient (float)
    """
    targ_binary = np.array([1 if itm == 'Yes' else 0 for itm in df[targ]])
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-3)
    clf.fit(df[col].values.reshape(-1, 1), targ_binary.ravel())
    return clf.coef_[0]


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

    ## Intentionally overfit the model b/c I only want to know feature_importances for existing data, not attempting to actually makes predictions.
    mod = xgb.XGBClassifier(n_estimators=500, learning_rate=.5, max_depth=4)
    mod.fit(X, y)
    feat_importances(df, mod)

    ## Add Binary version of target col back for plotting Logit and EDA
    # df['Like_Binary'] = np.array([1 if x == 'Yes' else 0 for x in y])
    # df['Age'] = age
    # get_logit_coef(df, 'Age')
    # plot_pairs(df)
    # plot_heights(df)
    # plot_venn()
    # plot_voters(dfv)
    # plot_attitude_counts(df)
