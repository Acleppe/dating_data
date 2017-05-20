# My So-Called (Dating) Life

As part of my helping teach the "_Data Science: Intro to Python_" course, I made a database of some of my dating history.  It was used to introduce new students to Pandas -- see the accompanying [Jupyter Notebook](/intro_to_pandas_jpw_lecture.ipynb) for the actual lecture -- and I then wondered about the importance of factors in my own personal affinity for dating certain people versus others.  So, I quickly built a model to do a little investigating.  

## Table of Contents
1. [The Data-base Breakdown](#the-data---base-breakdown)
    + [Feature Definitions](#feature-defintions)
2. [Super Model](#super-model)
    + [Data Prep](#data-prep)
3. [The First Time](#the-first-time)
    + [Feature Importance](#first-run-feature-importance)
    + [Relative Correlations for Age vs Attraction](#relative-correlations-for-age-vs-attraction)
    + [Logit Coefficients](#logit-coefficients)
4. [Age is Just a Number](#age-is-just-a-number)
    + [Binning](#binning)



### The Date-abase Breakdown
My dating database had the following variables:

Variable                |  Type       | Values
------------------------|-------------|------
Age                     | Numeric     | 21 - 45
Height (in)             | Numeric     | 60 - 74
Attraction              | Numeric     | 0 - 10
Intellectual Connection | Numeric     | 0 - 10
Humor                   | Numeric     | 0 - 10
Chemistry*              | Numeric     | 0 - 10
Hair                    | Categorical | Blonde, Brunette, Red
Life Attitude           | Categorical | Positive, Neutral, Negative, Complainer
Politics                | Categorical | Left, Right, Independent
Income                  | Categorical | Low, Medium, High
Divorced                | Binary      | Yes, No
Kids                    | Binary      | Yes, No
Second Date?            | Binary      | Yes, No
Did I Like This Person? | Binary      | Yes, No

<sub> __Table 1:__ The variables I assigned to each person in the database. Note the asterisk next to _Chemistry_.  This denotes that it was added only after an initial run without it included.</sub>

<BR>

##### Feature Definitions
Let me quickly clarify how certain variables were assigned.  
+ __Attraction:__ Physical attraction only.
+ __Intellectual Connection:__ Basically the mental equivalent of attraction.
+ __Humor:__ How well our senses of humor aligned; did I laugh a lot with this person?
+ __Chemistry:__ The level of physical chemistry.
+ __Life Attitude:__ This is an appraisal of the person's general attitude in life.  
    + __Positive__: Generally optimistic, able to handle changes (even minor things like change of plans on a   date) well.  If the waiter makes a small mistake on their order, they just take off the unwanted food or, if necessary to address it, do so politely and don't mention it again.  Has a good view of the "big picture."  
    + __Neutral__: Depends on the mood or situation.  More of a 'nondescript' attitude.
    + __Negative__: Generally pessimistic.  Sees obstacles as larger than they are, lets little things bother them.  Would say things like "I don't want to go to [Place X] because they messed up my order one time."
    + __Complainer__: A specific sub-set of _Negative_, in that I found the person to be routinely not just negative in a "pessimistic about life" sense, but to explicitly complain about many things and people.  If the waiter made a mistake with their order, they would let it impact their ability to enjoy the meal, be fairly snarky to the waiter while demanding a correction, and finally -- and this is the _coup de grâce_ -- they would mention it as the first thing when someone asked about the date.  Like, who cares?  Your order wasn't perfect.  Wow.  Big deal.  Get a life.  How on earth these people can find any happiness in life, I will never know.
+ __Income:__ Since I don't have the tax returns of everyone in this database, I don't know exactly where they fall in income.  Things such as vocation, if they owned a home, neighborhood they lived in, type of car, cost of hobbies, and clothing and jewelry.  It is imperfect but is likely representative to a fair degree.  So I made the following rough annual income approximations.
    + __Low__: less than $40,000
    + __Medium__: between $40,000 and $80,000
    + __High__: above $80,000
+ __Second Date:__ Whether I saw this person a second time or not.  On the surface this seems like it would be an easy winner for most predictive variable -- if I "liked" someone then surely I'd see them again, right?  Well, life isn't always so cut-and-dried.  Sometimes circumstances change.  Sometimes you like someone that doesn't like you.  Sometimes you met only by chance to begin with.  Hence the inclusion of this feature.
+ __Did I Like This Person?:__ At the end of the day, whether I would consider full-on dating this person or not.  Ultimately, this is the target of the whole endeavor -- to find someone you'd like to date and, in my case, marry.

<BR>

### Super Model
My main interest in this mini-project was simple: to see which variables had the strongest relationship to predicting whether or not I liked someone enough to consider dating them.  For this I used a Gradient Boosted Classifier from __XG Boost__.  I intentionally overfit the model (500 trees, tree depth of 4, and learning rate of 50%) because I am not interested in building a predictive algorithm using the limited variables I have.  The primary issue in doing so is that some of the variables such as attraction, humor, intellectual connection, chemistry, politics, and second date can only be answered fairly after I've met someone.  Thus this is a _descriptive_ project and not an _inferential_ or _predictive_ one.  

The final rankings of which features are most important in deciding whether or not I "liked" someone are given by the `.feature_importances_` attribute of the fitted model.  These are determined by the amount of [information gained](https://en.wikipedia.org/wiki/Entropy_(information_theory)) if using __entropy__ as the metric to evaluate successful node splits in a tree, or from the magnitude of increase in node purity if using the [Gini coefficient](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#giniimp) instead.


##### Data Prep
There were three stages to data prep in this project.  
1. Dummy the categorical and binary variables.
2. Address the _Age_ over-importance issue (see below).
3. Add _Chemistry_ feature.


### The First Time
Fo the first run of the model I had not yet created the _Chemistry_ variable, or as they say in Russia, there was no "_chim-ih-stree_."  Initially I did not have the _Chemistry_ variable included, and it turns out this mattered.  We'll come back to this in our second run below.  Here are the top 10 features for the (overfit) GBC model:

###### First Run Feature Importance
Rank | Feature                 | Importance
-----|-------------------------|-----------
1    | Age                     | 54.9%
2    | Humor                   | 13.6%
3    | Attraction              | 13.6%
4    | Height (in.)            | 8.1%
5    | Intellectual Connection | 3.2%
6    | Hair Brunette           | 1.9%
7    | Second Date (No)        | 1.7%
8    | Attitude Positive       | 1.3%
9    | Politics Left           | 0.8%
10   | Income Low              | 0.8%

<sub> __Table 2:__ The top 10 features from my initial run, without _Chemistry_, and without any data prep apart from dummied variables.</sub>

<BR>

My initial reaction to this result was one of surprise.  Age?  Really?  That's the overwhelmingly most predictive feature for whether or not I liked someone?  That sure didn't _feel_ right, so I did a quick visual inspection to see if this made sense by looking at the basic Pearson R<sup>2</sup> correlation between _Age_ and a binary version of the _Like_This_Person?_ target, as well as the same for _Attraction_.

###### Relative Correlations for Age vs Attraction
<img src="images/age_attraction_like_pair_plot.png" alt="Comparison of Age and Attraction correlation to 'Liking' someone.">

<sub> __Figure 1:__ Comparison of _Age_ and _Attraction_ correlation to 'Liking' someone.</sub>

<BR>

Visually the result doesn't support the feature importance results above.  The slope for _Age_ is much flatter than that for _Attraction_.  This is not a definitive result (for one, because we shouldn't use a linear regression in this situation); it's just a basic litmus test to see if things are in the right ballpark.  This plot suggests that we are not.  

##### Logit Coefficients
Curious, I wanted to find the basic correlation between _Age_, _Attraction_, and _Like_This_Person?_ to further support that the initial feature importances seemed askew.  Since our dependent variable of _Like_This_Person?_ is binary, a standard OLS linear regression is [not advised](http://thestatsgeek.com/2015/01/17/why-shouldnt-i-use-linear-regression-if-my-outcome-is-binary/).  So, using __Sci-Kit Learn__'s `LogisticRegression` module (`C=1.0, penalty='l1'`), the following coefficients were calculated for the two variables in question.

Variable   | Logit Coefficient
-----------|:----------------:|
Age        | -.011
Attraction |  .353

<sub> __Table 3:__ Logistic Regression coefficients for Age and Attraction, showing a greatly diminished importance for Age.</sub>

<BR>

This isn't meant to be a comprehensive logistic regression report using an optimally-tuned Logit model, but rather a guideline as to the relative importance of each feature.  So both graphically (__Figure 1__) and mathematically we are seeing that _Age_ does not appear to be particularly important.  But our (more advanced) gradient boosted model said that _Age_ was not just important, but the most important feature by a mile.  What gives?

#### Age Is Just A Number
Well, this is where understanding what's going on "under the hood" for the models we are using shows its importance.  When using a decision tree-based model, of which gradient boosting is an ensemble version, there is a specific susceptibility for inflated feature importance reporting if a feature can take on a large range of unique values relative to other features in the dataset. <sup>[__1__](#fn1)</sup>  As the decision tree makes a split, many of the features in this dataset are binary (dummied) and provide little difference for the model to measure.  

Looking back at [Table 1](#the-database-breakdown), we see the range of values that _Age_ can take is double any other numeric feature!  The model is getting a unique value on every split of _Age_ at around twice the rate it is for the other numeric columns and an order of magnitude more than the binary ones.  Thus, it seems safe to say that the reported importance of _Age_ is a relic of the database's construction and is not indicative of its true value in determining if I will "like" someone.

Clearly we need to fix this error.  A first thought might be to simply dummy the _Age_ variable as we did with the categorical ones at the beginning of the project.  However, this would result in two problems.  First, this would create N = `df['Age'].unique().shape[0]` = 20 dummy columns.  Second, we risk completely losing the essence of the _Age_ variable.  With 20 columns each only representing a single age, it is incredibly unlikely that any single age will register as "important."  This is due to there being relatively few entries of any given age, so the each split on a given column of age is unlikely to include a "yes" for that age -- there just aren't going to be a majority of people in this database that share the exact same age.  

Further, what if the model said "Age 31" is the most important age?  What would that really even mean?  Are we saying that I am only going to "like" someone if they're 31?  If I met that same person a week before or after they were 31 years old, would I not like them equally as much?  It appears that a basic dummying of the _Age_ column is not going to resolve our issue.  Fortunately, there is an easy and interpretable solution.

##### Binning
Instead we will __bin__ our _Age_ data.  As anyone who's made a basic histogram in MS Excel knows, you sort data into groups based on different cutoffs.  For our _Age_ data, we'll bin into quintiles (fifths).  This way we can have a group structure of "very low, low, average, high, very high" for the ages in the database.  There is one catch, though, and that is the _count_ of people in each group.  Are we going to make the group cutoffs based on percentile ranking or percentile membership?  

For a quintile binning, the ranking approach simply takes all the data points in the bottom 20th-percentile (the first quintile) and puts them in the "very low" bin, data points in the 20th- to 40th-percentile (the second quintile) and puts them in the "low" bin, all the way up to data points above 80th-percentile (the fifth quintile) being shoved into the "very high" bin.  If your data is distributed roughly evenly, this is fine, else it is vulnerable to outliers causing greatly imbalanced bins.  

Let's pretend there was a person of age 55 in this database.  Using a percentile ranking, this person would rank in the 100th-percentile and the nearest person below them would be in, oh, the ~70th-percentile.  As a result, the top bin of "very high" would have _only one person_.  What statistical conclusions can be drawn from a lone data point?  (Hint: none).  In Pandas, this method is `pd.cut()`.

The other option is to bin the data based on _count_.  We simply count the number of data points and put the bottom scoring fifth into the "very low" bucket, regardless of if they're all actually below the 20th-percentile in their value.  We repeat this for each bracket, up to taking the highest fifth of the dataset and putting it into the "very high" bucket.  This approach ensures (as close to possible) balanced bins and is impervious to extreme outliers.  This is the route we will follow in this model.  In Pandas, this method is `pd.qcut()` (which stands for "quantile cut" and can divide your data into any N number of divisions).


### Second Run
With our binned _Age_ variable, the feature importance results are notably different and, anecdotally speaking as the person who has met the people in this database, much more accurate.

###### Second Run Feature Importance

Rank | Feature                 | Importance
-----|-------------------------|-----------
1    | Height (in)             | 19.3%
2    | Attraction              | 15.1%
3    | Hair Brunette           | 14.9%
4    | Hair Blonde             | 12.7%
5    | Second Date (No)        | 8.5%
6    | Politics Left           | 8.1%
7    | Intellectual Connection | 7.8%
8    | Humor                   | 6.8%
9    | Attitude Positive       | 4.0%
10   | Income Low              | 2.8%










### What Factors Best Predict Whether I Will Want to Continue Dating Someone?










### References
<a name="fn1">1.</a> https://en.wikipedia.org/wiki/Information_gain_in_decision_trees#Drawbacks
