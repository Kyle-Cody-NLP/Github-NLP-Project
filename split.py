# Functions in this SPLIT file:
# train_validate_test_split(df, target, seed): versatile splitting function that
# returns a train( 56%), validate (24%) and test data (20%) set
#_____________________________________________________________________________
# Required imports for these files:
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
# filter out warnings
import warnings
warnings.filterwarnings('ignore')
#_____________________________________________________________________________
def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test.
    Test is 20% of the original dataset, validate is .30*.80= 24% of the
    original dataset, and train is .70*.80= 56% of the original dataset.
    The function returns, in this order, train, validate and test dataframes.
    '''
    train_validate, test = train_test_split(df, test_size=0.2,
                                            random_state=seed,
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3,
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

    