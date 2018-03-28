import numpy as np
import pandas as pd

def PreprocessData(raw_df):
    #Remove the 'name' col
    df = raw_df.drop(['name'], axis=1)
    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)

    df['sex'] = df['sex'].map({'female': 0, 'male': 1}).astype(int)

    x_OneHot_df = pd.get_dummies(data=df, columns = ["embarked"])
    ndarray = x_OneHot_df.values

    label = ndarray[:,0] #answer('survived' col)
    Features = ndarray[:,1:] #input(other cols)

    minmax_scale = preprocessing.MinMaxScaler(feature_range(0, 1))

    scaledFeatures = minmax_scale.fit_transform(Features)

    return scaledFeatures, label

TRAIN_FILE_PATH='./data/training_data(1000).xlsx'
TEST_FILE_PATH='./data/testing_data.xlsx'

train_df = pd.read_excel(TRAIN_FILE_PATH)
test_df = pd.read_excel(TEST_FILE_PATH)


# Headers: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
# Filter out 'Ticket', 'PassengerId' and 'Cabin' columns
# cols = ['Survived','Pclass','Name','Sex','Age','SibSp','Parch','Fare','Embarked']
# train_df = train_df[cols]

# Show top 2 records
print("\t[Info] Show top 2 records:")
print(train_df.as_matrix()[:10])
print("")
