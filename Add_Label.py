#In Original dataset, there is no label for comments. In this file, 5 categories will be added to facilitate ml process.
import pandas as pd

data = pd.read_csv("Comments_0.csv", index_col=False)
#Speed value for restaurants which have not own curriers was 0; now it will be 10.
data['speed'] = data['speed'].replace(0, 10)



#Defining label category by getting means of points.
data['means'] = (data.speed + data.service + data.flavour) / 3

#Creating labels by using pandas.cut function.
labels=[1, 2, 3, 4, 5]
bins=[0, 3, 5, 7, 9, 10]
data['means'] = pd.cut(data['means'], bins=bins, labels=labels)
print(data.head())

#Saving new dataset to csv file.
data.to_csv("comments.csv", index=False)