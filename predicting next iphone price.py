#import required packages
import pandas 
import numpy

#further dependencies
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#reading dataset using pandas from the csv
dataset=pandas.read_csv("/home/raghav/Documents/Hands on AI workshop/iphone_pro_max_prices_up_to_16.csv")

# Training
x=dataset[["Model"]] #feauture
y=dataset[["Launch Price (INR)"]] #target

#using train test split to split the data into training and testing sets [lets use 90 percent for training]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=42)
model=LinearRegression() #using linear regression model to exptrapolate to future prices

#training the model 
model.fit(x_train,y_train)

#predictions
prediction_input=numpy.array([[18]]) #predicting iphone 18 price
prediction_input1=numpy.array([[19]]) #predicting iphone 19 price
prediction_input2=numpy.array([[20]]) #predicting iphone 20 price

predicted_iphone_price=model.predict(prediction_input)
predicted_iphone_price1=model.predict(prediction_input1)
predicted_iphone_price2=model.predict(prediction_input2)

print(f"Predicted iPhone 18 Price: {predicted_iphone_price}")
print(f"Predicted iPhone 19 Price: {predicted_iphone_price1}")
print(f"Predicted iPhone 20 Price: {predicted_iphone_price2}")
