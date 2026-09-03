import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df=pd.read_csv('week4_logistics_dataset.csv')
X=df.drop(columns=['Shipment_ID','Delivery_Time_days']); y=df['Delivery_Time_days']
cat=['Origin','Destination','Transport_Mode','Weather','Priority']
num=['Distance_km','Shipment_Volume_kg','Traffic_Level','Warehouse_Delay_hours','Transportation_Cost_INR']
pre=ColumnTransformer([('num',Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]),num),('cat',Pipeline([('imp',SimpleImputer(strategy='most_frequent')),('oh',OneHotEncoder(handle_unknown='ignore'))]),cat)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
model=Pipeline([('pre',pre),('m',RandomForestRegressor(n_estimators=120,max_depth=12,min_samples_leaf=2,random_state=42))])
grid=GridSearchCV(model,{'m__max_depth':[8,12],'m__min_samples_leaf':[1,2]},cv=3,scoring='neg_root_mean_squared_error')
grid.fit(X_train,y_train);pred=grid.predict(X_test)
print('MAE:',mean_absolute_error(y_test,pred));print('RMSE:',mean_squared_error(y_test,pred)**0.5);print('R2:',r2_score(y_test,pred));print('Best parameters:',grid.best_params_)
