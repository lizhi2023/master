dfull = xgb.DMatrix(X_resampled, y_resampled)

param1 = {'silent':True
          ,"max_depth":5
          ,"eta":0.25
          ,"subsample": 0.7
          ,"gamma":0.35
          ,"lambda":0.2
          ,"alpha":0
          ,"colsample_bytree":1
          ,"colsample_bylevel":1
          ,"colsample_bynode":1
          ,"nfold":5}
num_round = 900

time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

fig,ax = plt.subplots(1,figsize=(16,10))
ax.set_ylim(top=0.6)
ax.grid()
ax.plot(range(1,num_round+1),cvresult1.iloc[:,0],c="red",label="train,original")
ax.plot(range(1,num_round+1),cvresult1.iloc[:,2],c="gray",label="test,original")

param2 = {'silent':True
          ,"max_depth":5
          ,"eta":0.25
          ,"subsample": 0.7
          ,"gamma":0.35
          ,"lambda":0.2
          ,"alpha":0
          ,"colsample_bytree":0.8
          ,"colsample_bylevel":0.8
          ,"colsample_bynode":0.9
          ,"nfold":5}

time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_round)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

ax.plot(range(1,num_round+1),cvresult2.iloc[:,0],c="green",label="train,last")
ax.plot(range(1,num_round+1),cvresult2.iloc[:,2],c="blue",label="test,last")

ax.legend(fontsize="xx-large")
plt.show()

param1 = {"colsample_bytree":np.arange(0.7,1.01,0.1)
          ,"colsample_bylevel":np.arange(0.7,1.01,0.1)
          ,"colsample_bynode":np.arange(0.7,1.01,0.1)}

reg = XGBC(n_estimators=900,max_depth=5,subsample=0.7,learning_rate=0.25,gamma=0.35,reg_lambda=0.2)
gscv = GridSearchCV(reg,param_grid=param1,scoring = "neg_mean_squared_error",cv=3)

time0=time()
gscv.fit(X_resampled, y_resampled)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))

print(gscv.best_params_,gscv.best_score_)
