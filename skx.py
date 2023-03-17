axisx = np.arange(0,1,0.05)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBC(n_estimators=900,max_depth=5,subsample=0.7,learning_rate=0.25,gamma=i,random_state=2021)
    cvresult = CVS(reg,X_resampled, y_resampled)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1-cvresult.mean())**2+cvresult.var())

print(axisx[rs.index(max(rs))],var[rs.index(max(rs))],max(rs))
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(max(ge))],var[ge.index(min(ge))],min(ge))

rs=np.array(rs)
var=np.array(var)

plt.figure(figsize=(20,8))
plt.plot(axisx,rs,c="red",label="XGB")
plt.plot(axisx,rs+var,c="blue",linestyle='-.')
plt.plot(axisx,rs-var,c="blue",linestyle='-.')
plt.legend()
plt.show()


param = {"lambda":np.arange(0,1.5,0.2),"alpha":np.arange(0,1.5,0.2)}

reg = XGBC(n_estimators=900,max_depth=5,subsample=0.7,learning_rate=0.25,gamma=0.35)

gscv = GridSearchCV(reg,param_grid=param,scoring = "neg_mean_squared_error",cv=3)

time0=time()
gscv.fit(X_resampled, y_resampled)
print(datetime.datetime.fromtimestamp(time()-time0).strftime("%M:%S:%f"))
print(gscv.best_params_,gscv.best_score_)
preds = gscv.predict(X_test)
print("MSE:",MSE(Y_test,preds))
