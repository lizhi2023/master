
def regassess(reg,X,Y,cv,scoring=["r2"],show=True):
    score=[]
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i],CVS(reg,X,Y,cv=cv,scoring=scoring[i]).mean()))
        score.append(CVS(reg,X,Y,cv=cv,scoring=scoring[i]).mean())
    return score
axisx=np.arange(0.05,0.31,0.05)
rs=[]
te=[]
for i in axisx:
    reg=XGBC(n_estimators=900,max_depth=5,learning_rate=i)
    score=regassess(reg,X_resampled, y_resampled,5,scoring=["r2","neg_mean_squared_error"],show=False)
    test=reg.fit(X_resampled, y_resampled).score(X_test,Y_test)
    rs.append(score[0])
    te.append(test)
    print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,5))
plt.plot(axisx,te,c="gray",label="XGB")
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()

axisx = np.arange(0.1,1,0.1)
rs = []
for i in axisx:
    reg = XGBC(n_estimators=900,max_depth=5,subsample=i,learning_rate=0.25,random_state=201)
    rs.append(CVS(reg,X_resampled, y_resampled,cv=5).mean())
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,8))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
