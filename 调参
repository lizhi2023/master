确定树的深度，选取合适n_estimators

max_depths = np.linspace(2, 15, 14, endpoint=True)
train_results = []
test_results = []

for max_depth in max_depths:
    reg=XGBC(max_depth=int(max_depth))
    reg.fit(X_resampled, y_resampled)
    train_pred = reg.predict(X_resampled)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_resampled, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = reg.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
dis=np.array(train_results)-np.array(test_results)

plt.style.use('seaborn')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


axisx = range(100,1020,100)
rs = []
for i in axisx:
    reg = XGBC(n_estimators=i,max_depth=5)
    rs.append(CVS(reg,X_resampled, y_resampled,cv=3).mean())#X_train,Y_train
print(axisx[rs.index(max(rs))],max(rs))
plt.figure(figsize=(20,8))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()

axisx = range(700,1010,50)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBC(n_estimators=i,max_depth=5)
    cvresult = CVS(reg,X_resampled, y_resampled)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1-cvresult.mean())**2+cvresult.var())

print(axisx[rs.index(max(rs))],var[rs.index(max(rs))],max(rs))
print(axisx[var.index(min(var))],rs[var.index(min(var))],min(var))
print(axisx[ge.index(min(ge))],rs[ge.index(max(ge))],var[ge.index(min(ge))],min(ge))
plt.figure(figsize=(20,8))
plt.plot(axisx,rs,c="red",label="XGB")
plt.legend()
plt.show()
