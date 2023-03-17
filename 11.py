
reg_after=XGBC(n_estimators=900,max_depth=5,learning_rate=0.25,subsample=0.7,gamma=0.35,reg_lambda=0.2,colsample_bylevel=0.8)
reg_after.fit(X_resampled, y_resampled)

print("测试集和训练集的分类精度分别为：",reg_after.score(X_test,Y_test),reg_after.score(X_resampled,y_resampled))
print("test_f1:",metrics.f1_score(reg_after.predict(X_test),Y_test))
print("train_f1",metrics.f1_score(reg_after.predict(X_resampled),y_resampled))


#模型精度变化
plt.bar(range(4)
        ,[reg_before.score(X_test,Y_test),reg_after.score(X_test,Y_test),reg_before.score(X_resampled,y_resampled),reg_after.score(X_resampled,y_resampled)]
        ,width=0.4,color=['red','red','gray','gray'])
plt.xticks(range(4),["reg_before_test","reg_after_test","reg_before_train","reg_after_train"])
plt.show()

#模型过拟合程度变化
plt.bar(range(2)
        ,[metrics.f1_score(reg_before.predict(X_resampled),y_resampled)-metrics.f1_score(reg_before.predict(X_test),Y_test)
        ,metrics.f1_score(reg_after.predict(X_resampled),y_resampled)-metrics.f1_score(reg_after.predict(X_test),Y_test)]
        ,width=0.3,color=['red','gray'])
plt.xticks(range(2),["reg_before_overfitting","reg_after_overfitting"])
plt.show()


#保存模型
import joblib
joblib.dump(reg_after,"xgb_model")


#调用模型
model=joblib.load("xgb_model")

from xgboost import plot_tree
fig, ax = plt.subplots(figsize=(50, 48))
plot_tree(model,ax=ax,rankdir='LR')
fig = plt.gcf()
fig.set_size_inches(150, 100)

#plt.show()
fig.savefig('tree.png')


#输出预测集的预测结果
pred_result = model.predict(feature2)
print("具有付费意愿的用户数量为",sum(pred_result))
具有付费意愿的用户数量为 59
#输出预测集的概率
pred_proba = model.predict_proba(feature2)
pred=pd.read_excel(r'./预测名单.xlsx')
pred["是否具有付费意愿"]=pd.DataFrame(pred_result)
pred["付费意愿的概率"]=pd.DataFrame(pred_proba[:,1])
outputpath='./付费意愿预测名单.xlsx'
pred.to_excel(outputpath,index=True,header=True)
