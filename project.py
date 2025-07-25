import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
df=pd.read_csv('Bank_Personal_Loan_Modelling (1).csv')
df.head()
personal_loan=df['Personal Loan']
df.drop(['Personal Loan'],axis=1,inplace=True)
df['Personal Loan']=personal_loan
df.head()
#shape of the data
rows_count,columns_count=df.shape
print('Total no.of Rows : ',rows_count)
print('Total no.of Columns : ',columns_count)
df.info()
df.isnull().sum()
df.isnull().values.any()
sns.heatmap(df.isna(),yticklabels=False,cbar=False,cmap='viridis')
df.nunique()
df.describe()
df_transpose=df.describe().T
df_transpose[['min','25%','50%','75%','max']]
sns.pairplot(df.iloc[:,1:])
df[df['Experience']<0]['Experience'].value_counts()
df[df['Experience']<0]['Experience'].count()
quantitiveVar=['Age','Income','CCAvg','Mortgage']
expGrid=sns.PairGrid(df,y_vars='Experience',x_vars=quantitiveVar)
expGrid.map(sns.regplot)
df_Positive_Experience=df[df['Experience']>0]
df_Negative_Experience=df[df['Experience']<0]
df_Negative_Experience_List=df_Negative_Experience['ID'].tolist()
for id in df_Negative_Experience_List:
    age_values=df.loc[np.where(df['ID']==id)]["Age"].tolist()[0]
    education_values=df.loc[np.where(df['ID']==id)]["Education"].tolist()[0]
    positive_Experience_Filtered=df_Positive_Experience[(df_Positive_Experience['Age']==age_values) & (df_Positive_Experience['Education']==education_values)]
    if positive_Experience_Filtered.empty:
        negative_Experience_Filtered=df_Negative_Experience[(df_Negative_Experience['Age'] == age_values) & (df_Negative_Experience['Education']==education_values)]
        exp=round(negative_Experience_Filtered['Experience'].median())
    else:
        exp=round(positive_Experience_Filtered['Experience'].median())
    df.loc[df.loc[np.where(df['ID']==id)].index,'Experience']=abs(exp)
df[df['Experience']<0]['Experience'].count()
df.Experience.describe()
sns.histplot(df['ID'])
sns.histplot(df['Age'])
sns.histplot(df['Experience'])
sns.histplot(df['Income'])
sns.histplot(df['ZIP Code'])
sns.histplot(df['CCAvg'])
sns.histplot(df['Education'])
sns.histplot(df['Mortgage'])
sns.histplot(df['Online'])
sns.histplot(df['CreditCard'])
loan_counts=pd.DataFrame(df["Personal Loan"].value_counts()).reset_index()
loan_counts.columns=["Labels","Personal Loan"]
loan_counts
fig1,ax1=plt.subplots()
explode=(0,0.15)
ax1.pie(loan_counts["Personal Loan"],explode=explode,labels=loan_counts["Labels"],autopct='%1.1f%%',shadow=True,startangle=70)
ax1.axis('equal')
plt.title("Personal Loan Percentage")
plt.show()
sns.boxplot(x='Family',y='Income',hue='Personal Loan',data=df)
sns.boxplot(x='Education',y='Income',hue='Personal Loan',data=df)
sns.boxplot(x='Education',y='Mortgage',hue='Personal Loan',data=df)
sns.countplot(x="Securities Account",data=df,hue="Personal Loan")
sns.countplot(x='Family',data=df,hue='Personal Loan')
sns.countplot(x='CD Account',data=df,hue='Personal Loan')
sns.boxplot(x='CreditCard',y='CCAvg',hue='Personal Loan',data=df)
sns.catplot(x='Age',y='Experience',hue='Personal Loan',data=df,height=8.27,aspect=11/5)
plt.figure(figsize=(10,4))
sns.histplot(df[df["Personal Loan"]==0]['CCAvg'],color='r',label='Personal Loan=0')
sns.histplot(df[df["Personal Loan"]==1]['CCAvg'],color='b',label='Personal Loan=1')
plt.legend()
plt.title('CCAvg Distribution')
print('Credit card spending of Non-Loan customers :',df[df['Personal Loan']==0]['CCAvg'].median()*1000)
print('Credit card spending of Loan customers :',df[df['Personal Loan']==1]['CCAvg'].median()*1000)
plt.figure(figsize=(10,4))
sns.histplot(df[df['Personal Loan']==0]['Income'],color='r',label='Personal Loan=0')
sns.histplot(df[df['Personal Loan']==1]['Income'],color='b',label='Personal Loan=1')
plt.legend()
plt.title("Income Distribution")
df.boxplot(return_type='axes',figsize=(20,5))
df.head(1)
df=df.drop(['ID','ZIP Code'],axis=1)
df.head(1)
loan_with_experience=df
loan_without_experience=df.drop(['Experience'],axis=1)
print('Columns With Experience :',loan_with_experience.columns)
print('Columns Without Experience :',loan_without_experience.columns)
X_Expr=loan_with_experience.drop('Personal Loan',axis=1)
Y_Expr=loan_with_experience[['Personal Loan']]
X_Without_Expr=loan_without_experience.drop('Personal Loan',axis=1)
Y_Without_Expr=loan_without_experience[['Personal Loan']]
X_Expr_train,X_Expr_test,y_Expr_train,y_Expr_test=train_test_split(X_Expr,Y_Expr,test_size=0.30,random_state=1)
print('x train data {}'.format(X_Expr_train.shape))
print('y train data {}'.format(y_Expr_train.shape))
print('x test data {}'.format(X_Expr_test.shape))
print('y test data {}'.format(y_Expr_test.shape))
X_train,X_test,y_train,y_test=train_test_split(X_Without_Expr,Y_Without_Expr,test_size=0.30,random_state=1)
print('x train data {}'.format(X_train.shape))
print('y train data {}'.format(y_train.shape))
print('x test data {}'.format(X_test.shape))
print('y test data {}'.format(y_test.shape))
print('----------------LOGISTIC REGRESSION---------\n')
### With Experience Column
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix
scaler=StandardScaler()
X_Expr_train_scaled=scaler.fit_transform(X_Expr_train)
X_Expr_test_scaled=scaler.transform(X_Expr_test)
y_Expr_train_flat=y_Expr_train.values.ravel()
y_Expr_test_flat=y_Expr_test.values.ravel()
logreg_expr_model=LogisticRegression(max_iter=1000)
logreg_expr_model.fit(X_Expr_train_scaled,y_Expr_train_flat)
print(logreg_expr_model,'\n')
logreg_expr_y_predicted=logreg_expr_model.predict(X_Expr_test_scaled)
logreg_expr_score=logreg_expr_model.score(X_Expr_test_scaled,y_Expr_test_flat)
logreg_expr_accuracy=accuracy_score(y_Expr_test_flat,logreg_expr_y_predicted)
logestic_confusion_matrix_expr=confusion_matrix(y_Expr_test_flat,logreg_expr_y_predicted)
print('Accuracy Score : ',logreg_expr_accuracy)
print('Confusion Matrix : \n',logestic_confusion_matrix_expr)
### Without Experience Column
#X_train,X_test,y_train,y_test
logreg_model=LogisticRegression()
logreg_model.fit(X_train,y_train)
logreg_y_predicted=logreg_model.predict(X_test)
logreg_score=logreg_model.score(X_test,y_test)
logreg_accuracy=accuracy_score(y_test,logreg_y_predicted)
logestic_confusion_matrix=metrics.confusion_matrix(y_test,logreg_y_predicted)
#Accuracy
print('Logistic Regression Model Accurcay score W/O Experience :%f' % logreg_accuracy)
print('Logistic regression Model Accuracy Score With Experinece :%f' % logreg_expr_accuracy)
#Confusion Matrix
print('\n Logistic Regression Confusion Matrix W/O Experience : \n',logestic_confusion_matrix)
print('\n True Positive =',logestic_confusion_matrix[1][1])
print('True Negative =',logestic_confusion_matrix[0][0])
print('False Positive =',logestic_confusion_matrix[0][1])
print('False Negative =',logestic_confusion_matrix[1][0])
print('\n Logistic Regression Confusion Matrix With Experience: \n',logestic_confusion_matrix_expr)
print('\n True Positive =',logestic_confusion_matrix_expr[1][1])
print('True Negative =',logestic_confusion_matrix_expr[0][0])
print('False Positive =',logestic_confusion_matrix_expr[0][1])
print('False Negative =',logestic_confusion_matrix_expr[1][0])
### Improvement of the model Iteration 2 For Logistic Regression With Experience
#X_Expr_train,X_Expr_test,y_Expr_train,y_Expr_test
X_train_scaled=preprocessing.scale(X_Expr_train)
X_test_scaled=preprocessing.scale(X_Expr_test)

scaled_logreg_model=LogisticRegression()
scaled_logreg_model.fit(X_train_scaled,y_Expr_train)
#Predicting for test Set
scaled_logreg_y_predicted=scaled_logreg_model.predict(X_test_scaled)
scaled_logreg_model_score=scaled_logreg_model.score(X_test_scaled,y_Expr_test)
scaled_logreg_accuracy=accuracy_score(y_Expr_test,scaled_logreg_y_predicted)

scaled_logreg_confusion_matrix=metrics.confusion_matrix(y_Expr_test,scaled_logreg_y_predicted)
print('----------------------FINAL ANALYSIS OF LOGISTIC REGRESSION----------\n')
print('After Scaling logistic Regression Model Accuracy Score With Experience : %f'% scaled_logreg_accuracy)
print('\n After Scaling Logistic Regression Confusion Matrix With Experience : \n',scaled_logreg_confusion_matrix)
print('\nTrue Positive = ',scaled_logreg_confusion_matrix[1][1])
print('True Negative = ',scaled_logreg_confusion_matrix[0][0])
print('False Positive = ',scaled_logreg_confusion_matrix[0][1])
print('False Negative = ',scaled_logreg_confusion_matrix[1][0])
print('\nK-NN classification Report : \n',metrics.classification_report(y_Expr_test,scaled_logreg_y_predicted))
conf_table=scaled_logreg_confusion_matrix
a=(conf_table[0,0]+conf_table[1,1])/(conf_table[0,0]+conf_table[0,1]+conf_table[1,0]+conf_table[1,1])
p=conf_table[1,1]/(conf_table[1,1]+conf_table[0,1])
r=conf_table[1,1]/(conf_table[1,1]+conf_table[1,0])
f=(2*p*r)/(p+r)
print('Accuracy of accepting Loan : ',round(a,2))
print('precision of accepting Loan :',round(p,2))
print('recall of accepting Loan :',round(r,2))
print('F1 score of accepting Loan :',round(f,2))
## ::-----------------K-NN-----------::
numberList=list(range(1,20))
neighbors=list(filter(lambda x:x%2!=0,numberList))
ac_scores=[]
for k in neighbors:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train.values.ravel())
    y_pred=knn.predict(X_test)
    scores=accuracy_score(y_test,y_pred)
    ac_scores.append(scores)
MSE=[1 -x for x in ac_scores]
# Determining best k
optimal_k=neighbors[MSE.index(min(MSE))]
print('Odd Neighbors : \n',neighbors)
print('\nAccuracy Score : \n',ac_scores)
print('\nMisclassification error : \n',MSE)
print('\nThe optimal number of neighbor is k=',optimal_k)
#plot misclassification error VS k
plt.plot(neighbors,MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
#Instantiating Learning model W/O Experience
knn_model=KNeighborsClassifier(n_neighbors=optimal_k,weights='uniform',metric='euclidean')
knn_model.fit(X_train,y_train)
knn_y_predicted=knn_model.predict(X_test)
knn_score=knn_model.score(X_test,y_test)
knn_accuracy=accuracy_score(y_test,knn_y_predicted)
knn_confusion_matrix=metrics.confusion_matrix(y_test,knn_y_predicted)
#Instantiating Learning model With Experinece
knn_model_expr=KNeighborsClassifier(n_neighbors=optimal_k,weights='uniform',metric='euclidean')
knn_model_expr.fit(X_Expr_train,y_Expr_train)
knn_expr_y_predicted=knn_model_expr.predict(X_Expr_test)
knn_expr_score=knn_model_expr.score(X_Expr_test,y_Expr_test)
knn_expr_accuracy=accuracy_score(y_Expr_test,knn_expr_y_predicted)
knn_confusion_matrix_expr=metrics.confusion_matrix(y_Expr_test,knn_expr_y_predicted)
#comparison
print('K-NN Model Accuracy Score W/O Experinece :%f'% knn_accuracy)
print('K-NN Model Accuracy Score With Experience :%f'% knn_expr_accuracy)
#confusion Matrix
print('\nK-NN Confuson Matrix W/O Experience :\n',knn_confusion_matrix)
print('\nTrue Positive =',knn_confusion_matrix[1][1])
print('True Negative =',knn_confusion_matrix[0][0])
print('False Positive =',knn_confusion_matrix[0][1])
print('False Negative =',knn_confusion_matrix[1][0])
print('\nK-NN Confuson Matrix With Experience :\n',knn_confusion_matrix_expr)
print('\nTrue Positive =',knn_confusion_matrix_expr[1][1])
print('True Negative =',knn_confusion_matrix_expr[0][0])
print('False Positive =',knn_confusion_matrix_expr[0][1])
print('False Negative =',knn_confusion_matrix_expr[1][0])
### Improvement of the model ---- Iteration 2 For K-NN Without Experience dataset----
X_train_scaled=preprocessing.scale(X_train)
X_test_scaled=preprocessing.scale(X_test)

scaled_knn_model=KNeighborsClassifier(n_neighbors=optimal_k,weights='uniform',metric='euclidean')
scaled_knn_model.fit(X_train_scaled,y_train)
scaled_knn_y_predict=scaled_knn_model.predict(X_test_scaled)
scaled_knn_score=scaled_knn_model.score(X_test_scaled,y_test)
scaled_knn_accuracy=accuracy_score(y_test,scaled_knn_y_predict)
scaled_knn_confusion_matrix=metrics.confusion_matrix(y_test,scaled_knn_y_predict)
print('--------------FINAL ANALYSIS OF K-NN--------------\n')
print('After Scaling K-NN Model Accuracy Score Without Experience :%f'%scaled_knn_accuracy)
print('\nAfter Scaling K-NN Confusion Matrix Without Experience :\n',scaled_knn_confusion_matrix)
print('\nTrue Positive =',scaled_knn_confusion_matrix[1][1])
print('True Negative =',scaled_knn_confusion_matrix[0][0])
print('False Positive =',scaled_knn_confusion_matrix[0][1])
print('False Negative =',scaled_knn_confusion_matrix[1][0])
print('\nK-NN Classification Report :\n',metrics.classification_report(y_test,scaled_knn_y_predict))
knn_conf_table=scaled_knn_confusion_matrix
a=(knn_conf_table[0,0]+knn_conf_table[1,1])/(knn_conf_table[0,0]+knn_conf_table[0,1]+knn_conf_table[1,0]+knn_conf_table[1,1])
p=knn_conf_table[1,1]/(knn_conf_table[1,1]+knn_conf_table[0,1])
r=knn_conf_table[1,1]/(knn_conf_table[1,1]+knn_conf_table[1,0])
f=(2*p*r)/(p+r)
print('\nAccuracy of accepting Loan :',round(a,2))
print('Precision of accepting Loan :',round(p,2))
print('recall of accepting Loan :',round(r,2))
print('F1 score of accepting Loan :',round(f,2))
##-------------------NAIVE BAYES---------------
#Model Building using W/O Experience
gnb_model=GaussianNB()
gnb_model.fit(X_train,y_train)
gnb_y_predicted=gnb_model.predict(X_test)
gnb_score=gnb_model.score(X_test,y_test)
gnb_accuracy=accuracy_score(y_test,gnb_y_predicted)
gnb_confusion_matrix=metrics.confusion_matrix(y_test,gnb_y_predicted)
#Model Building Using With Experience
gnb_expr_model=GaussianNB()
gnb_expr_model.fit(X_Expr_train,y_Expr_train)
gnb_expr_y_predicted=gnb_expr_model.predict(X_Expr_test)
gnb_expr_score=gnb_expr_model.score(X_Expr_test,y_Expr_test)
gnb_expr_accuracy=accuracy_score(y_Expr_test,gnb_expr_y_predicted)
gnb_expr_confusion_matrix=metrics.confusion_matrix(y_Expr_test,gnb_expr_y_predicted)
#Comparision
print('Naive Bayes Model Accuracy Score W/O Experience :%f'%gnb_accuracy)
print('Naive Bayes Model Acuuracy Score With Experience :%f'%gnb_expr_accuracy)
#Confusion Matrix
print('\nNaive Bayes Confusion Matrix W/O Experience :\n',gnb_confusion_matrix)
print('\nTrue Positive =',gnb_confusion_matrix[1][1])
print('True Negative =',gnb_confusion_matrix[0][0])
print('False Positive =',gnb_confusion_matrix[0][1])
print('False Negative =',gnb_confusion_matrix[1][0])
print('\nNaive Bayes Confusion Matrix With Experience :\n',gnb_expr_confusion_matrix)
print('\nTrue Positive =',gnb_expr_confusion_matrix[1][1])
print('True Negative =',gnb_expr_confusion_matrix[0][0])
print('False Positive =',gnb_expr_confusion_matrix[0][1])
print('False Negative =',gnb_expr_confusion_matrix[1][0])
###Improvement of the model --------Iteration 2 for Naive Bayes W/O Experience -------
scaled_gnb_model=GaussianNB()
scaled_gnb_model.fit(X_train_scaled,y_train)
scaled_gnb_y_predict=scaled_gnb_model.predict(X_test_scaled)
scaled_gnb_score=scaled_gnb_model.score(X_test_scaled,y_test)
scaled_gnb_accuracy=accuracy_score(y_test,scaled_gnb_y_predict)
scaled_gnb_confusion_matrix=metrics.confusion_matrix(y_test,scaled_gnb_y_predict)
print('-------------FIANL ANALYSIS OF NAIVE BAYES---------------\n')
print('After Scaling Naive Bayes Model Accuracy Score :%f'%scaled_gnb_accuracy)
print('After Scaling Naive Bayes Confusion Matrix :\n',scaled_gnb_confusion_matrix)
print('\nTrue Positive =',scaled_gnb_confusion_matrix[1][1])
print('True Negative =',scaled_gnb_confusion_matrix[0][0])
print('False Positive =',scaled_gnb_confusion_matrix[0][1])
print('False Negative =',scaled_gnb_confusion_matrix[1][0])
print('\nGaussian Naive Bayes Classification Report :\n',metrics.classification_report(y_test,gnb_y_predicted))
gnb_conf_table=scaled_gnb_confusion_matrix
a=(gnb_conf_table[0,0]+gnb_conf_table[1,1])/(gnb_conf_table[0,0]+gnb_conf_table[0,1]+gnb_conf_table[1,0]+knn_conf_table[1,1])
p=gnb_conf_table[1,1]/(gnb_conf_table[1,1]+gnb_conf_table[0,1])
r=gnb_conf_table[1,1]/(gnb_conf_table[1,1]+gnb_conf_table[1,0])
f=(2*p*r)/(p+r)
print('\nAccuracy of accepting Loan :',round(a,2))
print('Precision of accepting Loan :',round(p,2))
print('recall of accepting Loan :',round(r,2))
print('F1 Score of accepting Loan :',round(f,2))
##=============COMPARISON OF ABOVE THREE MODELS==============
print('Overall Model Accuracy After Scaling :\n')
print('Logistic Regression : {0:.0f}%'.format(scaled_logreg_accuracy*100))
print('K-Nearest Neighbors : {0:.0f}%'.format(scaled_knn_accuracy*100))
print('Naive Bayes : {0:.0f}%'.format(scaled_gnb_accuracy*100))

print('\nOverall Model Confusion Matrix After Scaling:\n')
print('\nLogistic Regression : \n',scaled_logreg_confusion_matrix)
print('\n True Positive =',scaled_logreg_confusion_matrix[1][1])
print(' True Negative =',scaled_logreg_confusion_matrix[0][0])
print(' False Positive =',scaled_logreg_confusion_matrix[0][1])
print(' False Negative =',scaled_logreg_confusion_matrix[1][0])

print('\nK-Nearest Neighbors : \n',scaled_knn_confusion_matrix)
print('\n True Positive =',scaled_knn_confusion_matrix[1][1])
print(' True Negative =',scaled_knn_confusion_matrix[0][0])
print(' False Positive =',scaled_knn_confusion_matrix[0][1])
print(' False Negative =',scaled_knn_confusion_matrix[1][0])

print('\nNaive Bayes :\n',scaled_gnb_confusion_matrix)
print('\nTrue Positive =',scaled_gnb_confusion_matrix[1][1])
print(' True Negative =',scaled_gnb_confusion_matrix[0][0])
print(' False Positive =',scaled_gnb_confusion_matrix[0][1])
print(' False Negative =',scaled_gnb_confusion_matrix[1][0])
print('\n\nReceiver Operating Characteristic(ROC) curve to evaluate the classifier output quality.If area of curve is closer to 1 which means better the model and if area of curve is closer to 0 which means poor the model.')
knn_fpr,knn_tpr,knn_threshold=metrics.roc_curve(y_test,scaled_knn_y_predict)
knn_roc_auc=metrics.roc_auc_score(y_test,scaled_knn_y_predict)
fig1_graph=plt.figure(figsize=(15,4))
fig1_graph.add_subplot(1,3,1)
plt.plot(knn_fpr,knn_tpr,label='KNN Model (area=%0.2f)'% knn_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(ROC)')
plt.legend(loc='lower right')
logistic_fpr,logistic_tpr,logistic_threshold=metrics.roc_curve(y_Expr_test,scaled_logreg_y_predicted)
logistic_roc_auc=metrics.roc_auc_score(y_Expr_test,scaled_logreg_y_predicted)
fig1_graph.add_subplot(1,3,2)
plt.plot(logistic_fpr,logistic_tpr,label='Logistic Model (area=%0.2f)'%logistic_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(ROC)')
plt.legend(loc='lower right')
nb_fpr,nb_tpr,nb_threshold=metrics.roc_curve(y_test,scaled_gnb_y_predict)
nb_roc_auc=metrics.roc_auc_score(y_test,scaled_gnb_y_predict)
fig1_graph.add_subplot(1,3,3)
plt.plot(nb_fpr,nb_tpr,label='Naive-Bayes Model (area=%0.2f)'% nb_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic(ROC)')
plt.legend(loc='lower right')
plt.show()