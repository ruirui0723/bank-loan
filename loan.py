#去除缺失值达一半以上及无关字段
import pandas as pd
loans_2007=pd.read_csv("E:\\pythonproject\\贷款案例\\LoanStats3a.csv",skiprows=1,engine="python")
half_count = len(loans_2007) / 2
loans_2007 = loans_2007.dropna(thresh=half_count, axis=1)
loans_2007 = loans_2007.drop(['desc', 'url'],axis=1)
loans_2007.drop_duplicates()
loans_2007 = loans_2007.drop(["id", "member_id", "funded_amnt", "funded_amnt_inv", "grade", "sub_grade", "emp_title", "issue_d"], axis=1)
loans_2007 = loans_2007.drop(["zip_code", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv", "total_rec_prncp"], axis=1)
loans_2007 = loans_2007.drop(["total_rec_int", "total_rec_late_fee", "recoveries", "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt"], axis=1)
#print (loans_2007.iloc[0])
#print (loans_2007.shape[1])
loans_2007.to_csv('loans_2007_1.csv', index=False)

#目标字段提取
loans_2007 = pd.read_csv("loans_2007_1.csv")
print(loans_2007['loan_status'].value_counts())
loans_2007 = loans_2007[(loans_2007['loan_status'] == "Fully Paid") | (loans_2007['loan_status'] == "Charged Off")]
status_replace = {
    "loan_status" : {
        "Fully Paid": 1,
        "Charged Off": 0,
    }
}
loans_2007=loans_2007.replace(status_replace)

#删除只有唯一属性值的字段
original_columns=loans_2007.columns
drop_columns=[]
for col in original_columns:
    col_series = loans_2007[col].dropna().unique()
    if len(col_series) == 1:
        drop_columns.append(col)
loans_2007=loans_2007.drop(drop_columns,axis=1)
print(drop_columns)
print(loans_2007.shape)

#缺失值处理
null_counts=loans_2007.isnull().sum()
print(null_counts)
loans=loans_2007.drop("pub_rec_bankruptcies", axis=1)
loans=loans.dropna(axis=0)

#各字段值属性转化以及离散化处理
print(loans.dtypes.value_counts())
object_columns=loans.select_dtypes(include=['object'])
print(object_columns.iloc[0])
cols = ['home_ownership', 'verification_status', 'emp_length', 'term', 'addr_state','purpose','title']
for c in cols:
    print(loans[c].value_counts())
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    }
}
loans = loans.drop(["last_credit_pull_d", "earliest_cr_line", "addr_state","title","pymnt_plan"], axis=1)
loans["int_rate"] = loans["int_rate"].str.rstrip("%").astype("float")
loans["revol_util"] = loans["revol_util"].str.rstrip("%").astype("float")
loans=loans.replace(mapping_dict)
cat_columns=["home_ownership", "verification_status", "purpose", "term"]
dummy=pd.get_dummies(loans[cat_columns])
loans=pd.concat([loans,dummy],axis=1)
loans=loans.drop(cat_columns,axis=1)
loans.to_csv('cleaned_loans2007.csv',index=False)
loans = pd.read_csv("cleaned_loans2007.csv")
#print(loans.info())

#建立罗吉斯回归模型
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score,KFold
lr = LogisticRegression(class_weight="balanced")
cols = loans.columns
train_cols = cols.drop("loan_status")
features = loans[train_cols]
target = loans["loan_status"]
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(lr, features, target, cv=kf)
predictions = pd.Series(predictions)
# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])
# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])
# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])
# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])
# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))
print(tpr)
#0.6591452789509437
print(fpr)
#0.3832239758822486
P1=predictions[:20]
print(P1)

#建立随机森林模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,KFold
rf = RandomForestClassifier(n_estimators=10,class_weight="balanced", random_state=1)
kf = KFold(features.shape[0], random_state=1)
predictions = cross_val_predict(rf, features, target, cv=kf)
predictions = pd.Series(predictions)
# False positives.
fp_filter = (predictions == 1) & (loans["loan_status"] == 0)
fp = len(predictions[fp_filter])
# True positives.
tp_filter = (predictions == 1) & (loans["loan_status"] == 1)
tp = len(predictions[tp_filter])
# False negatives.
fn_filter = (predictions == 0) & (loans["loan_status"] == 1)
fn = len(predictions[fn_filter])
# True negatives
tn_filter = (predictions == 0) & (loans["loan_status"] == 0)
tn = len(predictions[tn_filter])
# Rates
tpr = tp / float((tp + fn))
fpr = fp / float((fp + tn))
print(tpr)
#0.9745414808470422
print(fpr)
#0.941478985635751
P2=predictions[:20]
print(P2)








                           
