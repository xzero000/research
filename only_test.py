import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression as lr
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.compose import ColumnTransformer
import time
import csv
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

c_name = np.arange(0,43)
##data = pd.read_csv('KDDTrain+.txt',header = None,names = c_name)
##data_t = pd.read_csv('KDDTest+.txt',header = None, names = c_name)
'--------------- only ''normal'',''attack'' ----------------------'
data = pd.read_csv('NSL-KDD/KDDtrain_normal_attack_4type_shuffle.csv',header = None,names = c_name)
data_t = pd.read_csv('NSL-KDD/KDDtest_4type_attack.csv',header = None,names = c_name)


# data[0] & data[3] swap

'--- if ''KDDtrain_normal_attack_4type_shuffle.csv''------'
'---t_1 = normal dos probe R2L U2R-----'
t_1 = np.array(['normal', 'Dos','Probe',  'R2L', 'U2R'])
d_1 = np.array(['tcp', 'udp', 'icmp'])
d_2 = np.array(['ftp_data', 'other', 'private', 'http', 'remote_job', 'name',
       'netbios_ns', 'eco_i', 'mtp', 'telnet', 'finger', 'domain_u',
       'supdup', 'uucp_path', 'Z39_50', 'smtp', 'csnet_ns', 'uucp',
       'netbios_dgm', 'urp_i', 'auth', 'domain', 'ftp', 'bgp', 'ldap',
       'ecr_i', 'gopher', 'vmnet', 'systat', 'http_443', 'efs', 'whois',
       'imap4', 'iso_tsap', 'echo', 'klogin', 'link', 'sunrpc', 'login',
       'kshell', 'sql_net', 'time', 'hostnames', 'exec', 'ntp_u',
       'discard', 'nntp', 'courier', 'ctf', 'ssh', 'daytime', 'shell',
       'netstat', 'pop_3', 'nnsp', 'IRC', 'pop_2', 'printer', 'tim_i',
       'pm_dump', 'red_i', 'netbios_ssn', 'rje', 'X11', 'urh_i',
       'http_8001', 'aol', 'http_2784', 'tftp_u', 'harvest'])
d_3 = np.array(['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH'])

##X_train_0, X_test_0, y_train, y_test = \
##    train_test_split(data[c_name[0:41]],data[c_name[41]],test_size = 0.2)
##train_test_split(data[column_name[1:10]],data[column_name[10]],test_size=0.25,random_state=33)
## 取前10列为X，第10列为y，并且分割；random_state参数的作用是为了保证每次运行程序时都以同样的方式进行分割

X_train_0 = np.array(data[c_name[0:41]])
X_test_0 = np.array(data_t[c_name[0:41]])
y_train = np.array(data[c_name[41]])
y_test = np.array(data_t[c_name[41]])
print('data load ok! ')

'one hot encoder'
def rr(x):
    if len(x[0]) == 0:
        return 0
    else:
        return (x[0][0]+1)
for i in range(len(X_train_0)):
    x1 = np.where(d_1 == X_train_0[i][1])
    x2 = np.where(d_2 == X_train_0[i][2])
    x3 = np.where(d_3 == X_train_0[i][3])
    X_train_0[i][1:4] = [x1[0][0]+1,x2[0][0]+1,x3[0][0]+1]
    y  = np.where(t_1 == y_train[i])
    y_train[i] = rr(y)
for i in range(len(X_test_0)):
    x1 = np.where(d_1 == X_test_0[i][1])
    x2 = np.where(d_2 == X_test_0[i][2])
    x3 = np.where(d_3 == X_test_0[i][3])
    X_test_0[i][1:4] = [rr(x1),rr(x2),rr(x3)]
    y  = np.where(t_1 == y_test[i])
    y_test[i] = rr(y)

'standard scalar and minmax ,and normalizer'    
new_c = np.append(c_name[0],c_name[4:41])
new_c_hand = np.array([0, 4,  5,  7,  8,  9, 10, 12, 15, 16, 17, 18, 22, 23,31,32])
ss = StandardScaler()
mm = MinMaxScaler()

ct = ColumnTransformer([('', ss, new_c_hand)], remainder='passthrough')
X_train_s = ct.fit_transform(X_train_0)
X_test_s = ct.fit_transform(X_test_0)
arr = np.arange(len(new_c_hand))
ct = ColumnTransformer([('', mm, arr)], remainder='passthrough')
X_train_m = ct.fit_transform(X_train_s)
X_test_m = ct.fit_transform(X_test_s)
print('ss OK! ')

nor = Normalizer(norm='max')
nor_c = np.array([16,17,18,22])
ct = ColumnTransformer([('', nor, nor_c)], remainder='passthrough')
X_train = ct.fit_transform(X_train_m)
X_test = ct.fit_transform(X_test_m)
print('nor OK!')

X_train_0 = []
X_test_0 = []
X_train_s = []
X_test_s = []
X_train_m = []
X_test_m = []

y_train = y_train.astype(float)
y_test = y_test.astype(float)

def wrong(arr1,arr2):
    l = len(arr1)
    out = np.zeros(l)
    for i in range(l):
        if arr1[i] != arr2[i]:
            out[i] = 1
    return out

def wrong_with_type(y,pre):
    test_acc = np.zeros((5,6))
    #a = np.array(['normal', 'Dos','Probe',  'R2L', 'U2R','All'])
    test_acc[0] = np.array([1,2,3,4,5,6])
    
    for i in range(len(y)):
        x = int(y[i] -1)
        if pre[i] == y[i]:
            test_acc[2][x] += 1
        else:
            test_acc[3][x] += 1
        test_acc[1][x] += 1
    test_acc[1][5] = len(y)
    test_acc[2][5] = sum(test_acc[2])
    test_acc[3][5] = sum(test_acc[3])
    for i in range(6):
        test_acc[4][i] = test_acc[2][i] / test_acc[1][i]
    return test_acc

rfn = 100

t1 = time.time()
rf=RandomForestClassifier(n_estimators=rfn,verbose = 1, n_jobs = 4)  
rf.fit(X_train,y_train)    
print('rf ok!', 'time: ',time.time()-t1)
# Use the forest's predict method on the test data 
predictions = rf.predict(X_test)
# Calculate the absolute errors
score_rf = rf.score(X_test,y_test)
print('Score:', score_rf)
lt = len(y_test)
ac = accuracy_score(y_test,predictions,normalize = False)
err = lt - ac
err_rate = 1-(ac/lt)
print('Error:', err)
print('err_rate:',err_rate)
o1 = wrong(predictions,y_test)

## xgboost test
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
prexg = xgb.predict(X_test)
ac_xg = accuracy_score(y_test,prexg)
##sc = xgb.score(X_test, y_test)
print('xgboost test err : ', 1-ac_xg)
xgw = wrong(prexg,y_test)

np.set_printoptions(suppress=True)
t_1 = wrong_with_type(y_test,predictions)
print('RF in sk-learn: ')
print(t_1,'\n')

t_x = wrong_with_type(y_test,prexg)
print('xgboost: ')
print(t_x,'\n')


print('\nadd generate data:')
data_a = []
with open('g_nslkdd/g_all_test1_epoch_test1.csv','r',newline = '') as csvfile:
    rows = csv.reader(csvfile)
    for row in rows:
        data_a.append(row)
data_a = np.array(data_a)
data_a_y = rf.predict(data_a)

new_x = np.vstack((X_train,data_a))
new_y = np.append(y_train,data_a_y)


rf.fit(new_x,new_y)    
print('rf ok!', 'time: ',time.time()-t1)
# Use the forest's predict method on the test data 
predictions = rf.predict(X_test)
# Calculate the absolute errors
score_rf = rf.score(X_test,y_test)
print('Score:', score_rf)
lt = len(y_test)
ac = accuracy_score(y_test,predictions,normalize = False)
err = lt - ac
err_rate = 1-(ac/lt)
print('Error:', err)
print('err_rate:',err_rate)
o1 = wrong(predictions,y_test)

## xgboost test
xgb = XGBClassifier()
xgb.fit(new_x, new_y)
prexg = xgb.predict(X_test)
ac_xg = accuracy_score(y_test,prexg)
##sc = xgb.score(X_test, y_test)
print('xgboost test err : ', 1-ac_xg)
xgw = wrong(prexg,y_test)

t_1 = wrong_with_type(y_test,predictions)
print('RF in sk-learn: ')
print(t_1,'\n')

t_x = wrong_with_type(y_test,prexg)
print('xgboost: ')
print(t_x,'\n')
