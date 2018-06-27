import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from scipy import interp



def readCsv():
    df = pd.DataFrame(pd.read_csv('new_dota_data.csv',header=0,sep=','))
    all_match_count = df.shape[0];
    print df.info()
    print 'all_match_count = ' , all_match_count
    print df

    X_data = np.zeros(shape=(all_match_count, 2 * 120+1))
    y_data = np.zeros(all_match_count)

    synergyAllCountMatrix = np.zeros(shape=(120, 120))
    synergyWinCountMatrix = np.zeros(shape=(120, 120))
    counteringAllCountMatrix = np.zeros(shape=(120, 120))
    counteringWinCountMatrix = np.zeros(shape=(120, 120))
    hero_all_count = np.zeros(120)
    hero_win_count = np.zeros(120)

    radiant_win_sum = df.iloc[:,11].sum()
    print 'radiant_win_sum = ', radiant_win_sum

    numTemp = 0
    for indexs in df.index:
        tempRow = df.loc[indexs].values[0:-1]
        for i in range(1, 6):
            for j in range(i + 1, 6):
                synergyAllCountMatrix[tempRow[i]-1][tempRow[j]-1] += 1
                synergyAllCountMatrix[tempRow[j]-1][tempRow[i]-1] += 1
        for i in range(6, 11):
            for j in range(i + 1, 11):
                synergyAllCountMatrix[tempRow[i]-1][tempRow[j]-1] += 1
                synergyAllCountMatrix[tempRow[j]-1][tempRow[i]-1] += 1
        for i in range(1,6):
            for j in range(6,11):
                counteringAllCountMatrix[tempRow[i]-1][tempRow[j]-1] += 1;
        if tempRow[11] == 1:
            X_data[numTemp][240]=1
            y_data[numTemp] = 1
            for i in range(1,6):
                hero_win_count[tempRow[i]-1] +=1
                for j in range(i+1,6):
                    synergyWinCountMatrix[tempRow[i]-1][tempRow[j]-1] += 1
                    synergyWinCountMatrix[tempRow[j]-1][tempRow[i]-1] += 1
            for i in range(1, 6):
                for j in range(6, 11):
                    counteringWinCountMatrix[tempRow[i] - 1][tempRow[j] - 1] += 1;
        else:
            X_data[numTemp][240] = 1
            y_data[numTemp] = 0
            for i in range(6,11):
                for j in range(i+1,11):
                    synergyWinCountMatrix[tempRow[i]-1][tempRow[j]-1] += 1
                    synergyWinCountMatrix[tempRow[j]-1][tempRow[i]-1] += 1
            for i in range(6,11):
                hero_win_count[tempRow[i]-1] +=1

        for i in range(1,10):
            hero_all_count[tempRow[i]-1] +=1
        for i in range(1,6):
            X_data[numTemp][tempRow[i]-1] =1
            X_data[numTemp][tempRow[i+5]-1+120]=1

        numTemp+=1

    synergyMatrix = np.zeros(shape=(120, 120))
    for i in range(0,synergyAllCountMatrix.shape[0]):
        for j in range(0,synergyAllCountMatrix.shape[0]):
            if synergyAllCountMatrix[i][j] == 0:
                synergyMatrix[i][j] = 0
            else:
                synergyMatrix[i][j] = synergyWinCountMatrix[i][j]/synergyAllCountMatrix[i][j]
    counteringMatrix = np.zeros(shape=(120,120))
    for i in range(0,counteringMatrix.shape[0]):
        for j in range(0,i+1):
            if counteringAllCountMatrix[i][j] == 0:
                counteringMatrix[i][j] = 0
                counteringMatrix[j][i] = 0
            else:
                counteringMatrix[i][j] = counteringWinCountMatrix[i][j]/counteringAllCountMatrix[i][j]
                counteringMatrix[j][i] = 1 - counteringMatrix[i][j]
    print synergyMatrix
    print counteringMatrix

    hero_win_p = np.zeros(120)
    for i in range(1,120):
        if hero_all_count[i] == 0:
            hero_win_p[i] = 0
        else:
            hero_win_p[i] = hero_win_count[i]/hero_all_count[i]


    # plt.imshow(synergyMatrix, interpolation='bessel', cmap='cool', origin='upper')
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    # plt.imshow(counteringMatrix, interpolation='bessel', cmap='cool', origin='upper')
    # plt.colorbar()
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    #
    # labels = 'radiantWin', 'direWin'
    # radiantWin_p = (float)(radiant_win_sum)/all_match_count
    # fracs = [radiantWin_p,1-radiantWin_p]
    # explode = [0,0]
    # plt.axes(aspect=1)
    # plt.pie(x=fracs, labels=labels, explode=explode, autopct='%3.1f %%',
    #         shadow=True, labeldistance=1.1, startangle=90, pctdistance=0.6)
    # plt.show()
    #
    #
    # print hero_all_count
    # print hero_win_count
    # print hero_win_p
    # num_list = hero_win_p
    # plt.bar(range(len(num_list)), num_list)
    # plt.show()

    print X_data
    print y_data

    skf = StratifiedKFold(n_splits=10)
    precision = 0
    recall = 0
    f_score = 0
    for train_index, test_index in skf.split(X_data, y_data):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X_data[train_index], X_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        #model = LogisticRegression(C=0.005, random_state=42)
        #model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1)
        model = svm.SVC(kernel='rbf', probability=True)
        model.fit(X_train,y_train)
        predict = model.predict(X_test)
        precision += precision_score(predict, y_test, average='binary')
        recall += recall_score(predict, y_test, average='binary')
        f_score += f1_score(predict, y_test, average='binary')
    print precision/10
    print recall/10
    print f_score/10

    #x_train, x_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=42)

    # LR
    # model = LogisticRegression(C=0.005, random_state=42)
    # model.fit(x_train, y_train)
    # probabilities = model.predict_proba(x_test)
    # lr_prodict = model.predict(x_test)
    # #print lr_prodict
    # #print y_test
    # print precision_score(lr_prodict, y_test, average='binary')
    # print recall_score(lr_prodict, y_test, average='binary')
    # print f1_score(lr_prodict, y_test, average='binary')
    # mean_tpr = 0.0
    # mean_fpr = np.linspace(0, 1, 100)
    # fpr, tpr, thresholds = roc_curve(y_test, probabilities[:, 1])
    # mean_tpr += interp(mean_fpr, fpr, tpr)
    # mean_tpr[0] = 0.0
    # roc_auc = auc(fpr, tpr)
    # plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


    #SVM
    # clf = svm.SVC(kernel='rbf', probability=True)
    # clf.fit(x_train,y_train)
    # svm_predicted = clf.predict(x_test)
    # #svm_p = clf.predict_proba(x_test)
    # print precision_score(svm_predicted, y_test, average='binary')
    # print recall_score(svm_predicted, y_test, average='binary')
    # print f1_score(svm_predicted, y_test, average='binary')
    # mean_tprSVM = 0.0
    # mean_fprSVM = np.linspace(0, 1, 100)
    # all_tprSVM = []
    # fprSVM, tprSVM, thresholdsSVM = roc_curve(y_test, svm_p[:, 1])
    # mean_tprSVM += interp(mean_fprSVM, fprSVM, tprSVM)
    # mean_tprSVM[0] = 0.0
    # roc_aucSVM = auc(fprSVM, tprSVM)
    # plt.plot(fprSVM, tprSVM, lw=1, label='ROC SVM (area = %0.2f)' %  roc_aucSVM)
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()


    #Adaboost
    # model2 = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1)
    # model2.fit(x_train, y_train)
    # adaboost_prodict = model2.predict(x_test)
    # adaP = model2.predict_proba(x_test)
    # print precision_score(adaboost_prodict, y_test, average='binary')
    # print recall_score(adaboost_prodict, y_test, average='binary')
    # print f1_score(adaboost_prodict, y_test, average='binary')
    # mean_tpr2 = 0.0
    # mean_fpr2 = np.linspace(0, 1, 100)
    # fpr2, tpr2, thresholds2 = roc_curve(y_test, adaP[:, 1])
    # mean_tpr2 += interp(mean_fpr2, fpr2, tpr2)
    # mean_tpr2[0] = 0.0
    # roc_auc2 = auc(fpr2, tpr2)
    # plt.plot(fpr2, tpr2, lw=1, label=' Adaboost ROC (area = %0.2f)' % roc_auc2)
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    # plt.show()



#def sklearn(X_data,y_data,model):



if __name__ == "__main__":
    readCsv()
