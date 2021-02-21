def hyperSVM(X_train, y_train, X_test, y_test, title):

    f1_test = []
    f1_train = []
    kernel_func = ['rbf,'sigmod']
    for i in kernel_func:         
        clf = SVC(kernel=i, random_state=100)
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
                
    xvals = ['linear','sigmod']
    plt.plot(xvals, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(xvals, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('Kernel Function')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
def SVMGridSearchCV(X_train, y_train):
    #parameters to search:
    #penalty parameter, C
    #
    Cs = [1e-4, 1e-3, 1e-2, 1e01, 1]
    gammas = [1,10,100]
    param_grid = {'C': Cs, 'gamma': gammas}

    clf = GridSearchCV(estimator = SVC(kernel='rbf',random_state=100),
                       param_grid=param_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(clf.best_params_)
    return clf.best_params_['C'], clf.best_params_['gamma']


loanX,loanY,telescopeX,telescopeY = import_data()

X_train, X_test, y_train, y_test = train_test_split(np.array(loanX),np.array(loanY), test_size=0.20)
hyperSVM(X_train, y_train, X_test, y_test,title="Model Complexity Curve for SVM (Loan Data)\nHyperparameter : Kernel Function")
C_val, gamma_val = SVMGridSearchCV(X_train, y_train)
estimator_loan = SVC(C=C_val, gamma=gamma_val, kernel='rbf', random_state=100)
train_samp_loan, SVM_train_score_loan, SVM_fit_time_loan, SVM_pred_time_loan = plot_learning_curve(estimator_loan, X_train, y_train,title="SVM Loan Data")
final_classifier_evaluation(estimator_loan, X_train, X_test, y_train, y_test)


def SVMGridSearchCV(X_train, y_train):
    #parameters to search:
    #penalty parameter, C
    #
    Cs = [1e-4, 1e-3, 1e-2, 1e01, 1]
    gammas = [1,10,100]
    param_grid = {'C': Cs, 'gamma': gammas}

    clf = GridSearchCV(estimator = SVC(kernel='sigmoid',random_state=100),
                       param_grid=param_grid, cv=10)
    clf.fit(X_train, y_train)
    print("Per Hyperparameter tuning, best parameters are:")
    print(clf.best_params_)
    return clf.best_params_['C'], clf.best_params_['gamma']
                   
X_train, X_test, y_train, y_test = train_test_split(np.array(telescopeX),np.array(telescopeY), test_size=0.20)
hyperSVM(X_train, y_train, X_test, y_test,title="Model Complexity Curve for SVM (Telescope Data)\nHyperparameter : Kernel Function")
C_val, gamma_val = SVMGridSearchCV(X_train, y_train)
estimator_telescope = SVC(C=C_val, gamma=gamma_val, kernel='sigmoid', random_state=100)
estimator_telescope = SVC(kernel='sigmoid', random_state=100)
train_samp_telescope, SVM_train_score_telescope, SVM_fit_time_telescope, SVM_pred_time_telescope = plot_learning_curve(estimator_telescope, X_train, y_train,title="SVM Telescope Data")
final_classifier_evaluation(estimator_telescope, X_train, X_test, y_train, y_test)
