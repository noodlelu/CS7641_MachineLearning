def hyperKNN(X_train, y_train, X_test, y_test, title):
    
    f1_test = []
    f1_train = []
    klist = np.linspace(1,250,25).astype('int')
    for i in klist:
        clf = kNN(n_neighbors=i,n_jobs=-1)
        clf.fit(X_train,y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)
        f1_test.append(f1_score(y_test, y_pred_test))
        f1_train.append(f1_score(y_train, y_pred_train))
        
    plt.plot(klist, f1_test, 'o-', color='r', label='Test F1 Score')
    plt.plot(klist, f1_train, 'o-', color = 'b', label='Train F1 Score')
    plt.ylabel('Model F1 Score')
    plt.xlabel('No. Neighbors')
    
    plt.title(title)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


loanX,loanY,telescopeX,telescopeY = import_data()

X_train, X_test, y_train, y_test = train_test_split(np.array(loanX),np.array(loanY), test_size=0.20)
hyperKNN(X_train, y_train, X_test, y_test,title="Model Complexity Curve for kNN (Loan Data)\nHyperparameter : No. Neighbors")
estimator_loan = kNN(n_neighbors=20, n_jobs=-1)
train_samp_loan, kNN_train_score_loan, kNN_fit_time_loan, kNN_pred_time_loan = plot_learning_curve(estimator_loan, X_train, y_train,title="kNN Loan Data")
final_classifier_evaluation(estimator_loan, X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(np.array(telescopeX),np.array(telescopeY), test_size=0.20)
hyperKNN(X_train, y_train, X_test, y_test,title="Model Complexity Curve for kNN (Telescope Data)\nHyperparameter : No. Neighbors")
estimator_telescope = kNN(n_neighbors=10, n_jobs=-1)
train_samp_telescope, kNN_train_score_telescope, kNN_fit_time_telescope, kNN_pred_time_telescope = plot_learning_curve(estimator_telescope, X_train, y_train,title="kNN Telescope Data")
final_classifier_evaluation(estimator_telescope, X_train, X_test, y_train, y_test)
