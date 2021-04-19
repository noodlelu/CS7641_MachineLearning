X_train, X_test, y_train, y_test = train_test_split(np.array(loanX),np.array(loanY), test_size=0.20)
full_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
train_samp_full, NN_train_score_full, NN_fit_time_full, NN_pred_time_full = plot_learning_curve(full_est, X_train, y_train,title="Neural Net Loan: Full")
final_classifier_evaluation(full_est, X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(np.array(pca_loan),np.array(loanY), test_size=0.20)
pca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
train_samp_pca, NN_train_score_pca, NN_fit_time_pca, NN_pred_time_pca = plot_learning_curve(pca_est, X_train, y_train,title="Neural Net Loan: PCA")
final_classifier_evaluation(pca_est, X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(np.array(rca_loan),np.array(loanY), test_size=0.20)
rca_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
train_samp_rca, NN_train_score_rca, NN_fit_time_rca, NN_pred_time_rca = plot_learning_curve(rca_est, X_train, y_train,title="Neural Net Loan: RCA")
final_classifier_evaluation(rca_est, X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = train_test_split(np.array(rfc_loan),np.array(loanY), test_size=0.20)
rfc_est = MLPClassifier(hidden_layer_sizes=(50,), solver='adam', activation='logistic', learning_rate_init=0.01, random_state=100)
train_samp_rfc, NN_train_score_rfc, NN_fit_time_rfc, NN_pred_time_rfc = plot_learning_curve(rfc_est, X_train, y_train,title="Neural Net Loan: RFC")
final_classifier_evaluation(rfc_est, X_train, X_test, y_train, y_test)
