The code for the CoSTCo and CP-WOPT tensor completion models are in costco.py
A tutorial explaining CoSTCo is in costco_tutorial.py
costco_opt.py and costco_test.py find the optimal hyperparameters for costco and then test in on the test data set respectivly
cp_wopt_opt.py and cp_wopt_test.py find the optimal hyperparameters for cp_wopt_opt and then test in on the test data set respectivly
Results are stored in CoSTCo_results and FL_CoSTCo_results(this file is just copied from CoSTCo_results for the time being)
middle_results are results from previous iterations of the models
costco_vs_fl_costco_bar_plot.py creates a bar plot comparing costco and FL-CoSTCo


FL-CoSTCo Psuedo-Algorithm

Create a single CoSTCo model
Partition the train data set into each of the users data
For each users data:
	Use tensorflow gradient tape to get the gradients using the users data as a batch
	Update the users row in the user factor matrix using its gradient
	Remove that gradient from the list of gradients
	Encrypt the remaining gradients
	Add noise to the remaining gradients
	Add to a list of encrypted, disorted gradients
Combine the encrypted disorted gradients
Update the model
Repeat until convergance
