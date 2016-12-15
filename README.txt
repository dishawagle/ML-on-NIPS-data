Project Partners: Avruti Srivastava(avrsriva@iu.edu), Disha Wagle(dmwagle@iu.edu)


We have created two Jar files: LogisticRegression.jar and GNB.jar(for Gaussian Naive Bayes)

We have made use of WEKA API in java in order to perform 'Principal Component Analysis'. We have provided the source code for the same, however, not the Jar file for PCA, as it takes several hours to run and return the transformed data for the large datasets (You can test PCA.java file). Therefore, along with the code and original datsets (madelon(original).csv and gisette(original).csv), we have provided the transformed data for two variations of each of the original datasets directly:
{madelon_top40.csv: containing the top 40 attributes received after PCA(transformed data),
 madelon_top80.csv: containing the top 80 attributes received after PCA(transformed data),
 gisette_top80.csv: containing the top 80 attributes received after PCA(transformed data), 
 gisette_top350.csv: containing the top 350 attributes received after PCA(transformed data)}
 


'No commandLine arguments' are to be passed to the jar files. The program prompts the user for the different inputs required. It gives as ouput: 'Average Confusion Matrix', 'Accuracy', 'True Positive Rate', 'False Positive Rate' and 'Area under the ROC curve'.(As mentioned in the miniproject.pdf file provided by the professor).

 The LogisticRegression.jar provides two options: 1.Run LR with Regularization and 2. Run LR without Regulaization. Select either of two when prompted.

   >For Madelon Dataset(for original as well as transformed Datasets provided):

	Example:

	1)for GNB.jar :

		C:\Users\avruti\Desktop> java -jar GNB.jar	
		Enter data path:

	2)for LogisticRegression.jar :

		C:\Users\avruti\Desktop> java -jar LogisticRegression.jar
		Enter the file path:
		C:\Users\avruti\Desktop\amlproj\data\Madelon(original).csv
		Select one of the following (Press 1 or 2):

		1.Logistic Regression without Regularization
		2.Logistic Regression with Regularization
		1

                
   >For Gisette Dataset:

	For the original dataset: 'gisette(original).csv', as it contains a large number of features, we need to increase the heap size for GNB.jar(only), therefore we pass -Xmx3G command		as follows:

 	1)for gisette(original).csv using GNB.jar :

		C:\Users\avruti\Desktop> java -Xmx3G -jar GNB.jar	
		Enter data path:

	
	2) for the transformed Gisette datasets(gisette_top80.csv and gisette_top350.cv), we need not increase the memory heap size as the number of features have reduced significantly.
		
		C:\Users\avruti\Desktop> java -jar GNB.jar	
		Enter data path:

	3) for LogisticRegression.jar (for all variations of the gisette datset):

		C:\Users\avruti\Desktop> java -jar LogisticRegression.jar
		Enter data path: 
		C:\Users\avruti\Desktop\amlproj\data\gisette(original).jar
		Select one of the following (Press 1 or 2):

		1.Logistic Regression without Regularization
		2.Logistic Regression with Regularization
		1
  