/**
 * Project Partners: Avruti Srivastava(avrsriva@iu.edu), Disha Wagle(dmwagle@iu.edu)
 */
package amlproj;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;

import mloss.roc.Curve;


public class LogisticRegression {
	
	//entire dataset without labels
	public static ArrayList<ArrayList<Double>> data=new ArrayList<ArrayList<Double>>(); 
	//class labels
	public static ArrayList<Double> label=new ArrayList<Double>(); 	
	//training set features
	public static ArrayList<ArrayList<Double>> trainX; 		
	//test set features
	public static ArrayList<ArrayList<Double>> testX; 		
	//training set labels
	public static ArrayList<Double> trainY; 
	//test set labels
	public static ArrayList<Double> testY;
	public static double weights[];
	public static double learningRate=0.1;
	public static int reg_option;
	/**
	 * Parses data from file and stores it in the data structures
	 * 
	 * 
	 * @param data file path
	 * @return
	 */
	public static void getData(String path) throws IOException
	{
		File f1 = new File(path);
		int sz = 0;
			String line = null;
			int f=0;
			BufferedReader bufferedReader1 = new BufferedReader(new FileReader(f1));
			while ((line = bufferedReader1.readLine()) != null) 
			{
				if(f==0)
				{
					f=1;
					continue;
				}
				ArrayList<Double> temp = new ArrayList<Double>();
				String[] numbers = line.split(",");
				sz = numbers.length - 1;
				for (int j = 0; j < sz; j++)
					if (numbers[j] != null && !numbers[j].isEmpty()) {
						temp.add(Double.parseDouble(numbers[j]));
					}
				label.add(Double.parseDouble(numbers[sz]));
				data.add(temp);
			}
			weights=new double[sz];
			}
	/**
	 * Updates the weights while iterating through each training instance
	 * 
	 * @return
	 */
	public static void updateWeights()
	{
		for(int i=0;i<100;i++)
			{
				double likelihood=0.0;
				for(int j=0;j<trainX.size();j++)
				{
					double hy=classifier(trainX.get(j));
					double y=trainY.get(j);
					double t=learningRate*(y-hy);
					
					for(int k=0;k<weights.length;k++)
					{
						
						if(reg_option==2)
							//weight update with regularization
							weights[k]=weights[k]+t*trainX.get(j).get(k)-(learningRate*weights[k]*0.6); 
						else
							//weight update without regularization
							weights[k]=weights[k]+t*trainX.get(j).get(k);
					}
					likelihood+=(y*Math.log(classifier(trainX.get(j)))+(1-y)*Math.log(1-classifier(trainX.get(j))));
				
					}
			}
		}
	/**
	 * Calculates hypothesis using weights for the given data 
	 * 
	 * 
	 * @param Training/test instance
	 * @return sigmoid function output
	 */
	public static double classifier(ArrayList<Double> x)
		{
			double z=0;
			for(int i=0;i<weights.length;i++)
			{
				z+=weights[i]*x.get(i);
			}
			//returning P=(Y=1|X)
			return 1.0 / (1.0 + Math.exp(-z));
		}
		
	/**
	 * Calculates Area under the Curve
	 * 
	 * 
	 * @param list of class label predictions
	 * @return AUC ROC
	 */
	public static double ROC(ArrayList<Double> pred_label) 
		{

			double[] pl = new double[pred_label.size()];
			Integer[] al = new Integer[testY.size()];
			for (int i = 0; i < pred_label.size(); i++) {
				pl[i] = pred_label.get(i);
				al[i] = testY.get(i).intValue();
			}
			Iterable<Integer> al1 = Arrays.asList(al);
			Curve analysis = new Curve.PrimitivesBuilder().predicteds(pred_label).actuals(al1).build();
			double area = analysis.rocArea();
			return area;

		}

	/**
	 * Performs 10-fold cross validation and rest of training and testing
	 * Also, calculates final outputs
	 * 
	 * @param 
	 * @return
	 */
	public static void test()
	{
		int totalSize=data.size();
		double totalAccuracy=0.0;
		double tpravg=0.0;
		double fpravg=0.0;
		double rocavg=0.0;
		double tp=0.0;
		double fp=0.0;
		double tn=0.0;
		double fn=0.0;
		
		int testSize=(int)(totalSize*0.1);
		int testStart=totalSize-testSize;
		//10-fold cross validation
		for (int i=0;i<10;i++)
		{
		int l=0;
		trainX=new ArrayList<ArrayList<Double>>(); 
		testX=new ArrayList<ArrayList<Double>>(); 
		trainY=new ArrayList<Double>(); 
		testY=new ArrayList<Double>();
		for(int ij=0;ij<totalSize;ij++)
		{
			if(ij>=testStart&&ij<(testStart+testSize))
			{
			//preparing test set for the iteration
			testX.add(data.get(ij));
			testY.add(label.get(ij));
			}
			else
			{	
				//preparing training set for the iteration
				trainX.add(data.get(ij));
				trainY.add(label.get(ij));
			}
				
		}
		testStart-=testSize;
		
		updateWeights();
		int c=0;
		ArrayList<Double> pred=new ArrayList<Double>();
		for(int a=0;a<testX.size();a++)
		{
			//getting the predictions on test set
			if(classifier(testX.get(a))>=0.5)
				pred.add(1.0);
			else
				pred.add(0.0);
			
		}

		
		
		//calculating confusion matrix, accuracy, true positive rate and false positive rate
		double acc = 0;
		double truePos = 0;
		double trueNeg = 0;
		double falsePos = 0;
		double falseNeg = 0;
		for (int ik = 0; ik < testY.size(); ik++) {
			double A = pred.get(ik);
			double B = testY.get(ik);
			if (A == 1 && B == 1)
				truePos++;
			else if (A == 0 && B == 0)
				trueNeg++;
			else if (A == 0 && B == 1)
				falseNeg++;
			else if (A == 1 && B == 0)
				falsePos++;
		}
		acc = (truePos + trueNeg) / testY.size();
		tp+=truePos;
		fp+=falsePos;
		tn+=trueNeg;
		fn+=falseNeg;
		double tpr = truePos / (truePos + falseNeg);
		tpravg = tpravg + tpr;
		double fpr = falsePos / (trueNeg + falsePos);
		fpravg = fpravg + fpr;
		totalAccuracy +=acc;
		double a = ROC(pred);
		rocavg = rocavg + a;
	}
		
		System.out.println("\nRESULTS:");
		System.out.println("\n\nAverage Confusion Matrix");
		System.out.println("\tPredicted TRUE \t Predicted FALSE\n");
		System.out.println("Actual TRUE\t" + tp/10 + "\t" + fn/10 + "\n");
		System.out.println("Actual FALSE\t" + fp/10 + "\t" + tn/10 + "\n");
		System.out.println("Average Accuracy: " + totalAccuracy / 10);
		System.out.println("Average True positive rate: " + tpravg / 10);
		System.out.println("Average False positive rate: " + fpravg / 10);
		System.out.println("Average Area under the curve(ROC): " + rocavg / 10);


	}


	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		System.out.println("Enter the file path");
		Scanner sc=new Scanner(System.in);
		String path=sc.next();
		System.out.println("\nSelect one of the following (Press 1 or 2):");
		System.out.println("\n1.Logistic Regression without Regularization ");
		System.out.println("2.Logistic Regression with Regularization ");
		reg_option=sc.nextInt();
		System.out.println("Running...");
		getData(path);
		test();
		
		
		

	}

}

