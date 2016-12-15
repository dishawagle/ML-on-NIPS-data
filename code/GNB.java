/**
 * Project Partners: Avruti Srivastava (avrsriva@iu.edu), Disha Wagle (dmwagle@iu.edu)
 */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Scanner;
import java.util.Set;
import mloss.roc.*;
import mloss.roc.Curve.PrimitivesBuilder;

public class GNB {
	public static double p0;
	public static double p1;
	public static ArrayList<ArrayList<Double>> class0;
	public static ArrayList<ArrayList<Double>> class1;
	public static ArrayList<ArrayList<Double>> dataSet = new ArrayList<ArrayList<Double>>();
	public static ArrayList<ArrayList<Double>> attsummaries1;
	public static ArrayList<ArrayList<Double>> attsummaries0;
	public static ArrayList<ArrayList<Double>> trainX;
	public static ArrayList<ArrayList<Double>> testX;
	public static int[] aclabels;
	public static double acfin = 0;
	public static double rocavg = 0;
	public static double tpravg = 0;
	public static double fpravg = 0;
    /**
     * Reads the data      
     * @param fileName
     * @return
     */
	public static ArrayList<ArrayList<Double>> getData(String fileName) {
		ArrayList<ArrayList<Double>> Vectors = new ArrayList<ArrayList<Double>>();
		String line = null;
		FileReader fReader = null;
		try {
			fReader = new FileReader(fileName);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		BufferedReader bufferedReader = new BufferedReader(fReader);
		try {
			int g = 0;
			one: while ((line = bufferedReader.readLine()) != null) {
				if (g == 0) {
					g++;
					continue one;
				}
				ArrayList<Double> temp = new ArrayList<Double>();
				String[] numbers = line.split(",");
				int sz = numbers.length;
				for (int j = 0; j < sz; j++)
					if (numbers[j] != null && !numbers[j].isEmpty()) {
						temp.add(Double.parseDouble(numbers[j]));
					}
				Vectors.add(temp);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		return Vectors;// The data is stored in Vectors
	}
	/**
	 *calculates the class probabilities
	 * @param train_set
	 * @param test_set
	 */

	public static void probpy(ArrayList<ArrayList<Double>> train_set, ArrayList<ArrayList<Double>> test_set) { 
		int lastindex = train_set.get(0).size();
		class0 = new ArrayList<ArrayList<Double>>();
		class1 = new ArrayList<ArrayList<Double>>();
		p1 = 0;
		p0 = 0;

		for (int i = 0; i < train_set.size(); i++) {
			if (train_set.get(i).get(lastindex - 1).equals(1.0)) {
				p1++;
				class1.add(train_set.get(i));
			} else {
				p0++;
				class0.add(train_set.get(i));
			}
		}
		p1 = p1 / train_set.size();
		p0 = p0 / train_set.size();
	}

	public static ArrayList<ArrayList<Double>> setValues(ArrayList<ArrayList<Double>> Vectors) { // pass class0 or class1 for its respective values
		
		ArrayList<ArrayList<Double>> attributeValues = new ArrayList<ArrayList<Double>>();

		for (int i = 0; i < Vectors.get(0).size(); i++) {
			ArrayList<Double> temp1 = new ArrayList<Double>();
			for (int j = 0; j < Vectors.size(); j++) {
				double h = Vectors.get(j).get(i);
				temp1.add(h);

			}
			attributeValues.add(temp1); // stores attribute wise values
		}
		return attributeValues;
	}

	public static double mean(ArrayList<Double> data) { //calculates the mean
		double sum = 0;
		for (double d : data) {
			sum = sum + d;
		}
		return sum / data.size();
	}
 /**
  * It calculates the mean, variance and Standard Deviation. 
  * @param data
  * @return
  */
	public static ArrayList<Double> stats(ArrayList<Double> data) { 
		ArrayList<Double> s = new ArrayList<Double>();
		double mean = mean(data);
		s.add(mean);
		double var = 0;
		for (int i = 0; i < data.size(); i++) {
			double y = 0;
			y = Math.pow(data.get(i) - mean, 2);
			var = var + y;
		}
		var = var / (data.size());
		double std = Math.sqrt(var);
		s.add(std);
		return s;
	}
/**
 * Applies the Gaussian function 
 * @param x
 * @param mean
 * @param std
 * @return
 */
	public static double calc(double x, double mean, double std) {

		double e = Math.exp(-(Math.pow(x - mean, 2) / (2 * Math.pow(std, 2))));
		return (1 / (Math.sqrt(2 * Math.PI) * std)) * e;
	}
    /**
     * Returns the label for a particular data instance
     * @param data
     * @return
     */
	public static double cp(ArrayList<Double> data) {// pass each instance of data to return its label
		double pr1 = 1;
		double pr0 = 1;
		for (int i = 0; i < data.size() - 1; i++) {
			pr1 = pr1 * calc(data.get(i), attsummaries1.get(i).get(0), attsummaries1.get(i).get(1));
		}
		pr1 = pr1* p1;
		for (int i = 0; i < data.size() - 1; i++) {
			pr0 = pr0 * calc(data.get(i), attsummaries0.get(i).get(0), attsummaries0.get(i).get(1));
		}
		pr0 = pr0* p0;

		if (pr1 > pr0)
			return 1.0;
		else
			return 0.0;

	}
 public static double tpa=0;
 public static double tna=0;
 public static double fpa=0;
 public static double fna=0;
 /**
  * Calculates the accuracy, true positive rate, false positive rate and Confusion Matrix for the given testset.
  * @param tSet
  */
 public static void test(ArrayList<ArrayList<Double>> tSet) {
		ArrayList<Double> pred_label = new ArrayList<Double>();
		ArrayList<Double> actual_label = new ArrayList<Double>();

		for (ArrayList ins : tSet)

		{
			double l = cp(ins);
			pred_label.add(l);
		}
		double acc = 0;
		int sz = tSet.get(0).size() - 1;
		double truePos = 0;
		double trueNeg = 0;
		double falsePos = 0;
		double falseNeg = 0;
		for (int i = 0; i < tSet.size(); i++) {
			double A = pred_label.get(i);
			double B = tSet.get(i).get(sz);
			if (A == 1 && B == 1)
				truePos++;
			else if (A == 0 && B == 0)
				trueNeg++;
			else if (A == 0 && B == 1)
				falseNeg++;
			else if (A == 1 && B == 0)
				falsePos++;
			actual_label.add(B);
		}
		acc = (truePos + trueNeg) / tSet.size();
		double tpr = truePos / (truePos + falseNeg);
		tpa=tpa+truePos;
		tna=tna+trueNeg;
		fna=fna+falseNeg;
		fpa=fpa+falsePos;
		tpravg = tpravg + tpr;
		double fpr = falsePos / (trueNeg + falsePos);
		fpravg = fpravg + fpr;
		acfin = acfin + acc;
		double a = ROC(pred_label, actual_label);
		rocavg = rocavg + a;
	}
 /**
  * Performs cross-validation and gives the average accuracy, tpr, fpr and Confusion Matrix
  */
	public static void cv() {
		int s = dataSet.size();
		int testSize = (int) (s * 0.1);
		int testStart = s - testSize;
		for (int i = 0; i < 10; i++) {
			trainX = new ArrayList<ArrayList<Double>>();
			testX = new ArrayList<ArrayList<Double>>();
			for (int ij = 0; ij < s; ij++) {
				if (ij >= testStart && ij < (testStart + testSize)) {
					testX.add(dataSet.get(ij));

				} else {

					trainX.add(dataSet.get(ij));
				}
			}
			testStart -= testSize;
			GNB(trainX, testX);

		}
		
		System.out.println("\nRESULTS: ");
		System.out.println("\nConfusion Matrix:");
		  System.out.println("\tPredicted TRUE \t Predicted FALSE\n");
		  System.out.println("Actual TRUE\t" + tpa/10 + "\t" + fna/10 + "\n");
		  System.out.println("Actual FALSE\t" + fpa/10 + "\t" + tna/10 + "\n");
		  System.out.println("Accuracy: " + acfin /10);
		System.out.println("True positive rate: " + tpravg /10);
		System.out.println("False positive rate: " + fpravg / 10);
		System.out.println("Area under the curve(ROC): " + rocavg / 10);
	}

	public static void GNB(ArrayList<ArrayList<Double>> train_set, ArrayList<ArrayList<Double>> test_set) {
		ArrayList<ArrayList<Double>> t1 = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> t0 = new ArrayList<ArrayList<Double>>();
		attsummaries1 = new ArrayList<ArrayList<Double>>();// Attribute summaries for class zero
		attsummaries0 = new ArrayList<ArrayList<Double>>();// Attribute summaries for class one

		probpy(train_set, test_set);
		//class wise, attribute wise values
		t0 = setValues(class0);
		t1 = setValues(class1);
		int count = 0;

		for (ArrayList<Double> x : t0) {
			if (count < t0.size() - 1) {
				ArrayList<Double> r = new ArrayList<Double>();
				r = stats(x);
				attsummaries0.add(r); 
				count++;
			}
		}
		count = 0;
		for (ArrayList<Double> x : t1) {
			if (count < t1.size() - 1) {
				ArrayList<Double> r = new ArrayList<Double>();
				r = stats(x);
				attsummaries1.add(r);
				count++;
			}
		}
		test(test_set);
	}
/**
 * Returns the area under ROC curve
 * @param pred_label
 * @param actual_label
 * @return
 */
	public static double ROC(ArrayList<Double> pred_label, ArrayList<Double> actual_label) {

		double[] pl = new double[pred_label.size()];
		Integer[] al = new Integer[actual_label.size()];
		for (int i = 0; i < pred_label.size(); i++) {
			pl[i] = pred_label.get(i);
			al[i] = actual_label.get(i).intValue();

		}
		Iterable<Integer> al1 = Arrays.asList(al);
		Curve analysis = new Curve.PrimitivesBuilder().predicteds(pred_label).actuals(al1).build();
		double area = analysis.rocArea();
		return area;

	}

	public static void main(String[] args) throws FileNotFoundException {

		Scanner sc= new Scanner(System.in);
		System.out.println("Enter data path:");
		String datapath=sc.next();
		System.out.println("Running.. ");
		dataSet = getData(datapath);
		
		cv();
	}

}
