/**
 * Project Partners: Avruti Srivastava(avrsriva@iu.edu), Disha Wagle(dmwagle@iu.edu)
 */
package amlproj;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import java.util.Set;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GreedyStepwise;
import weka.attributeSelection.Ranker;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.core.converters.ArffLoader;
import weka.core.converters.CSVLoader;
import weka.core.converters.CSVSaver;
import weka.filters.Filter;
import weka.filters.UnsupervisedFilter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.gui.beans.DataSource;


public class pca {
	


	public static void main(String[] args) throws Exception {
		ArffLoader loader = new ArffLoader();
		loader.setSource(new File("C:\\Users\\Disha\\Desktop\\madelon_original.arff"));
		Instances instance = loader.getDataSet();
		instance.setClassIndex(instance.numAttributes()-1);
		PrincipalComponents pcaObj = new PrincipalComponents(); 
		pcaObj.setInputFormat(instance);
		Instances newData = Filter.useFilter(instance, pcaObj);
		CSVSaver saver = new CSVSaver();
		saver.setInstances(newData);
		saver.setFile(new File("C:\\Users\\Disha\\Desktop\\madelon_train_final.csv"));
		saver.writeBatch();



	}

}

