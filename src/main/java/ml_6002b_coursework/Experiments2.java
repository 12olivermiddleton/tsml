package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import tsml.classifiers.shapelet_based.ShapeletTree;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

// Main class for running experiments

public class Experiments2 {


    public static String data = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/UCI Discrete/UCI Discrete/UCI Discrete";

    public static void testJ48Classifier() throws Exception {
        DatasetLists dataSets = new DatasetLists();

        for(String set: dataSets.nominalAttributeProblems){
            String path = data + "/" + set + "/" + set + ".arff";
            Instances dataSet;
            dataSet = DatasetLoading.loadData(path);
            dataSet.setClassIndex(dataSet.numAttributes() -1);
            System.out.println("Relation name"+ dataSet.relationName());
            System.out.println("Num Instances: "+dataSet.numInstances());
            System.out.println("Num Classes: "+dataSet.numClasses()+"\n");

            dataSet.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
            Instances dataSetTrainData = dataSet.trainCV(30, 29);
            Instances dataSetTestData = dataSet.testCV(30, 29);
            Classifier optClassifier = new J48();
            optClassifier.buildClassifier(dataSetTrainData);
            Evaluation evalDataSet = new Evaluation(dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTestData);
            System.out.println(dataSet.relationName());
            System.out.println("1-NN accuracy for test data set: \n"+ evalDataSet.pctCorrect()/100);
            System.out.println("opt\n"+evalDataSet.toSummaryString());
        }



    }

    public static void testID3() throws Exception {
        DatasetLists dataSets = new DatasetLists();

        for(String set: dataSets.nominalAttributeProblems){
            String path = data + "/" + set + "/" + set + ".arff";
            Instances dataSet;
            dataSet = DatasetLoading.loadData(path);
            dataSet.setClassIndex(dataSet.numAttributes() -1);
//            System.out.println(dataSet.toSummaryString());

            dataSet.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
            Instances dataSetTrainData = dataSet.trainCV(30, 29);
            Instances dataSetTestData = dataSet.testCV(30, 29);
            Classifier optClassifier = new Id3();
            optClassifier.buildClassifier(dataSetTrainData);
            Evaluation evalDataSet = new Evaluation(dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTestData);
            System.out.println(dataSet.relationName());
            System.out.println("1-NN accuracy for test data set: \n"+ evalDataSet.pctCorrect()/100);
            System.out.println("opt\n"+evalDataSet.toSummaryString());
        }
    }

    public static void testRandomForest() throws Exception {
        DatasetLists dataSets = new DatasetLists();

        for(String set: dataSets.nominalAttributeProblems){
            String path = data + "/" + set + "/" + set + ".arff";
            Instances dataSet;
            dataSet = DatasetLoading.loadData(path);
            dataSet.setClassIndex(dataSet.numAttributes() -1);
//            System.out.println(dataSet.toSummaryString());

            dataSet.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
            Instances dataSetTrainData = dataSet.trainCV(30, 29);
            Instances dataSetTestData = dataSet.testCV(30, 29);
            Classifier optClassifier = new RandomForest();
            optClassifier.buildClassifier(dataSetTrainData);
            Evaluation evalDataSet = new Evaluation(dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTestData);
            System.out.println(dataSet.relationName());
            System.out.println("1-NN accuracy for test data set: \n"+ evalDataSet.pctCorrect()/100);
            System.out.println("opt\n"+evalDataSet.toSummaryString());
        }
    }

    public static void testJ48Ensemble() throws Exception {
        DatasetLists dataSets = new DatasetLists();

        for(String set: dataSets.nominalAttributeProblems){
            String path = data + "/" + set + "/" + set + ".arff";
            Instances dataSet;
            dataSet = DatasetLoading.loadData(path);
            dataSet.setClassIndex(dataSet.numAttributes() -1);
//            System.out.println(dataSet.toSummaryString());

            dataSet.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
            Instances dataSetTrainData = dataSet.trainCV(30, 29);
            Instances dataSetTestData = dataSet.testCV(30, 29);
            Classifier optClassifier = new TreeEnsemble();
            optClassifier.buildClassifier(dataSetTrainData);
            Evaluation evalDataSet = new Evaluation(dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTestData);
            System.out.println(dataSet.relationName());
            System.out.println("1-NN accuracy for test data set: \n"+ evalDataSet.pctCorrect()/100);
            System.out.println("opt\n"+evalDataSet.toSummaryString());
        }
    }

    public static void testJ48TreeAttStats() throws Exception {
        DatasetLists dataSets = new DatasetLists();

        for(String set: dataSets.nominalAttributeProblems){
            String path = data + "/" + set + "/" + set + ".arff";
            Instances dataSet;
            dataSet = DatasetLoading.loadData(path);
            dataSet.setClassIndex(dataSet.numAttributes() -1);
//            System.out.println(dataSet.toSummaryString());

            dataSet.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
            Instances dataSetTrainData = dataSet.trainCV(30, 29);
            Instances dataSetTestData = dataSet.testCV(30, 29);
            J48 optClassifier = new J48();
            optClassifier.setOptions(new String[] { "-C", "0.25", "-M", "2" });
            optClassifier.buildClassifier(dataSetTrainData);
            Evaluation evalDataSet = new Evaluation(dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTestData);
            System.out.println(dataSet.relationName());
            System.out.println("1-NN accuracy for test data set: \n"+ evalDataSet.pctCorrect()/100);
            System.out.println("opt\n"+evalDataSet.toSummaryString());
        }
    }

    public static void testShapeletTransform() throws Exception {
        DatasetLists dataSets = new DatasetLists();

        for(String set: dataSets.nominalAttributeProblems){
            String path = data + "/" + set + "/" + set + ".arff";
            Instances dataSet;
            dataSet = DatasetLoading.loadData(path);
            dataSet.setClassIndex(dataSet.numAttributes() -1);
//            System.out.println(dataSet.toSummaryString());

            dataSet.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
            Instances dataSetTrainData = dataSet.trainCV(30, 29);
            Instances dataSetTestData = dataSet.testCV(30, 29);
            Classifier optClassifier = new ShapeletTree();
            optClassifier.buildClassifier(dataSetTrainData);
            Evaluation evalDataSet = new Evaluation(dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTrainData);
            evalDataSet.evaluateModel(optClassifier, dataSetTestData);
            System.out.println(dataSet.relationName());
            System.out.println("1-NN accuracy for test data set: \n"+ evalDataSet.pctCorrect()/100);
            System.out.println("opt\n"+evalDataSet.toSummaryString());
        }
    }




    public static void main(String[] args) throws Exception {
        testJ48Classifier();
        testID3();
        testRandomForest();
//        testJ48Ensemble();
        testJ48TreeAttStats();
        testShapeletTransform();

    }



}
