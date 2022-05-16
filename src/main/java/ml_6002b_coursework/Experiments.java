package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import java.io.IOException;
import java.util.Arrays;

public class Experiments {

    public static Instances opt;
    public static Instances whiskey;
    public static Instances china;
    public static Instances chinaTest;
    public static Instances faces_TRAIN;
    public static Instances faces_TEST;


    public static void loadExperimentsData() throws IOException {
        String optData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/optdigits.arff";
        String whiskeyData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        String chinaData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/Chinatown_TRAIN.arff";
        String chinaDataTest = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/Chinatown_TEST.arff";
        String facesUCR_TRAIN = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/FacesUCR/FacesUCR_TRAIN.arff";
        String facesUCR_TEST = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/FacesUCR/FacesUCR_TEST.arff";

        opt = DatasetLoading.loadData(optData);
        whiskey = DatasetLoading.loadData(whiskeyData);
        china = DatasetLoading.loadData(chinaData);
        chinaTest = DatasetLoading.loadData(chinaDataTest);
        faces_TRAIN = DatasetLoading.loadData(facesUCR_TRAIN);
        faces_TEST = DatasetLoading.loadData(facesUCR_TEST);


        opt.setClassIndex(opt.numAttributes()-1);
        whiskey.setClassIndex(whiskey.numAttributes()-1);
        china.setClassIndex(china.numAttributes()-1);
        chinaTest.setClassIndex(china.numAttributes()-1);
        faces_TRAIN.setClassIndex(faces_TRAIN.numAttributes()-1);
        faces_TEST.setClassIndex(faces_TEST.numAttributes()-1);

    }




    public static void runJ48Experiments() throws Exception {


        opt.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances optTrainData = opt.trainCV(30, 29);
        Instances optTestData = opt.testCV(30, 29);
        Classifier optClassifier = new J48();
        optClassifier.buildClassifier(optTrainData);
        Evaluation evalOpt = new Evaluation(optTrainData);
        evalOpt.evaluateModel(optClassifier, optTrainData);
        evalOpt.evaluateModel(optClassifier, optTestData);
        System.out.println("1-NN accuracy for test data set opt: \n"+ evalOpt.pctCorrect()/100);
        System.out.println("opt\n"+evalOpt.toSummaryString());

        whiskey.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances whiskeyTrain = whiskey.trainCV(10, 9);
        Instances whiskeyTest = whiskey.testCV(10, 9);
        Classifier whiskeyClassifier = new J48();
        whiskeyClassifier.buildClassifier(whiskeyTrain);
        Evaluation evalWhiskey = new Evaluation(whiskeyTrain);
        evalWhiskey.evaluateModel(whiskeyClassifier, whiskeyTrain);
        evalWhiskey.evaluateModel(whiskeyClassifier, whiskeyTest);
        System.out.println("1-NN accuracy for test data set whiskey: \n"+ evalWhiskey.pctCorrect()/100);
        System.out.println("whiskey\n"+evalWhiskey.toSummaryString());


        chinaTest.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances chinaTrainData = chinaTest.trainCV(20, 19);
        Instances chinaTestData = chinaTest.testCV(20, 19);
        Classifier chinaClassifier = new J48();
        chinaClassifier.buildClassifier(chinaTrainData);
        Evaluation evalChina = new Evaluation(chinaTrainData);
        evalChina.evaluateModel(chinaClassifier, chinaTrainData);
        evalChina.evaluateModel(chinaClassifier, chinaTestData);
        System.out.println("1-NN accuracy for test data set china: \n"+ evalChina.pctCorrect()/100);
        System.out.println("china\n"+evalChina.toSummaryString());


        faces_TEST.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances faces_TRAINData = faces_TEST.trainCV(30, 29);
        Instances faces_TESTData = faces_TEST.testCV(30, 29);
        Classifier facesClassifier = new J48();
        facesClassifier.buildClassifier(faces_TRAINData);
        Evaluation evalFaces = new Evaluation(faces_TRAINData);
        evalFaces.evaluateModel(facesClassifier, faces_TRAINData);
        evalFaces.evaluateModel(facesClassifier, faces_TESTData);
        System.out.println("1-NN accuracy for test data set faces: \n"+ evalFaces.pctCorrect()/100);
        System.out.println("faces\n"+evalFaces.toSummaryString());

    }


    public static void runID3Experiments() throws Exception {
        opt.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances optTrainData = opt.trainCV(30, 29);
        Instances optTestData = opt.testCV(30, 29);
        Classifier optClassifier = new Id3();
        optClassifier.buildClassifier(optTrainData);
        Evaluation evalOpt = new Evaluation(optTrainData);
        evalOpt.evaluateModel(optClassifier, optTrainData);
        evalOpt.evaluateModel(optClassifier, optTestData);
        System.out.println("1-NN accuracy for test data set opt: \n"+ evalOpt.pctCorrect()/100);
        System.out.println("opt\n"+evalOpt.toSummaryString());

        whiskey.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances whiskeyTrain = whiskey.trainCV(10, 9);
        Instances whiskeyTest = whiskey.testCV(10, 9);
        Classifier whiskeyClassifier = new Id3();
        whiskeyClassifier.buildClassifier(whiskeyTrain);
        Evaluation evalWhiskey = new Evaluation(whiskeyTrain);
        evalWhiskey.evaluateModel(whiskeyClassifier, whiskeyTrain);
        evalWhiskey.evaluateModel(whiskeyClassifier, whiskeyTest);
        System.out.println("1-NN accuracy for test data set whiskey: \n"+ evalWhiskey.pctCorrect()/100);
        System.out.println("whiskey\n"+evalWhiskey.toSummaryString());


//        chinaTest.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
//        Instances chinaTrainData = chinaTest.trainCV(20, 19);
//        Instances chinaTestData = chinaTest.testCV(20, 19);
//        Classifier chinaClassifier = new Id3();
//        chinaClassifier.buildClassifier(chinaTrainData);
//        Evaluation evalChina = new Evaluation(chinaTrainData);
//        evalChina.evaluateModel(chinaClassifier, chinaTrainData);
//        evalChina.evaluateModel(chinaClassifier, chinaTestData);
//        System.out.println("1-NN accuracy for test data set china: \n"+ evalChina.pctCorrect()/100);
//        System.out.println("china\n"+evalChina.toSummaryString());

//
//        faces_TEST.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
//        Instances faces_TRAINData = faces_TEST.trainCV(30, 29);
//        Instances faces_TESTData = faces_TEST.testCV(30, 29);
//        Classifier facesClassifier = new Id3();
//        facesClassifier.buildClassifier(faces_TRAINData);
//        Evaluation evalFaces = new Evaluation(faces_TRAINData);
//        evalFaces.evaluateModel(facesClassifier, faces_TRAINData);
//        evalFaces.evaluateModel(facesClassifier, faces_TESTData);
//        System.out.println("1-NN accuracy for test data set faces: \n"+ evalFaces.pctCorrect()/100);
//        System.out.println("faces\n"+evalFaces.toSummaryString());

    }

    public static void runCWTReeExperiments() throws Exception {
        opt.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances optTrainData = opt.trainCV(30, 29);
        Instances optTestData = opt.testCV(30, 29);
        Classifier optClassifier = new CourseworkTree();
        optClassifier.buildClassifier(optTrainData);
        Evaluation evalOpt = new Evaluation(optTrainData);
        evalOpt.evaluateModel(optClassifier, optTrainData);
        evalOpt.evaluateModel(optClassifier, optTestData);
        System.out.println("1-NN accuracy for test data set opt: \n"+ evalOpt.pctCorrect()/100);
        System.out.println("opt\n"+evalOpt.toSummaryString());

        whiskey.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances whiskeyTrain = whiskey.trainCV(10, 9);
        Instances whiskeyTest = whiskey.testCV(10, 9);
        Classifier whiskeyClassifier = new CourseworkTree();
        whiskeyClassifier.buildClassifier(whiskeyTrain);
        Evaluation evalWhiskey = new Evaluation(whiskeyTrain);
        evalWhiskey.evaluateModel(whiskeyClassifier, whiskeyTrain);
        evalWhiskey.evaluateModel(whiskeyClassifier, whiskeyTest);
        System.out.println("1-NN accuracy for test data set whiskey: \n"+ evalWhiskey.pctCorrect()/100);
        System.out.println("whiskey\n"+evalWhiskey.toSummaryString());


        chinaTest.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances chinaTrainData = chinaTest.trainCV(20, 19);
        Instances chinaTestData = chinaTest.testCV(20, 19);
        Classifier chinaClassifier = new CourseworkTree();
        chinaClassifier.buildClassifier(chinaTrainData);
        Evaluation evalChina = new Evaluation(chinaTrainData);
        evalChina.evaluateModel(chinaClassifier, chinaTrainData);
        evalChina.evaluateModel(chinaClassifier, chinaTestData);
        System.out.println("1-NN accuracy for test data set china: \n"+ evalChina.pctCorrect()/100);
        System.out.println("china\n"+evalChina.toSummaryString());


        faces_TEST.randomize(new java.util.Random());	// randomize order of instances before splitting dataset
        Instances faces_TRAINData = faces_TEST.trainCV(30, 29);
        Instances faces_TESTData = faces_TEST.testCV(30, 29);
        Classifier facesClassifier = new CourseworkTree();
        facesClassifier.buildClassifier(faces_TRAINData);
        Evaluation evalFaces = new Evaluation(faces_TRAINData);
        evalFaces.evaluateModel(facesClassifier, faces_TRAINData);
        evalFaces.evaluateModel(facesClassifier, faces_TESTData);
        System.out.println("1-NN accuracy for test data set faces: \n"+ evalFaces.pctCorrect()/100);
        System.out.println("faces\n"+evalFaces.toSummaryString());

    }


    public static void main(String[] args) throws Exception {
        loadExperimentsData();
//        runJ48Experiments();
        runID3Experiments();
//        runCWTReeExperiments();






        System.out.println();



    }


}
