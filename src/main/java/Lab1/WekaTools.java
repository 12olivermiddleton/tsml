package Lab1;

import experiments.data.DatasetLoading;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.Randomize;
import weka.filters.unsupervised.instance.RemovePercentage;

import java.io.FileReader;
import java.util.Random;


public class WekaTools {

    public static Instances LoadData(String file_path){

        Instances dataset = null;
        try{
            FileReader reader = new FileReader(file_path);
            dataset = new Instances(reader);
        }catch(Exception e) {
            System.out.println("Exception caught: " + e);
        }

        return dataset;
    }

    public static double accuracy(Classifier c, Instances test) throws Exception {

        c.buildClassifier(test);

        int count =0;
        for(Instance in:test){
            double pred=c.classifyInstance(in);
            double actual = in.classValue();
            System.out.println(" Actual = "+actual+" Predicted = "+pred);
            if(pred==actual)
                count++;
            double[] p = c.distributionForInstance(in);
            for(double d:p)
                System.out.println(d);

        }
        System.out.println(" Number correct = "+count);
        double accuracy = count/(double)test.numInstances();
        System.out.println(" Accuracy = "+accuracy);
        return accuracy;
    }

    public static Instances[] splitData(Instances all, double proportion) throws Exception {

        Instances[] split=new Instances[2];

        // Randomize data
        Randomize rand = new Randomize();
        rand.setInputFormat(all);
        rand.setRandomSeed(42);
        all = Filter.useFilter(all, rand);

        // Remove percentage * test from the data to leave training data set
        RemovePercentage rp = new RemovePercentage();
        rp.setInputFormat(all);
        rp.setPercentage(proportion);
        split[0] = Filter.useFilter(all, rp);

        // Remove percentage * train from data to leave testing data set
        rp = new RemovePercentage();
        rp.setInputFormat(all);
        rp.setPercentage(proportion);
        rp.setInvertSelection(true);
        split[1] = Filter.useFilter(all, rp);

        return split;
    }

    public static double[] classDistribution(Instances data) throws Exception{
        //TODO implement body for this method
    }

    public static void main(String[] args) throws Exception {
        Instances arsenal = LoadData("C:\\Users\\omidd\\OneDrive\\Documents\\University\\Third Year\\Machine Learning\\tsml\\src\\main\\java\\Lab1\\Arsenal_TEST.arff");

        Instances[] split = splitData(arsenal, 30.0);
        System.out.println(split[0]);
        System.out.println("------------");
        System.out.println(split[1]);
    }

}
