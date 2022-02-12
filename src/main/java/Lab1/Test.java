package Lab1;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;

public class Test {


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


    public static void main(String[] args) {

        String Arsenal_TRAIN = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/Week 1 intro/Lab1/src/Arsenal_TRAIN.arff";
        String Arsenal_TEST = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/Week 1 intro/Lab1/src/Arsenal_TEST.arff";
        Instances train = LoadData(Arsenal_TRAIN);
        Instances test = LoadData(Arsenal_TEST);
        System.out.println("Number of Instances = " + train.numInstances());
        System.out.println("Number of attributes = " + test.numAttributes());
        train.setClassIndex(test.numAttributes()-1);
        System.out.println("Number of classes = " + train.numClasses());

        int count = 0;
        for(Instance inst:train){
            if(inst.classValue()==2){
                count+=1;
            }
            System.out.println("class value = " + inst.classValue());
            System.out.println(inst.toString());
        }
        System.out.println(count);
        Attribute att = train.attribute(0);
        System.out.println(att);

        Instance fifth = train.instance(4);
        System.out.println(fifth);
        double[] raw = fifth.toDoubleArray();
        for(double d:raw){
            System.out.println("d: "+d);
        }
//        System.out.println(train.toString());
//        train.deleteAttributeAt(2);
//        System.out.println(train.toString());

        // Set the attribute class value
        train.setClassIndex(train.numAttributes()-1);

    }
}