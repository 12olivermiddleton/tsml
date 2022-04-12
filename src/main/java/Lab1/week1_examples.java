package Lab1;


import experiments.data.DatasetLoading;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;

public class week1_examples {

//    public static Instances LoadData(String file_path){
//
//        Instances dataset = null;
//        try{
//            FileReader reader = new FileReader(file_path);
//            dataset = new Instances(reader);
//        }catch(Exception e) {
//            System.out.println("Exception caught: " + e);
//        }
//
//        return dataset;
//    }


    public static void main(String[] args) throws Exception {
        System.out.println("passed");
        Instances wdbc = DatasetLoading.loadData("C:\\Users\\omidd\\OneDrive\\Documents\\University\\Third Year\\Machine Learning\\tsml\\src\\main\\java\\Lab1\\wdbc.arff"); //TODO change the string filename to complete file path
        System.out.println("passed1");
        MyClassifier cls= new MyClassifier();
        System.out.println("passed classifier");
        cls.buildClassifier(wdbc);
        int count =0;
        for(Instance in:wdbc){
            double pred=cls.classifyInstance(in);
            double actual = in.classValue();
            System.out.println(" Actual = "+actual+" Predicted = "+pred);
            if(pred==actual)
                count++;
            double[] p = cls.distributionForInstance(in);
            for(double d:p)
                System.out.println(d);

        }
        System.out.println(" Number correct = "+count);
        System.out.println(" Accuracy = "+count/(double)wdbc.numInstances());

        System.out.println("-----------");
        double accuracy = WekaTools.accuracy(cls, wdbc);
        System.out.println("accuracy of cls classifier= "+accuracy);

    }




}