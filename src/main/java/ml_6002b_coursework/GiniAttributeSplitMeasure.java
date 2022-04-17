package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;

import java.io.FileReader;


public class GiniAttributeSplitMeasure extends AttributeSplitMeasure{

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

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {

        //TODO will make call to AttributeMeasures and use measureGini
        //TODO measureGini takes a contingency table for given Attribute att

        return 0;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) {

        String WhiskeyData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        Instances whiskey = LoadData(WhiskeyData);
        System.out.println(whiskey);
        GiniAttributeSplitMeasure gini = new GiniAttributeSplitMeasure();
        Instances[] whiskey_split = gini.splitData(whiskey, Peaty);
        System.out.println(whiskey_split);



    }


}
