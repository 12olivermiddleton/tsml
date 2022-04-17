package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;

import java.io.FileReader;


public class ChiSquaredAttributeSplitMeasure extends AttributeSplitMeasure{

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
        return 0;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) {
        System.out.println("Not Implemented.");
    }

}
