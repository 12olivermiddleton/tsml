package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;

import java.io.FileReader;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

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

        String WhiskeyData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        Instances whiskey = LoadData(WhiskeyData);
        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
        Attribute Peaty = whiskey.attribute("Peaty");
        Instances[] whiskey_split = ig.splitData(whiskey, Peaty);

        System.out.println(whiskey_split[1]);
    }

}