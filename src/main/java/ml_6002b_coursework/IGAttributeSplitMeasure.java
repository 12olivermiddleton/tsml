package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
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

        int[][] peaty = new int[data.attribute("Peaty").numValues()][data.numClasses()];


        System.out.println(peaty);
        for(Instance ins:data){
            peaty[(int)ins.value(0)][(int)ins.classValue()]++;
        }
        for(int[] x:peaty) {
            for (int y : x)
                System.out.print(y + ",");
            System.out.print("\n");
        }



        return 0;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {

        String WhiskeyData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        Instances whiskey = LoadData(WhiskeyData);
        int num = whiskey.numClasses();
        System.out.println(num);
//        System.out.println(whiskey.numClasses());

//        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
//        Attribute Peaty = whiskey.attribute("Peaty");
//
////        Instances[] whiskey_split = ig.splitData(whiskey, Peaty);
//        double attribute_quality = ig.computeAttributeQuality(whiskey, Peaty);
//        System.out.println("Attribute Quality: " + attribute_quality);

    }

}