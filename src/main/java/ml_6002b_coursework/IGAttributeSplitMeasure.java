package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure {

    /**
     * Loads the arff file at the target location and sets the last attribute to be the class value,
     * or returns null on any error, such as not finding the file or it being malformed
     *
     * @param fullPath path to the file to try and load
     * @return Instances from file.
     */
    public static Instances loadData(String fullPath) throws IOException {
        return loadDataThrowable(new File(fullPath));
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
//        int num = whiskey.numClasses();
//        System.out.println(num);
        System.out.println(whiskey.numClasses());

//        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
//        Attribute Peaty = whiskey.attribute("Peaty");
//
////        Instances[] whiskey_split = ig.splitData(whiskey, Peaty);
//        double attribute_quality = ig.computeAttributeQuality(whiskey, Peaty);
//        System.out.println("Attribute Quality: " + attribute_quality);

    }

}