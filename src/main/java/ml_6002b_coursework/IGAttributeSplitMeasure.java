package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;


public class IGAttributeSplitMeasure extends AttributeSplitMeasure{

    // When useGain is false, use information gain
    // When useGain is true, use information gain ratio

    public boolean useGain = false;


    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {

        int attribute_index = att.index();

        int[][] att_cont_table = new int[data.attribute(attribute_index).numValues()][data.numClasses()];

        for(Instance ins:data){
            att_cont_table[(int)ins.value(attribute_index)][(int)ins.classValue()]++;
        }
//        for(int[] x:att_cont_table) {
//            for (int y : x)
//                System.out.print(y + ",");
//            System.out.print("\n");
//        }
//        System.out.println(att_cont_table[0][0] + ":" + att_cont_table[0][1] + ":" + att_cont_table[1][0] + ":" + att_cont_table[1][1]);

        AttributeMeasures am = new AttributeMeasures();
        if(useGain==false){
            double ig = am.measureInformationGain(att_cont_table);
            System.out.println("ig: "+ig);
            return ig;
        }else{
            double igr = am.measureInformationGainRatio(att_cont_table);
            System.out.println("igr: "+ igr);
            return igr;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {
        String WhiskeyData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        Instances whiskey = DatasetLoading.loadData(WhiskeyData);

        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
        // set the useGain
        ig.useGain=true;
        Attribute Peaty = whiskey.attribute("Peaty");

//        Instances[] whiskey_split = ig.splitData(whiskey, Peaty);
        double attribute_quality = ig.computeAttributeQuality(whiskey, Peaty);
        System.out.println("Attribute Quality: " + attribute_quality);

    }

}