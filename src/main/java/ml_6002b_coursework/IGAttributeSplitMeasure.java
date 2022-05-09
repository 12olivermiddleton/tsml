package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.File;
import java.net.URL;


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

        AttributeMeasures am = new AttributeMeasures();
        if(useGain==false){
            double ig = am.measureInformationGain(att_cont_table);
            return ig;
        }else{
            double igr = am.measureInformationGainRatio(att_cont_table);
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

            Attribute Peaty = whiskey.attribute("Peaty");
            Attribute Woody = whiskey.attribute("Woody");
            Attribute Sweet = whiskey.attribute("Sweet");

            double peaty_attribute_quality_ig = ig.computeAttributeQuality(whiskey, Peaty);
            double woody_attribute_quality_ig = ig.computeAttributeQuality(whiskey, Woody);
            double sweet_attribute_quality_ig = ig.computeAttributeQuality(whiskey, Sweet);

            System.out.println("measureInformationGain for Peaty splitting diagnosis = "+peaty_attribute_quality_ig);
            System.out.println("measureInformationGain for Woody splitting diagnosis = "+woody_attribute_quality_ig);
            System.out.println("measureInformationGain for Sweet splitting diagnosis = "+sweet_attribute_quality_ig);

            // set the useGain
            ig.useGain=true;
            double peaty_attribute_quality_igr = ig.computeAttributeQuality(whiskey, Peaty);
            double woody_attribute_quality_igr = ig.computeAttributeQuality(whiskey, Woody);
            double sweet_attribute_quality_igr = ig.computeAttributeQuality(whiskey, Sweet);

            System.out.println("measureInformationGainRatio for Peaty splitting diagnosis = " + peaty_attribute_quality_igr);
            System.out.println("measureInformationGainRatio for Woody splitting diagnosis = "+woody_attribute_quality_igr);
            System.out.println("measureInformationGainRatio for Sweet splitting diagnosis = "+sweet_attribute_quality_igr);
    }

}