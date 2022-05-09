package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;


public class ChiSquaredAttributeSplitMeasure extends AttributeSplitMeasure{


    @Override
    public double computeAttributeQuality(Instances data, Attribute att){


        int attribute_index = att.index();

        int[][] att_cont_table = new int[data.attribute(attribute_index).numValues()][data.numClasses()];

        for(Instance ins:data){
            att_cont_table[(int)ins.value(attribute_index)][(int)ins.classValue()]++;
        }

        AttributeMeasures am = new AttributeMeasures();

        double chi = am.measureChiSquared(att_cont_table);

        return chi;
    }

    /**
     * Main method.
     *
     * @param args the options for the split measure main
     */
    public static void main(String[] args) throws Exception {

        String WhiskeyData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        Instances whiskey = DatasetLoading.loadData(WhiskeyData);

        ChiSquaredAttributeSplitMeasure chi = new ChiSquaredAttributeSplitMeasure();
        Attribute Peaty = whiskey.attribute("Peaty");
        Attribute Woody = whiskey.attribute("Woody");
        Attribute Sweet = whiskey.attribute("Sweet");

        double peaty_attribute_quality_chi = chi.computeAttributeQuality(whiskey, Peaty);
        double woody_attribute_quality_chi = chi.computeAttributeQuality(whiskey, Woody);
        double sweet_attribute_quality_chi = chi.computeAttributeQuality(whiskey, Sweet);

        System.out.println("measureChiSquared for Peaty splitting diagnosis =  " + peaty_attribute_quality_chi);
        System.out.println("measureChiSquared for Woody splitting diagnosis =  " + woody_attribute_quality_chi);
        System.out.println("measureChiSquared for Sweet splitting diagnosis =  " + sweet_attribute_quality_chi);

    }

}
