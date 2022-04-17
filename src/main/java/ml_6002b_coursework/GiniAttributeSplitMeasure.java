package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instances;


public class GiniAttributeSplitMeasure extends AttributeSplitMeasure{

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
        System.out.println("Not Implemented.");
    }
}
