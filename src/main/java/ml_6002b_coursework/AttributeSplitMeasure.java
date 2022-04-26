package ml_6002b_coursework;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Enumeration;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 */
public abstract class AttributeSplitMeasure {

    public abstract double computeAttributeQuality(Instances data, Attribute att) throws Exception;



    public Instances[] splitDataOnNumeric(Instances data, Attribute att) throws Exception {
        if(att.isNumeric()){
            int noOfSplits = 2;
            Instances[] splitData = new Instances[noOfSplits];
            for (int i = 0; i < noOfSplits; i++) {
                splitData[i] = new Instances(data, data.numInstances());
            }

            int splitPoint = (int)data.meanOrMode(att);
            int subset;

            for (Instance inst: data) {
                if(inst.value(att)<splitPoint){
                    splitData[0].add(inst);
                }else{
                    splitData[1].add(inst);
                }
            }

            for (Instances split : splitData) {
                split.compactify();
            }

            return splitData;

        }else{
            throw new Exception("Unknown data type");
        }


//        int[] count = new int[data.numClasses()];
//        for(Instance ins:data){
//            int c=(int)ins.classValue();
//            count[c]++;
//        }
//        double[] classDistribution = new double[data.numClasses()];
//        for(int i=0;i<data.numClasses();i++){
//            classDistribution[i]=count[i]/(double)data.numInstances();

    }

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
    public Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];
        for (int i = 0; i < att.numValues(); i++) {
            splitData[i] = new Instances(data, data.numInstances());
        }

        for (Instance inst: data) {
            splitData[(int) inst.value(att)].add(inst);
        }

        for (Instances split : splitData) {
            split.compactify();
        }

        return splitData;
    }

}