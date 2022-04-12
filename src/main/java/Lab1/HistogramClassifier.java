package Lab1;

import utilities.InstanceTools;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;



public class HistogramClassifier implements Classifier {
    int numBins=10;
    int[][] counts;
    double[][] distributions;


    int attribute=0;
    double min, max, interval;
    @Override
    public void buildClassifier(Instances data) throws Exception {
        int numClasses=data.numClasses();
        counts=new int[numClasses][numBins];
//Find min and max of attribute
        min=data.attributeStats(attribute).numericStats.min;
        max=data.attributeStats(attribute).numericStats.max;
        double range =max-min;
        interval=range/numBins; //Will this capture all

        for(Instance ins:data){
            double value=ins.value(attribute);
            int classVal=(int)ins.classValue();
            //Find bin from value: slow way of doing it, could use integer division if we are clever
            double x=min;
            int c=0;
            while(x<=value){
                x+=interval;
                c++;
            }
            c--;
            if(x>=max-0.00001)
                c=numBins-1;
            System.out.println(" c = "+c+" value = "+value+" min = "+min+" max = "+max);
            counts[classVal][c]++;
            distributions = new double[data.numClasses()][numBins];
            for(int j=0;j<data.numClasses();j++) {
                double sum = 0;
                for (int i = 0; i < counts[j].length; i++) {
                    sum += counts[j][i];
                }
                for (int i = 0; i < counts[j].length; i++)
                    distributions[j][i] = counts[j][i] / sum;
            }

        }
    }
    private static int maxIndex( int[] x){
        int index=0;
        for(int i=1;i<x.length;i++){
            if(x[i]>x[index])
                index=i;
        }
        return index;
    }
    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double value=instance.value(attribute);
        //Find which bin its in
        double x=min;
        int c=0;
        while(x<value){
            x+=interval;
            c++;
        }
        if(x>=max-0.001)
            c=numBins-1;
        // C is the bin that the new instance attribute 0 is in
        // we call it discretisation: making a continuous variable discrete
        //Find max count
        int[] ct=new int[instance.numClasses()];
        for(int i=0;i<counts.length;i++)
            ct[i]=counts[i][c];
        return maxIndex(ct);

    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] dist = new double[instance.numClasses()];
        double value=instance.value(attribute);
        //Find bin from value
        double x=min;
        int c=0;
        while(x<value){
            x+=interval;
            c++;
        }
        if(x>=max-0.001)
            c=numBins-1;
        //Find max count: Maybe more efficient to transpose counts completely
        double sum=0;
        for(int i=0;i<counts.length;i++) {
            dist[i] = counts[i][c];
            sum += dist[i];
        }
        for(int i=0;i<counts.length;i++)
            dist[i]/=sum;
        return dist;
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }



    @Override
    public String toString(){
        String res= "interval = "+interval+"\n";
        res+= "Histogram::\n";
        for(int[] hist:counts) {
            for (int c : hist)
                res += c + ",";
            res += "\n";
        }
        res+= "Probabilities::\n";
        for(double[] d:distributions) {
            for (double c : d)
                res += c + ",";
            res += "\n";
        }
        return res;
    }

    public static void main(String[] args) throws Exception {
        Instances all;
        String dataPath="C:\\Users\\Tony\\OneDrive - University of East Anglia\\Teaching\\2020-2021\\Machine " +
                "Learning\\Week 2 - Decision Trees\\Week 2 Live " +
                "Class\\tsml-master\\src\\main\\java\\experiments\\data\\uci\\iris\\";
        all=experiments.data.DatasetLoading.loadData(dataPath+"iris");


        //Build on all the iris data
        HistogramClassifier hc=new HistogramClassifier();
        hc.buildClassifier(all);
        System.out.println("MODEL = "+hc.toString());


        int correct=0;
        for(Instance ins:all){
            int pred=(int)hc.classifyInstance(ins);
            int actual = (int)ins.classValue();
//            System.out.println(" Actual = "+actual+" Predicted ="+pred);
            if(pred==actual)
                correct++;
        }
        System.out.println(" num correct = "+correct+" accuracy  = "+ (correct/(double)all.numInstances()));
        Instances[] split = InstanceTools.resampleInstances(all,0,0.5);
        hc.buildClassifier(split[0]);
        System.out.println("MODEL = "+hc.toString());

        correct=0;
        for(Instance ins:split[1]){
            int pred=(int)hc.classifyInstance(ins);
            int actual = (int)ins.classValue();
//            System.out.println(" Actual = "+actual+" Predicted ="+pred);
            if(pred==actual)
                correct++;
        }
        System.out.println(" num correct = "+correct+" accuracy  = "+ (correct/(double)split[1].numInstances()));




    }
}