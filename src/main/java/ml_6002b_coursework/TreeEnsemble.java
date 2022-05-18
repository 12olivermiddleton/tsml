package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

public class TreeEnsemble extends AbstractClassifier {

    ArrayList<Classifier> ensemble;
    public int numTrees = 50;
    // If averageDistributions false, use counting, else use average distributions
    public boolean averageDistributions = false;


    public TreeEnsemble(){

    }

    @Override
    public void buildClassifier(Instances data) throws Exception {

        ensemble = new ArrayList<>();
        for(int i=0; i<numTrees; i++){
            String option = randomiseClassifierOptions();

            Classifier c = new CourseworkTree();


            data.randomize(new Random());
            Instances tr = new Instances(data, 0, data.numInstances()/2);
            c.buildClassifier(tr);
            ensemble.add(c);
        }
    }


    // Randomize options for decision tree
    public String randomiseClassifierOptions(){
        String[] options = {"ig", "igr", "chi", "gini"};
        Random random = new Random();
        int index = random.nextInt(options.length);
        String option = options[index];
        return option;

    }



    @Override
    public double classifyInstance(Instance inst) throws Exception {
        int[] counts = new int[inst.numClasses()];
        for(Classifier c: ensemble){
            int vote = (int)c.classifyInstance(inst);
            counts[vote]++;
        }
        int argMax=0;
        for(int i=1; i<counts.length; i++){
            if(counts[i]>counts[argMax])
                argMax=i;
        }
        return argMax;
    }

    @Override
    public double[] distributionForInstance(Instance inst) throws Exception {


        double[] probs = new double[inst.numClasses()];
        for (Classifier c : ensemble) {
            double[] d = c.distributionForInstance(inst);
            for (int i = 0; i < d.length; i++) {
                probs[i] += d[i];
            }
        }


        if (!averageDistributions) {
            double sum = 0;
            for (double prob : probs) {
                sum += prob;
            }
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }

            return probs;
        } else {
            double sum = 0;
            for (double prob : probs) {
                sum += prob;
            }
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }

            return probs;

        }

    }


}
