package ml_6002b_coursework;

import experiments.data.DatasetLoading;
import scala.tools.nsc.backend.jvm.GenASM;
import tsml.transformers.shapelet_tools.quality_measures.InformationGain;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.*;

import java.io.IOException;
import java.util.Arrays;

/**
 * A basic decision tree classifier for use in machine learning coursework (6002B).
 */
public class CourseworkTree extends AbstractClassifier {

    /** Measure to use when selecting an attribute to split the data with. */
    private AttributeSplitMeasure attSplitMeasure = new IGAttributeSplitMeasure();

    /** Maxiumum depth for the tree. */
    private int maxDepth = Integer.MAX_VALUE;

    /** The root node of the tree. */
    private TreeNode root;

    //TODO implement this function to set the split measure, including information gain or ratio
    public void setOptions(String splitMeasure){
        if(splitMeasure=="ig"){
            setAttSplitMeasure(new IGAttributeSplitMeasure());

        }else if(splitMeasure=="igr"){
            IGAttributeSplitMeasure attSplit = new IGAttributeSplitMeasure();
            attSplit.useGain = true;
            setAttSplitMeasure(attSplit);

        }else if(splitMeasure=="chi"){
            setAttSplitMeasure(new ChiSquaredAttributeSplitMeasure());

        }else if(splitMeasure=="gini"){
            setAttSplitMeasure(new GiniAttributeSplitMeasure());

        }
        // In the else case, default measure to IG and output defaulting measure
        else{
            System.out.println("Invalid selection of attribute split measure, defaulting to Information Gain");
            setAttSplitMeasure(new IGAttributeSplitMeasure());

        }

    }


    /**
     * Sets the attribute split measure for the classifier.
     *
     * @param attSplitMeasure the split measure
     */
    public void setAttSplitMeasure(AttributeSplitMeasure attSplitMeasure) {
        this.attSplitMeasure = attSplitMeasure;
    }

    /**
     * Sets the max depth for the classifier.
     *
     * @param maxDepth the max depth
     */
    public void setMaxDepth(int maxDepth){
        this.maxDepth = maxDepth;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return the capabilities of this classifier
     */
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);

        //instances
        result.setMinimumNumberInstances(2);

        return result;
    }

    /**
     * Builds a decision tree classifier.
     *
     * @param data the training data
     */
    @Override
    public void buildClassifier(Instances data) throws Exception {
        if (data.classIndex() != data.numAttributes() - 1) {
            throw new Exception("Class attribute must be the last index.");
        }

        root = new TreeNode();
        root.buildTree(data, 0);
    }

    /**
     * Classifies a given test instance using the decision tree.
     *
     * @param instance the instance to be classified
     * @return the classification
     */
    @Override
    public double classifyInstance(Instance instance) {
        double[] probs = distributionForInstance(instance);

        int maxClass = 0;
        for (int n = 1; n < probs.length; n++) {
            if (probs[n] > probs[maxClass]) {
                maxClass = n;
            }
        }

        return maxClass;
    }

    /**
     * Computes class distribution for instance using the decision tree.
     *
     * @param instance the instance for which distribution is to be computed
     * @return the class distribution for the given instance
     */
    @Override
    public double[] distributionForInstance(Instance instance) {
        return root.distributionForInstance(instance);
    }

    /**
     * Class representing a single node in the tree.
     */
    private class TreeNode {

        /** Attribute used for splitting, if null the node is a leaf. */
        Attribute bestSplit = null;

        /** Best gain from the splitting measure if the node is not a leaf. */
        double bestGain = 0;

        /** Depth of the node in the tree. */
        int depth;

        /** The node's children if it is not a leaf. */
        TreeNode[] children;

        /** The class distribution if the node is a leaf. */
        double[] leafDistribution;

        /**
         * Recursive function for building the tree.
         * Builds a single tree node, finding the best attribute to split on using a splitting measure.
         * Splits the best attribute into multiple child tree node's if they can be made, else creates a leaf node.
         *
         * @param data Instances to build the tree node with
         * @param depth the depth of the node in the tree
         */
        void buildTree(Instances data, int depth) throws Exception {
            this.depth = depth;

            // Loop through each attribute, finding the best one.
            for (int i = 0; i < data.numAttributes() - 1; i++) {
                double gain = attSplitMeasure.computeAttributeQuality(data, data.attribute(i));

                if (gain > bestGain) {
                    bestSplit = data.attribute(i);
                    bestGain = gain;
                }
            }

            // If we found an attribute to split on, create child nodes.
            if (bestSplit != null) {
                Instances[] split = attSplitMeasure.splitData(data, bestSplit);
                children = new TreeNode[split.length];

                // Create a child for each value in the selected attribute, and determine whether it is a leaf or not.
                for (int i = 0; i < children.length; i++){
                    children[i] = new TreeNode();

                    boolean leaf = split[i].numDistinctValues(data.classIndex()) == 1 || depth + 1 == maxDepth;

                    if (split[i].isEmpty()) {
                        children[i].buildLeaf(data, depth + 1);
                    } else if (leaf) {
                        children[i].buildLeaf(split[i], depth + 1);
                    } else {
                        children[i].buildTree(split[i], depth + 1);
                    }
                }
                // Else turn this node into a leaf node.
            } else {
                leafDistribution = classDistribution(data);
            }
        }

        /**
         * Builds a leaf node for the tree, setting the depth and recording the class distribution of the remaining
         * instances.
         *
         * @param data remaining Instances to build the leafs class distribution
         * @param depth the depth of the node in the tree
         */
        void buildLeaf(Instances data, int depth) {
            this.depth = depth;
            leafDistribution = classDistribution(data);
        }

        /**
         * Recursive function traversing node's of the tree until a leaf is found. Returns the leafs class distribution.
         *
         * @return the class distribution of the first leaf node
         */
        double[] distributionForInstance(Instance inst) {
            // If the node is a leaf return the distribution, else select the next node based on the best attributes
            // value.
            if (bestSplit == null) {
                return leafDistribution;
            } else {
                return children[(int) inst.value(bestSplit)].distributionForInstance(inst);
            }
        }

        /**
         * Returns the normalised version of the input array with values summing to 1.
         *
         * @return the class distribution as an array
         */
        double[] classDistribution(Instances data) {
            double[] distribution = new double[data.numClasses()];
            for (Instance inst : data) {
                distribution[(int) inst.classValue()]++;
            }

            double sum = 0;
            for (double d : distribution){
                sum += d;
            }

            if (sum != 0){
                for (int i = 0; i < distribution.length; i++) {
                    distribution[i] = distribution[i] / sum;
                }
            }

            return distribution;
        }

        /**
         * Summarises the tree node into a String.
         *
         * @return the summarised node as a String
         */
        @Override
        public String toString() {
            String str;
            if (bestSplit == null){
                str = "Leaf," + Arrays.toString(leafDistribution) + "," + depth;
            } else {
                str = bestSplit.name() + "," + bestGain + "," + depth;
            }
            return str;
        }
    }

    /**
     * Main method.
     *
     * @param args the options for the classifier main
     */
    public static void main(String[] args) throws Exception {

//        String optData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/test_data/optdigits.arff";
        String optData = "C:/Users/omidd/OneDrive/Documents/University/Third Year/Machine Learning/tsml/tsml/src/main/java/ml_6002b_coursework/Whiskey_Region_Data.arff";
        Instances opt = DatasetLoading.loadData(optData);


        opt.randomize(new java.util.Random());	// randomize instance order before splitting dataset
        Instances trainData = opt.trainCV(9, 8);
        Instances testData = opt.testCV(7, 6);
//        System.out.println(trainData.numInstances());
//        System.out.println(testData.numInstances());


        CourseworkTree cwTree = new CourseworkTree();

        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();

        IGAttributeSplitMeasure igr = new IGAttributeSplitMeasure();
        ig.useGain=true;

        ChiSquaredAttributeSplitMeasure chi = new ChiSquaredAttributeSplitMeasure();

        GiniAttributeSplitMeasure gini = new GiniAttributeSplitMeasure();


        cwTree.setAttSplitMeasure(chi);
        cwTree.getCapabilities();


        cwTree.buildClassifier(opt);
        Evaluation eval_ig = new Evaluation(trainData);
        eval_ig.evaluateModel(cwTree, testData);
        System.out.println(eval_ig.toSummaryString("\nResults\n======\n", false));

//        System.out.println(cwTree.root);
//        System.out.println(cwTree.root.children[0]+"-:-"+cwTree.root.children[1]);
//        System.out.println("-----------------------------");



//        // train classifier
//        Classifier cls = new J48();
//        cls.buildClassifier(train);
//        // evaluate classifier and print some statistics
//        Evaluation eval = new Evaluation(train);
//        eval.evaluateModel(cls, test);
//        System.out.println(eval.toSummaryString("\nResults\n======\n", false));
//
//        CourseworkTree cwTreeOptions = new CourseworkTree();
//        cwTreeOptions.setOptions("igr");
//        cwTreeOptions.buildClassifier(whiskey);
//        System.out.println(cwTreeOptions.root+" -:- "+cwTreeOptions.root.children[0]);



//
//        IGAttributeSplitMeasure igOpt = new IGAttributeSplitMeasure();
//
//        Attribute att_22 = opt.attribute("att3");
//        Instances[] splitData = igOpt.splitDataOnNumeric(opt, att_22);
//        System.out.println(splitData[0].stream().count());
//        System.out.println(splitData[1].stream().count());
    }
}