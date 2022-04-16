package ml_6002b_coursework;

import java.text.DecimalFormat;

/**
 * Empty class for PArt 2.1 of the coursework
 *
 */
public class AttributeMeasures {
    //TODO The edge case occurs when the entropy of a given class is 0

    public static double log2(double x) {
        return (Math.log(x) / Math.log(2));
    }

    public static double entropy(double class_probability){
        return -((class_probability * log2(class_probability)));
    }

    // Function to calculate the entropy of a given class
    public static double calcRootEntropy(int[][] attribute) {

        double class1_count = attribute[0][0] + attribute[1][0];
        double class2_count = attribute[0][1] + attribute[1][1];
        double total_cases = class1_count + class2_count;
        double class1_prob = class1_count / total_cases;
        double class2_prob = class2_count / total_cases;

        // returns root entropy
        return -((class1_prob * log2(class1_prob)) + (class2_prob * log2(class2_prob)));
    }

    public static double splitInfo(int[][] attribute){
        double yes_count = attribute[0][0] + attribute[0][1];
        double no_count = attribute[1][0] + attribute[1][1];
        double total_count = yes_count + no_count;

        double prob_yes = yes_count / total_count;
        double prob_no = no_count / total_count;

        // Returns the split info
        return -((prob_yes*log2(prob_yes)) +(prob_no*log2(prob_no)));
    }


    // Returns the information gain for a given attributes contingency table
    public static double measureInformationGain(int[][] attribute){

        double class0_count = attribute[0][0] + attribute[1][0];
        double class1_count = attribute[0][1] + attribute[1][1];
        double total_cases = class0_count + class1_count;
//        double class0_prob = class0_count / total_cases;
//        double class1_prob = class1_count / total_cases;

        // entropy for yes
        double yes_cases = attribute[0][0] + attribute[0][1];
        double prob_yes_class0 = attribute[0][0] / yes_cases;
        double prob_yes_class1 = attribute[0][1] / yes_cases;
        double yes_proportion = yes_cases / total_cases;

        // entropy for no
        double no_cases = attribute[1][0] + attribute[1][1];
        double prob_no_class0 = attribute[1][0] / no_cases;
        double prob_no_class1 = attribute[1][1] / no_cases;
        double no_proportion = no_cases / total_cases;

        double yes_entropy = entropy(prob_yes_class0) + entropy(prob_yes_class1);
        double no_entropy = entropy(prob_no_class0) + entropy(prob_no_class1);

        // handle edge cases where the entropy for given attribute class may be NaN
        if(Double.isNaN(yes_entropy)){
            yes_entropy = 0.0;
        }
        if(Double.isNaN(no_entropy)){
            no_entropy = 0.0;
        }
        double total_entropy = (yes_proportion * yes_entropy) + (no_proportion * no_entropy);

        return calcRootEntropy(attribute) - total_entropy;
    }

    // Returns the information gain ratio for given attributes contingency table
    public static double measureInformationGainRatio(int[][] attribute){
        // Attribute information gain
        double attribute_IG = measureInformationGain(attribute);
        double attribute_split_info = splitInfo(attribute);

        // Gain ration = information gain / split info
        // Returns the Information gain ratio
        return attribute_IG / attribute_split_info;
    }



    // Returns the gini measure for a given attributes contingency table
    public static double measureGini(int[][] attribute){
        double yes_count = attribute[0][0] + attribute[0][1];
        double no_count = attribute[1][0] + attribute[1][1];
        double total_count = yes_count + no_count;

        double prob_yes = yes_count / total_count;
        double prob_no = no_count / total_count;

        double gini_root_impurity = 1 - (((prob_yes)*(prob_yes))+((prob_no)*(prob_no)));


        return gini_root_impurity;
    }


    // Returns chi squared measure for a given attributes contingency table
    public static double measureChiSquared(int[][] attribute){

        double yes_count = attribute[0][0] + attribute[0][1];
        double no_count = attribute[1][0] + attribute[1][1];

        double islay_count = attribute[0][0] + attribute[1][0];
        double speyside_count = attribute[0][1] + attribute[1][1];

        double total_count = islay_count + speyside_count;

        double prob_islay = islay_count / total_count;
        double prob_speyside = speyside_count / total_count;

        double[][] expected_attribute_values = {
                {yes_count*prob_islay, yes_count*prob_speyside},
                {no_count*prob_islay, no_count*prob_speyside}
        };



        double chi_squared = (
                (Math.pow((attribute[0][0] - expected_attribute_values[0][0]), 2)/expected_attribute_values[0][0])
                + (Math.pow((attribute[0][1] - expected_attribute_values[0][1]), 2)/expected_attribute_values[0][1])
                + (Math.pow((attribute[1][0] - expected_attribute_values[1][0]), 2)/expected_attribute_values[1][0])
                + (Math.pow((attribute[1][1] - expected_attribute_values[1][1]), 2)/expected_attribute_values[1][1])
        );
        return chi_squared;

    }

    /**
     * Main method.
     *
     * @param args the options for the attribute measure main
     */
    public static void main(String[] args) {
        /**
         * Data in form:
         * This example data is the data for peaty
         *
         *    Islay     Speyside
         *      4          0        Yes(1)
         *      1          5        No(0)
         *
          */

        // Contingency tables for peaty, woody and sweet
        int[][] peaty = {{4,0},{1,5}};
        int[][] woody = {{2,3},{3,2}};
        int[][] sweet = {{2,5},{3,0}};

        // Decimal format for calculations to maintain consistency
        DecimalFormat df = new DecimalFormat("##.#####");
        double ig_peaty = measureInformationGain(peaty);
        System.out.println("Information Gain: " + ig_peaty);

        double ig_ratio = measureInformationGainRatio(peaty);
        System.out.println("Information Gain Ratio: " + ig_ratio);

        double gini_measure = measureGini(peaty);
        System.out.println("Gini Measure: " + gini_measure);

        double chi_squared = measureChiSquared(peaty);
        System.out.println("Chi Squared: " + chi_squared);

    }

}
