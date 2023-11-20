package tests_for_feature_extraction;

import java.util.Arrays;

import org.apache.commons.math3.analysis.function.Identity;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.transform.FastFourierTransformer;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.differentiation.DerivativeStructure;
import org.apache.commons.math3.analysis.differentiation.UnivariateDifferentiableFunction;
import org.apache.commons.math3.analysis.function.Sin;
import org.apache.commons.math3.analysis.function.Sqrt;
public class FeatureExtraction {

    public static double[][] normalise(double[][] sequence) {
        /*
         * Normalizes a matrix of accelerometer values.
         */
        RealMatrix matrix = new Array2DRowRealMatrix(sequence);
        double[] norms = calculateL2Norms(matrix);

        // Handle the case where norm is 0 to avoid division by zero
        for (int i = 0; i < norms.length; i++) {
            if (norms[i] == 0) {
                norms[i] = 1;
            }
        }

        // Divide each row by its norm
        for (int i = 0; i < sequence.length; i++) {
            for (int j = 0; j < sequence[0].length; j++) {
                sequence[i][j] /= norms[i];
            }
        }

        return sequence;
    }

    private static double[] calculateL2Norms(RealMatrix matrix) {
        double[] norms = new double[matrix.getRowDimension()];
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            double norm = 0;
            for (int j = 0; j < matrix.getColumnDimension(); j++) {
                norm += Math.pow(matrix.getEntry(i, j), 2);
            }
            norms[i] = Math.sqrt(norm);
        }
        return norms;
    }


    public static double[][] fft(double[][] data) {
        // Extract x, y, and z data
        double[] xData = getColumn(data, 0);
        double[] yData = getColumn(data, 1);
        double[] zData = getColumn(data, 2);

        // Pad the arrays with 3 extra zeros at the end
        double[] paddedXData = padArray(xData, 3);
        double[] paddedYData = padArray(yData, 3);
        double[] paddedZData = padArray(zData, 3);

        // Apply FFT to each axis
        Complex[] xFft = performFft(paddedXData);
        Complex[] yFft = performFft(paddedYData);
        Complex[] zFft = performFft(paddedZData);

        // Remove the last 3 values from the FFT result
        Complex[] truncatedXFft = truncateArray(xFft, 3);
        Complex[] truncatedYFft = truncateArray(yFft, 3);
        Complex[] truncatedZFft = truncateArray(zFft, 3);

        // The result is complex numbers, so you may want to take the magnitude
        double[] xMagnitude = getMagnitude(truncatedXFft);
        double[] yMagnitude = getMagnitude(truncatedYFft);
        double[] zMagnitude = getMagnitude(truncatedZFft);

        int length = xMagnitude.length;
        double[][] representation = new double[length][3];

        for (int i = 0; i < length; i++) {
            representation[i] = new double[]{xMagnitude[i], yMagnitude[i], zMagnitude[i]};
        }

        return representation;
    }

    private static double[] padArray(double[] array, int paddingSize) {
        double[] paddedArray = new double[array.length + paddingSize];
        System.arraycopy(array, 0, paddedArray, 0, array.length);
        return paddedArray;
    }

    private static Complex[] truncateArray(Complex[] array, int truncationSize) {
        Complex[] truncatedArray = new Complex[array.length - truncationSize];
        System.arraycopy(array, 0, truncatedArray, 0, truncatedArray.length);
        return truncatedArray;
    }

    private static Complex[] performFft(double[] data) {
        FastFourierTransformer transformer = new FastFourierTransformer(null);
        return transformer.transform(data, org.apache.commons.math3.transform.TransformType.FORWARD);
    }

    private static double[] getMagnitude(Complex[] fftResult) {
        double[] magnitude = new double[fftResult.length];
        for (int i = 0; i < fftResult.length; i++) {
            magnitude[i] = fftResult[i].abs();
        }
        return magnitude;
    }

    private static double[] getColumn(double[][] matrix, int columnIndex) {
        double[] column = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            column[i] = matrix[i][columnIndex];
        }
        return column;
    }

    public static double[][] differential(double[][] data) {
        // Extract x, y, and z data
        double[] xData = getColumn(data, 0);
        double[] yData = getColumn(data, 1);
        double[] zData = getColumn(data, 2);

        // Compute the differences between consecutive data points
        double[] xDiff = computeDifference(xData);
        double[] yDiff = computeDifference(yData);
        double[] zDiff = computeDifference(zData);

        // Combine the differential values into a representation
        int length = xDiff.length;
        double[][] representation = new double[length + 1][3];

        // Set the first row to make it correct length
        representation[0] = new double[]{0, 0, 0};

        for (int i = 0; i < length; i++) {
            representation[i + 1] = new double[]{xDiff[i], yDiff[i], zDiff[i]};
        }

        return representation;
    }

    private static double[] computeDifference(double[] data) {
        double[] diff = new double[data.length - 1];
        for (int i = 0; i < diff.length; i++) {
            diff[i] = data[i + 1] - data[i];
        }
        return diff;
    }

    public static double[][] derivative(double[][] data) {
        // Extract x, y, and z data
        double[] xData = getColumn(data, 0);
        double[] yData = getColumn(data, 1);
        double[] zData = getColumn(data, 2);

        // Compute the derivative of the data
        double[] xDerivative = computeDerivative(xData);
        double[] yDerivative = computeDerivative(yData);
        double[] zDerivative = computeDerivative(zData);

        // Combine the derivative values into a representation
        int length = xDerivative.length;
        double[][] representation = new double[length][3];

        for (int i = 0; i < length; i++) {
            representation[i] = new double[]{xDerivative[i], yDerivative[i], zDerivative[i]};
        }

        return representation;
    }

    private static double[] computeDerivative(double[] data) {
        UnivariateDifferentiableFunction derivativeFunction = new Identity();
        double[] derivative = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            DerivativeStructure x = new DerivativeStructure(1, 1, 0, data[i]);
            derivative[i] = derivativeFunction.value(x).getPartialDerivative(1);
        }
        return derivative;
    }

    public static double[][] mergeArrays(double[][] arr1, double[][] arr2, double[][] arr3, double[][] arr4) {
        int numRows = arr1.length;
        int numCols = arr1[0].length + arr2[0].length + arr3[0].length + arr4[0].length;

        double[][] result = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            System.arraycopy(arr1[i], 0, result[i], 0, arr1[i].length);
            System.arraycopy(arr2[i], 0, result[i], arr1[i].length, arr2[i].length);
            System.arraycopy(arr3[i], 0, result[i], arr1[i].length + arr2[i].length, arr3[i].length);
            System.arraycopy(arr4[i], 0, result[i], arr1[i].length + arr2[i].length + arr3[i].length, arr4[i].length);
        }

        return result;
    }

    public static double[][] convertArrayListToDoubleArray(List<Float[]> list) {
        int numRows = list.size();
        int numCols = list.get(0).length; // Assuming all inner arrays have the same size

        double[][] result = new double[numRows][numCols];

        for (int i = 0; i < numRows; i++) {
            Float[] innerArray = list.get(i);
            for (int j = 0; j < numCols; j++) {
                result[i][j] = innerArray[j].doubleValue();
            }
        }

        return result;
    }

    public static List<Float[]> convertDoubleArrayToArrayList(double[][] doubleArray) {
        int numRows = doubleArray.length;
        int numCols = doubleArray[0].length; // Assuming all inner arrays have the same size

        List<Float[]> result = new ArrayList<>();

        for (int i = 0; i < numRows; i++) {
            Float[] innerArray = new Float[numCols];
            for (int j = 0; j < numCols; j++) {
                innerArray[j] = (float) doubleArray[i][j];
            }
            result.add(innerArray);
        }

        return result;
    }

    public static List<Float[]> extract_features(List<Float[]> raw_data) {

        double[][] raw_data_matrix = convertArrayListToDoubleArray(raw_data);

        double[][] normalized_matrix = normalise(raw_data_matrix);

        //get transform
        double[][] fourier_transform = fft(normalized_matrix);

        //get differentials
        double[][] differentials = differential(normalized_matrix);

        //get gradients
        double[][] gradients = derivative(normalized_matrix);

        //combine into one matrix
        double[][] combined_data = mergeArrays(normalized_matrix, fourier_transform, differentials, gradients);

        return convertDoubleArrayToArrayList(combined_data);

    }

    }