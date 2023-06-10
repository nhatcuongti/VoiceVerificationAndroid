package com.example.voiceverification.utils;

import org.apache.commons.math3.util.MathArrays;

public class InputVadGenerator {
    public static final int BATCH_SIZE = 2048;
    public static final int FRAMES = 30;
    public static final int NUM_FEATURES = 24;

    public static double[][][] generate(double[][] mfcc, double[][] delta) {
        double[][][] inputVad = new double[BATCH_SIZE][FRAMES][NUM_FEATURES];
        int numFrame = mfcc.length;
        int indexBatch = 0;
        while (indexBatch < numFrame - FRAMES) {
            // mfcc : indexBatch -> indexBatch + 30
            int indexOfBatch = 0;
            for (int frameIndex = indexBatch; frameIndex < indexBatch + FRAMES; frameIndex++)
                inputVad[indexBatch][indexOfBatch++] = MathArrays.concatenate(mfcc[frameIndex], delta[frameIndex]);
            // indexBatch += 1
            indexBatch++;
        }

        double[][] zeroPadding = new double[FRAMES][NUM_FEATURES];
        for (int i = indexBatch; i < BATCH_SIZE; i++)
            inputVad[i] = zeroPadding;

        return inputVad;
    }
}
