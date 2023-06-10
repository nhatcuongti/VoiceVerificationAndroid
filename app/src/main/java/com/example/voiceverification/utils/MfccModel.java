package com.example.voiceverification.utils;


import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.apache.commons.math3.util.MathArrays;
import org.checkerframework.checker.units.qual.A;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

public class MfccModel {
    private static final int NUM_FILTERS = 26; // Number of mel filters
    private static final int NUM_CEPSTRAL_COEFFS = 13; // Number of cepstral coefficients
    private static final int FRAME_SIZE = 2048; // Frame size in samples
    private static final double PRE_EMPHASIS_ALPHA = 0.97; // Pre-emphasis filter coefficient
    private static final double MEL_LOW_FREQ = 0; // Lowest frequency in mel filter bank
    private static final double MEL_HIGH_FREQ = 8000; // Highest frequency in mel filter bank
    private static final double SAMPLE_RATE = 16000; // Sampling rate in Hz
    private static final double FRAME_STEP = 0.03; // Frame step size in seconds
    private static final double FRAME_LENGTH = 0.12; // Frame length in seconds

    public static double[][] executeMfcc(double[] signal) {
        // Convert signal to frames
        double[] zeroPaddingArrays = new double[3 * 480];
        double[] paddingSignal = MathArrays.concatenate(zeroPaddingArrays, signal);
        int numFrames = paddingSignal.length / 480;
        double[][] frameSignal = new double[numFrames][480];

        int index = 0;
        for (int i = 0; i < paddingSignal.length; i += 480){
            frameSignal[index++] = Arrays.copyOfRange(paddingSignal, i, i + 480);
        }

        // Pre-emphasis
        double[][] emphasizedSignalTmp = new double[frameSignal.length][frameSignal[0].length];
        double[] emphasizedSignal = new double[frameSignal.length * frameSignal[0].length];
        emphasizedSignalTmp[0] = frameSignal[0];
        for (int i = 1; i < frameSignal.length; i++) {
            for (int j = 0; j < frameSignal[0].length; j++) {
                emphasizedSignalTmp[i][j] = frameSignal[i][j] - PRE_EMPHASIS_ALPHA * frameSignal[i - 1][j];
            }
        }

        for (int i = 0; i < emphasizedSignalTmp.length; i++)
            for (int j = 0; j < emphasizedSignalTmp[0].length; j++) {
                emphasizedSignal[i * emphasizedSignalTmp[0].length + j] = emphasizedSignalTmp[i][j];
            }

        // Calculate frame shift and frame size in samples
        int frameStepSize = (int) (FRAME_STEP * SAMPLE_RATE);
        int frameSize = (int) (FRAME_LENGTH * SAMPLE_RATE);

        // Frame blocking
        double[][] frames = framesig(emphasizedSignal, frameSize, frameStepSize, true);
        numFrames = frames.length;

        // Compute power spectrum
        double[][] powerSpectrum = powspec(frames, FRAME_SIZE);

        double[] energy = new double[numFrames];
        for (int i = 0; i < numFrames; i++) {
            energy[i] = 0;
            for (int j = 0; j < powerSpectrum[0].length; j++) {
                energy[i] += powerSpectrum[i][j];
            }
        }
        for (int i = 0; i < numFrames; i++) {
            if (energy[i] == 0) {
                energy[i] = Double.MIN_VALUE;
            }
        }


        // Mel filter bank
        double [][] filterBank = getFilterbanks(NUM_FILTERS, FRAME_SIZE, (int) SAMPLE_RATE, (int) MEL_LOW_FREQ, (int) MEL_HIGH_FREQ);

        // Apply filter bank
        double [][] feat = applyFilterbank(powerSpectrum, filterBank);
        for (int i = 0; i < feat.length; i++)
            for (int j = 0; j < feat[i].length; j++)
                feat[i][j] = Math.log(feat[i][j]);

        // Compute cepstral coefficients using DCT
        double[][] dctCoeffs = new double[numFrames][NUM_CEPSTRAL_COEFFS];

//        FastCosineTransformer dct = new FastCosineTransformer(DctNormalization.ORTHOGONAL_DCT_I);
        for (int i = 0; i < feat.length; i++) {
            double[] row = feat[i];
            row = normalizeDct(row);
            for (int j = 0; j < NUM_CEPSTRAL_COEFFS; j++) {
                dctCoeffs[i][j] = row[j];
            }
        }

        int cepLifter = 22;
        dctCoeffs = lifter(dctCoeffs, cepLifter);

        // Replace first cepstral coefficent
        for (int i = 0; i < dctCoeffs.length; i++)
            dctCoeffs[i][0] = Math.log(energy[i]);


        // Remove the 0th cepstral coefficient and return the rest
        double[][] mfcc = new double[numFrames][12];
        for (int i = 0; i < dctCoeffs.length; i++)
            for (int j = 1; j < dctCoeffs[0].length; j++)
                mfcc[i][j - 1] = dctCoeffs[i][j];

        return mfcc;
    }

    public static double[][] executeDelta(double[][] feat, int N) {
        if (N < 1) {
            throw new IllegalArgumentException("N must be an integer >= 1");
        }
        int NUMFRAMES = feat.length;
        double denominator = 0;
        for (int i = 1; i <= N; i++) {
            denominator += i * i * 2;
        }
        double[][] deltaFeat = new double[NUMFRAMES][feat[0].length];
        double[][] padded = pad(feat, N);
        for (int t = 0; t < NUMFRAMES; t++) {
            for (int i = -N; i <= N; i++) {
                int index = Math.min(Math.max(t + i, 0), NUMFRAMES - 1);
                double weight = i / denominator;
                for (int j = 0; j < feat[0].length; j++) {
                    deltaFeat[t][j] += weight * padded[index + N][j];
                }
            }
        }
        return deltaFeat;
    }

    private static double[][] pad(double[][] feat, int N) {
        int NUMFRAMES = feat.length;
        int numFeatures = feat[0].length;
        double[][] padded = new double[NUMFRAMES + 2 * N][numFeatures];
        for (int t = 0; t < NUMFRAMES; t++) {
            System.arraycopy(feat[t], 0, padded[t + N], 0, numFeatures);
        }
        for (int i = 0; i < N; i++) {
            System.arraycopy(feat[0], 0, padded[i], 0, numFeatures);
            System.arraycopy(feat[NUMFRAMES - 1], 0, padded[NUMFRAMES + N + i], 0, numFeatures);
        }
        return padded;
    }
    public static double[][] framesig(double[] sig, int frame_len, int frame_step, boolean stride_trick) {
        int slen = sig.length;
        frame_len = Math.round(frame_len);
        frame_step = Math.round(frame_step);
        int numframes = (slen <= frame_len) ? 1 : 1 + (int) Math.ceil((1.0 * slen - frame_len) / frame_step);

        int padlen = (numframes - 1) * frame_step + frame_len;
        double[] zeros = new double[padlen - slen];
        double[] padsignal = MathArrays.concatenate(sig, zeros);

        double[] win = new double[frame_len];
        for (int i = 0; i < frame_len; i++) {
            win[i] = 1.0;
        }

        double[][] frames;
        if (stride_trick) {
            frames = rollingWindow(padsignal, frame_len, frame_step);
        } else {
            int numframes_nostridetricks = (slen - frame_len) / frame_step + 1;
            frames = new double[numframes_nostridetricks][frame_len];
            for (int i = 0; i < numframes_nostridetricks; i++) {
                for (int j = 0; j < frame_len; j++) {
                    frames[i][j] = padsignal[i * frame_step + j] * win[j];
                }
            }
        }

        return frames;
    }

    public static double[] normalizeDct(double[] x) {
        int N = x.length;
        double[] y = new double[N];
        double c = Math.sqrt(2.0/N);
        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                sum += x[j] * Math.cos(Math.PI*i*(2*j+1)/(2.0*N));
            }
            if (i == 0) {
                y[i] = c*sum/Math.sqrt(2.0);
            } else {
                y[i] = c*sum;
            }
        }
        return y;
    }

    public static boolean isAllZeroArray(double[] a) {
        for (int i = 0; i < a.length; i++)
            if (a[i] != 0) return false;

        return true;
    }

    public static double[][] rollingWindow(double[] a, int window, int step) {
        int shape[] = new int[] { a.length - window + 1, window };
//        double[] flat = new double[a.length * window];
        ArrayList<double[]> resultTmp = new ArrayList<>();
        for (int i = 0; i < shape[0]; i += step) {
            double[] flat = new double[window];
            for (int j = 0; j < window; j++)
                flat[j] = a[i + j];

            if (isAllZeroArray(flat))
                break;

            resultTmp.add(flat);
        }
        double[][] result = new double[resultTmp.size()][shape[1]];
        int realLength = 0;
        for (int i = 0; i < result.length; i++) {
            result[i] = resultTmp.get(i);
//            result[i] = Arrays.copyOfRange(flat, i * window, (i + 1) * window);
//            if (isAllZeroArray(result[i]))
//                break;
//
//            realLength++;
        }

//        double[][] realResult = new double[realLength][window];
//        for (int i = 0; i < realLength; i++)
//            realResult[i] = result[i];
        return result;
    }

    public static double[][] powspec(double[][] frames, int NFFT) {
        int numFrames = frames.length;
        int numBins = NFFT / 2 + 1;
        double[][] powerSpec = new double[numFrames][numBins];

        double normFactor = 1.0 / NFFT;
        double[][] magSpec = magspec(frames, NFFT);

        for (int i = 0; i < numFrames; i++) {
            for (int j = 0; j < numBins; j++) {
                powerSpec[i][j] = normFactor * magSpec[i][j] * magSpec[i][j];
            }
        }

        return powerSpec;
    }

    public static double[][] magspec(double[][] frames, int NFFT) {
        int numFrames = frames.length;
        int frameSize = frames[0].length;

        if (frameSize > NFFT) {
            System.out.println("Warning: Frame length is greater than FFT size, frames will be truncated.");
        }

        // Zero-padding the frames
        double[][] paddedFrames = new double[numFrames][NFFT];
        for (int i = 0; i < numFrames; i++) {
            for (int j = 0; j < frameSize; j++) {
                paddedFrames[i][j] = frames[i][j];
            }
        }

        int numCores = Runtime.getRuntime().availableProcessors();
        int framesPerCore = (int) Math.ceil((double) numFrames / numCores);

        // Compute the magnitude spectrum
        long startTime = System.currentTimeMillis();
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);
        double[][] magnitudeSpectrum = new double[numFrames][NFFT / 2 + 1];

        ExecutorService executorService = Executors.newFixedThreadPool(numCores);
        List<Future<?>> futures = new ArrayList<>();

        for (int core = 0; core < numCores; core++) {
            int startFrame = core * framesPerCore;
            int endFrame = Math.min(startFrame + framesPerCore, numFrames);
            futures.add(executorService.submit(() -> {
                for (int i = startFrame; i < endFrame; i++) {
                    Complex[] fftResult = fft.transform(paddedFrames[i], TransformType.FORWARD);
                    for (int j = 0; j < NFFT / 2 + 1; j++) {
                        double real = fftResult[j].getReal();
                        double imag = fftResult[j].getImaginary();
                        magnitudeSpectrum[i][j] = Math.sqrt(real * real + imag * imag);
                    }
                }
            }));
        }

        executorService.shutdown();
        try {
            executorService.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        // Interrupt any remaining running tasks
        for (Future<?> future : futures) {
            if (!future.isDone()) {
                future.cancel(true);
            }
        }
//        Complex[] fftResult = null;
//        for (int i = 0; i < numFrames; i++) {
//            fftResult = fft.transform(paddedFrames[i], TransformType.FORWARD);
//            for (int j = 0; j < NFFT / 2 + 1; j++) {
//                double real = fftResult[j].getReal();
//                double imag = fftResult[j].getImaginary();
//                magnitudeSpectrum[i][j] = Math.sqrt(real * real + imag * imag);
//            }
//        }
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;

        return magnitudeSpectrum;
    }

    public static double[][] getFilterbanks(int nfilt, int nfft, int samplerate, int lowfreq, Integer highfreq) {
        if (highfreq == null) {
            highfreq = samplerate / 2;
        }
        double lowmel = hz2mel(lowfreq);
        double highmel = hz2mel(highfreq);
        double[] melpoints = new double[nfilt + 2];
        for (int i = 0; i < nfilt + 2; i++) {
            melpoints[i] = lowmel + ((highmel - lowmel) / (nfilt + 1)) * i;
        }
        int[] bin = new int[nfilt + 2];
        for (int i = 0; i < nfilt + 2; i++) {
            bin[i] = (int) Math.floor((nfft + 1) * mel2hz(melpoints[i]) / samplerate);
        }
        double[][] fbank = new double[nfilt][nfft / 2 + 1];
        for (int j = 0; j < nfilt; j++) {
            for (int i = bin[j]; i < bin[j + 1]; i++) {
                fbank[j][i] = (i - bin[j]) / (double) (bin[j + 1] - bin[j]);
            }
            for (int i = bin[j + 1]; i < bin[j + 2]; i++) {
                fbank[j][i] = (bin[j + 2] - i) / (double) (bin[j + 2] - bin[j + 1]);
            }
        }
        return fbank;
    }

    public static double[][] applyFilterbank(double[][] pspec, double[][] fb) {
        double[][] feat = new double[pspec.length][fb.length];

        for (int i = 0; i < pspec.length; i++) {
            for (int j = 0; j < fb.length; j++) {
                feat[i][j] = 0;
                for (int k = 0; k < pspec[i].length; k++) {
                    feat[i][j] += pspec[i][k] * fb[j][k];
                }
            }
        }

        // Apply floor to avoid taking log of zero
        for (int i = 0; i < feat.length; i++) {
            for (int j = 0; j < feat[i].length; j++) {
                feat[i][j] = (feat[i][j] == 0) ? Double.MIN_VALUE : feat[i][j];
            }
        }

        return feat;
    }

    // Lifter
    public static double[][] lifter(double[][] cepstra, int L) {
        int nframes = cepstra.length;
        int ncoeff = cepstra[0].length;
        double[][] lifteredCepstra = new double[nframes][ncoeff];
        if (L > 0) {
            double[] n = new double[ncoeff];
            for (int i = 0; i < ncoeff; i++) {
                n[i] = (double) i;
            }
            double[] lift = new double[ncoeff];
            for (int i = 0; i < ncoeff; i++) {
                lift[i] = 1 + (L/2.)*Math.sin(Math.PI*n[i]/L);
            }
            for (int i = 0; i < nframes; i++) {
                for (int j = 0; j < ncoeff; j++) {
                    lifteredCepstra[i][j] = lift[j] * cepstra[i][j];
                }
            }
        } else {
            // values of L <= 0, do nothing
            lifteredCepstra = Arrays.copyOf(cepstra, nframes);
        }
        return lifteredCepstra;
    }

    // Convert frequency to mel scale
    public static double hz2mel(double hz) {
        return 2595.0 * Math.log10(1.0 + hz / 700.0);
    }

    public static double mel2hz(double mel) {
        return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
    }

}
