package com.example.voiceverification.services;

import android.content.Context;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class VadService {
    Interpreter vadModel = null;
    Interpreter mfccModel = null;
    Context context = null;

    private int sr = 16000;
    private float threshold = (float) 0.5;
    private float frameLen = (float) 0.01;
    private int nFrameLen;
    private float frameOverlap = (float) (0.15 - 0.01) / 2;
    private int nFrameOverlap;
    private float[] buffer;
    private String prevChar;

    public void initialDataForCalculation() {
        this.sr = 16000;

        this.nFrameLen = (int) (frameLen * this.sr);
        this.nFrameOverlap = (int) (frameOverlap * this.sr);

        this.buffer = new float[2 * this.nFrameOverlap + this.nFrameLen];
        this.reset();
    }

    public VadService(Context context) throws IOException {
        this.context = context;
        Interpreter.Options options = new Interpreter.Options();
        Interpreter.Options optionsGPU = new Interpreter.Options();
        options.setNumThreads(4);

        CompatibilityList compatList = new CompatibilityList();
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        optionsGPU.addDelegate(gpuDelegate);

        MappedByteBuffer vadBuffer = FileUtil.loadMappedFile(this.context, "vad_marblenet_primary.tflite");
        vadModel = new Interpreter(vadBuffer, options);

        MappedByteBuffer preprocessBuffer = FileUtil.loadMappedFile(this.context, "preprocessing_mfcc_primary.tflite");
        mfccModel = new Interpreter(preprocessBuffer, options);

        this.initialDataForCalculation();
    }

    private float[] inferSignal() {
        TensorBuffer mfccOutputData = TensorBuffer.createFixedSize(new int[]{1, 64, 16}, DataType.FLOAT32);
        TensorBuffer mfccInputData = TensorBuffer.createFixedSize(new int[]{2400}, DataType.FLOAT32);
        mfccInputData.loadArray(this.buffer);
        this.mfccModel.run(mfccInputData.getBuffer(), mfccOutputData.getBuffer());

        TensorBuffer vadOutputData = TensorBuffer.createFixedSize(new int[]{1, 2}, DataType.FLOAT32);
        TensorBuffer vadInputData = TensorBuffer.createFixedSize(new int[]{1, 64, 16}, DataType.FLOAT32);
        vadInputData.loadArray(mfccOutputData.getFloatArray());

        this.vadModel.run(vadInputData.getBuffer(), vadOutputData.getBuffer());
        return vadOutputData.getFloatArray();
    }

    private boolean decode(float[] frame) {
        assert frame.length == this.nFrameLen;

        System.arraycopy(this.buffer, this.nFrameLen, this.buffer, 0, this.buffer.length - this.nFrameLen);
        System.arraycopy(frame, 0, this.buffer, this.buffer.length - this.nFrameLen, this.nFrameLen);

        float[] logits = inferSignal();
        boolean decoded = greedyDecoder(logits);
        return decoded;
    }

    public boolean transcribe(float[] frame) {
        if (frame == null) {
            frame = new float[this.nFrameLen];
        }
        if (frame.length < this.nFrameLen) {
            float[] paddedFrame = new float[this.nFrameLen];
            System.arraycopy(frame, 0, paddedFrame, 0, frame.length);
            frame = paddedFrame;
        }
        return this.decode(frame);
    }

    public void reset() {
        this.buffer = new float[this.buffer.length];
        this.prevChar = "";
    }

    private float[] softMax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float value : logits) {
            max = Math.max(max, value);
        }

        float sum = 0.0f;
        float[] probabilities = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = (float) Math.exp(logits[i] - max);
            sum += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    private boolean greedyDecoder(float[] logits) {
        float probabilitySpeech = this.softMax(logits)[1];
        return probabilitySpeech > this.threshold;
    }
}
