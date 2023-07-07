package com.example.voiceverification.services;

import android.content.Context;
import android.media.AudioRecord;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class AudioProcessService {
    public static final int RECORDING_GETTING_LENGTH = 32000;
    VadService vadService;
    Interpreter full = null;
    Interpreter cosineSimilarity = null;

    // Constants
    private static final int BUFFER_DURATION_MS = 10;
    private static final int TARGET_DURATION_MS = 150;
    private static final int SAMPLE_RATE = 16000;
    private static final int BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION_MS / 1000;
    private static final int SAMPLE_DURATION_MS = 2000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);


    public AudioProcessService(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        Interpreter.Options optionsGPU = new Interpreter.Options();
        options.setNumThreads(4);

        CompatibilityList compatList = new CompatibilityList();
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        optionsGPU.addDelegate(gpuDelegate);
        this.vadService = new VadService(context);

        ByteBuffer fullModel = FileUtil.loadMappedFile(context, "v2-512.tflite");
        full = new Interpreter(fullModel, options);

        ByteBuffer cosineSimilarityModel = FileUtil.loadMappedFile(context, "cosine.tflite");
        cosineSimilarity = new Interpreter(cosineSimilarityModel, options);;
    }


    private short[] processAudioDataWithVad(AudioRecord record) {

        int speechDataIndex = 0;
        short[] speechData = new short[RECORDING_GETTING_LENGTH];
        short[] recordingBuffer = new short[BUFFER_SIZE];
        while (true) {
            int bytesRead = record.read(recordingBuffer, 0, BUFFER_SIZE);
            float[] recordingBufferWithFloat = new float[recordingBuffer.length];
            for (int i = 0; i < recordingBuffer.length; i++)
                recordingBufferWithFloat[i] = (float) recordingBuffer[i];
            boolean isSpeech = vadService.transcribe(recordingBufferWithFloat);

            if (isSpeech) {
                int bytesToCopy = Math.min(bytesRead, RECORDING_LENGTH - speechDataIndex);
                System.arraycopy(recordingBuffer, 0, speechData, speechDataIndex, bytesToCopy);
                speechDataIndex += bytesToCopy;
                if (speechDataIndex >= RECORDING_LENGTH) {
                    break;
                }
            }
        }
        return speechData;

    }

    public float[] voiceToVec(AudioRecord record) {
        float[] inputBuffer = new float[RECORDING_LENGTH];
        record.startRecording();
        Log.e("tfliteSupport", "Start recording");
        long startTime = System.nanoTime();
        short[] speechData = this.processAudioDataWithVad(record);
        for (int i = 0; i < speechData.length; i++)
            inputBuffer[i] = (float) speechData[i];

        TensorBuffer outputBuffer2 =
                TensorBuffer.createFixedSize(new int[]{10, 512}, DataType.FLOAT32);
        if (null != full){
            full.run(inputBuffer, outputBuffer2.getBuffer());
        }
        long estimatedTime = System.nanoTime() - startTime;
        Log.d("tfliteSupport", ""+estimatedTime);
        Log.d("tfliteSupport", Arrays.toString(outputBuffer2.getFloatArray()));
        return outputBuffer2.getFloatArray();
    }

    public float verify(float[] v1, float[] v2){
        TensorBuffer outputBuffer1 =
                TensorBuffer.createFixedSize(new int[]{10, 512}, DataType.FLOAT32);
        TensorBuffer outputBuffer2 =
                TensorBuffer.createFixedSize(new int[]{10, 512}, DataType.FLOAT32);

        outputBuffer1.loadArray(v1);
        outputBuffer2.loadArray(v2);
        Log.d("tfliteSupport", Arrays.toString(outputBuffer1.getFloatArray()));
        Log.d("tfliteSupport", Arrays.toString(outputBuffer2.getFloatArray()));
        Object[] inputArray = {outputBuffer1.getBuffer(), outputBuffer2.getBuffer()};
        Map<Integer, Object> outputMap = new HashMap<Integer, Object>();
        outputMap.put(0, new float[1]);
        cosineSimilarity.runForMultipleInputsOutputs(inputArray, outputMap);

        return ((float []) Objects.requireNonNull(outputMap.get(0)))[0];
    }
}
