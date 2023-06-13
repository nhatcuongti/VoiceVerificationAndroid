package com.example.voiceverification;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Vibrator;
import android.preference.PreferenceManager;
import android.util.Log;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.example.voiceverification.utils.InputVadGenerator;
import com.example.voiceverification.utils.MfccModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class EnrollActivity extends AppCompatActivity {
    static MappedByteBuffer preprocessingModel;
    static MappedByteBuffer voiceVerificationModel;
    MappedByteBuffer vadModel;
    MappedByteBuffer marblenetVadModel;
    MappedByteBuffer preprocessingMfccModel;
    static MappedByteBuffer fullModel;

    private static Interpreter preprocessing = null;
    private static Interpreter voiceVerification = null;
    Interpreter vadIntepreter = null;
    Interpreter marblenetVadIntepreter = null;
    Interpreter preprocessingMfccIntepreter = null;
    private static Interpreter full = null;

    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 2000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);

    AudioRecord record = null;

    // Working variables.
    short[] recordingBuffer = new short[RECORDING_LENGTH];

    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;

    private static SharedPreferences prefs;
    private static Vibrator vibe;
    private static MediaPlayer mp;
    private static MediaPlayer mp_complete;
    private TextView txtSpeech;
    private static int numAudio = 0;

    public float[] convert3Dto1D(float[][][] arr3D){
        float[] arr1D = new float[arr3D.length * arr3D[0].length * arr3D[0][0].length];
        int index = 0;
        for (int i = 0; i < arr3D.length; i++)
            for (int j = 0; j < arr3D[0].length; j++)
                for (int k = 0; k < arr3D[0][0].length; k++)
                    arr1D[index++] = (float) arr3D[i][j][k];
        return arr1D;
    }

    ArrayList<double[]> dataList = new ArrayList<>();

    float[] generateRandomArray(Integer n) {
        float[] data = new float[n];
        Random rd = new Random();
        for (int i = 0; i < n; i++)
            data[i] = rd.nextFloat();

        return data;
    }

    public int getIndex(int i, int j) {
        return i * 64 + j;
    }

    public float[][][] reshapeData(float[] rawData) {
        float[][][] reshapeData = new float[201][64][32];
        for (int windowIndex = 0; windowIndex < 201 - 32 + 1; windowIndex++) {
            for (int mfccIndex = 0; mfccIndex < 64; mfccIndex++) {
                for (int frameIndex = 0; frameIndex < 32; frameIndex++) {
                    reshapeData[windowIndex][mfccIndex][frameIndex] =
                            rawData[mfccIndex * 64 + windowIndex + frameIndex];
                }
            }
        }

        return reshapeData;
    }

    public void onClickTestVad(View v) {
        try {
            int index = 0;
            long timeMfcc = 0;
            long timeVad = 0;
            long timeReshape = 0;
            float[] data = null;
            double[][] delta = null;
            float[] input = null;
            double[] signal = null;
            long maxDuration = Long.MIN_VALUE;
            long minDuration = Long.MAX_VALUE;


            int i = 0;
            while (i++ < 100) {
                input = this.generateRandomArray(4960);

                /**
                 * Start time MFCC
                 */
                long startTimeMfcc = System.currentTimeMillis();
                TensorBuffer mfccOutputData = TensorBuffer.createFixedSize(new int[]{1, 64, 32}, DataType.FLOAT32);
                TensorBuffer mfccInputData = TensorBuffer.createFixedSize(new int[]{4960}, DataType.FLOAT32);
                mfccInputData.loadArray(input);
                this.preprocessingMfccIntepreter.run(mfccInputData.getBuffer(), mfccOutputData.getBuffer());
                long endTimeMfcc = System.currentTimeMillis();
                long durationMfcc = endTimeMfcc - startTimeMfcc;
                timeMfcc += durationMfcc;

                /**
                 * Start Time Reshape
                 */
                long startTimeReshape = System.currentTimeMillis();
                TensorBuffer vadOutputData = TensorBuffer.createFixedSize(new int[]{1, 2}, DataType.FLOAT32);
                TensorBuffer vadInputData = TensorBuffer.createFixedSize(new int[]{1, 64, 32}, DataType.FLOAT32);
                vadInputData.loadArray(mfccOutputData.getFloatArray());
                long endTimeReshape = System.currentTimeMillis();
                long durationReshape = endTimeReshape - startTimeReshape;
                timeReshape += durationReshape;

                /**
                 * Start Time VAD
                 */
                long startTimeVad = System.currentTimeMillis();
                this.marblenetVadIntepreter.run(vadInputData.getBuffer(), vadOutputData.getBuffer());
                long endTimeVad = System.currentTimeMillis();
                long durationVad = endTimeVad - startTimeVad;
                timeVad += durationVad;

                if (durationMfcc > maxDuration) maxDuration = durationMfcc;
                if (durationMfcc < minDuration) minDuration = durationMfcc;
            }

            Log.d("hao_performance", "timeMfcc = " + timeMfcc);
            Log.d("hao_performance", "timeReshape = " + timeReshape);
            Log.d("hao_performance", "timeVad = " + timeVad);


        } catch (Exception e) {
            e.printStackTrace();
            Log.d("hao_performance", "index: " + e.getMessage());

        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_enroll);

        numAudio = 0;

        prefs = PreferenceManager.getDefaultSharedPreferences(this);

        Interpreter.Options options = new Interpreter.Options();
        Interpreter.Options optionsGPU = new Interpreter.Options();
        options.setNumThreads(4);

        CompatibilityList compatList = new CompatibilityList();
        GpuDelegate.Options delegateOptions = compatList.getBestOptionsForThisDevice();
        GpuDelegate gpuDelegate = new GpuDelegate(delegateOptions);
        optionsGPU.addDelegate(gpuDelegate);


        try{
            fullModel = FileUtil.loadMappedFile(this, "v2-512.tflite");
            full = new Interpreter(fullModel, options);

            vadModel = FileUtil.loadMappedFile(this, "model.tflite");
            vadIntepreter = new Interpreter(vadModel, options);

            marblenetVadModel = FileUtil.loadMappedFile(this, "vad_marblenet.tflite");
            marblenetVadIntepreter = new Interpreter(marblenetVadModel, options);

            preprocessingMfccModel = FileUtil.loadMappedFile(this, "preprocessing_mfcc.tflite");
            preprocessingMfccIntepreter = new Interpreter(preprocessingMfccModel, options);


        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }

        requestMicrophonePermission();
        record = new AudioRecord(
                MediaRecorder.AudioSource.DEFAULT,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                32000*2);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e("tfliteSupport", "Audio Record can't initialize!");
            return;
        }


        txtSpeech = (TextView) findViewById(R.id.txtSpeech);
        getSupportActionBar().hide();
        mp = MediaPlayer.create(this, R.raw.alert);
        mp_complete = MediaPlayer.create(this, R.raw.ting);
        vibe = (Vibrator) this.getSystemService(Context.VIBRATOR_SERVICE);
    }

    @Override
    protected void onDestroy() {
        if (preprocessing != null)
            preprocessing.close();
        if (voiceVerification != null)
            voiceVerification.close();
        if (full != null)
            full.close();

        if (record != null)
            record.release();

        super.onDestroy();
    }


    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
    }

    private float[] voiceToVec() {
        float[] inputBuffer = new float[RECORDING_LENGTH];
        record.startRecording();
        Log.e("tfliteSupport", "Start recording");
        record.read(new short[3840], 0, 3840);
        record.read(recordingBuffer, 0, recordingBuffer.length);
        for (int i = 0; i < RECORDING_LENGTH; ++i) {
            inputBuffer[i] = recordingBuffer[i] / 32767.0f;
        }
        record.stop();

        long startTime = System.nanoTime();
//        TensorBuffer outputBuffer1 =
//                TensorBuffer.createFixedSize(new int[]{10, 64, 402, 1}, DataType.FLOAT32);

        TensorBuffer outputBuffer2 =
                TensorBuffer.createFixedSize(new int[]{10, 512}, DataType.FLOAT32);

//        if ((null != preprocessing) && (null != voiceVerification)){
//            preprocessing.run(inputBuffer, outputBuffer1.getBuffer());
//            voiceVerification.run(outputBuffer1.getBuffer(), outputBuffer2.getBuffer());
//        }
        if (null != full){
            full.run(inputBuffer, outputBuffer2.getBuffer());
        }
        long estimatedTime = System.nanoTime() - startTime;
        Log.d("tfliteSupport", ""+estimatedTime);
        Log.d("tfliteSupport", Arrays.toString(outputBuffer2.getFloatArray()));
        return outputBuffer2.getFloatArray();
    }


    public void onclickEnroll(View v){
        mp.start();
        vibe.vibrate(160);
        Toast.makeText(getBaseContext(), "Done", Toast.LENGTH_LONG).show();
        float[] vec = voiceToVec();
//        String s_vec = Arrays.toString(vec);
        prefs.edit().putString("audio_"+ ++numAudio, Arrays.toString(vec)).apply();

        txtSpeech.setText("Let's say "+ (3-numAudio) +" more time(s)");
        mp_complete.start();
        if (numAudio >= 3){
            prefs.edit().putBoolean("isEnroll", true).apply();
//            startActivity(new Intent(this, MainActivity.class));
            Intent returnIntent = new Intent();
            setResult(Activity.RESULT_CANCELED, returnIntent);
            finish();
        }
    }


}