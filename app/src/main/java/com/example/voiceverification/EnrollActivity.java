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
    static MappedByteBuffer fullModel;

    private static Interpreter preprocessing = null;
    private static Interpreter voiceVerification = null;
    Interpreter vadIntepreter = null;
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

    public float[] convert3Dto1D(double[][][] arr3D){
        float[] arr1D = new float[arr3D.length * arr3D[0].length * arr3D[0][0].length];
        int index = 0;
        for (int i = 0; i < arr3D.length; i++)
            for (int j = 0; j < arr3D[0].length; j++)
                for (int k = 0; k < arr3D[0][0].length; k++)
                    arr1D[index++] = (float) arr3D[i][j][k];
        return arr1D;
    }

    ArrayList<double[]> dataList = new ArrayList<>();


    public void onClickTestVad(View v) {
        try {
            int index = 0;
            long s = 0;
            double[][] data = null;
            double[][] delta = null;
            double[][][] input = null;
            double[] signal = null;

            int i = 0;
            while (i < dataList.size()) {
                signal = dataList.get(i);
                data = null;
                delta = null;
                input = null;
                long startTime = System.currentTimeMillis();
                data = MfccModel.executeMfcc(signal);
                delta = MfccModel.executeDelta(data, 2);
                input = InputVadGenerator.generate(data, delta);
                TensorBuffer vadOutputData = TensorBuffer.createFixedSize(new int[]{2048, 2}, DataType.FLOAT32);
                TensorBuffer vadInputData = TensorBuffer.createFixedSize(new int[]{2048, 30, 24}, DataType.FLOAT32);
                vadInputData.loadArray(this.convert3Dto1D(input));
                this.vadIntepreter.run(vadInputData.getBuffer(), vadOutputData.getBuffer());
                long endTime = System.currentTimeMillis();
                long duration = endTime - startTime;
                Log.d("hao_performance", "each duration: " + duration);

                s += duration;
                index++;
                i++;
                Thread.sleep(300);
            }
            Log.d("hao_performance", "s: " + s);
            Log.d("hao_performance", "index: " + index);


        } catch (Exception e) {
            e.printStackTrace();
            Log.d("hao_performance", "index: " + e.getMessage());

        }
//        try {
//            int index = 0;
//            long s = 0;
//            double[][] data = null;
//            double[][] delta = null;
//            double[][][] input = null;
//            double[] signal = null;
//
//            int i = 0;
//            while (i < 100) {
//                data = null;
//                delta = null;
//                input = null;
//                Random random = new Random(123);
//
//                float[] randomData = new float[64 * 64]; // Total number of elements in the shape [1, 64, 64] is 64 * 64
//                for (int indexRandom = 0; indexRandom < randomData.length; indexRandom++) {
//                    randomData[indexRandom] = random.nextFloat(); // Generates a random float value between 0.0 and 1.0
//                }
//
//                long startTime = System.currentTimeMillis();
//                TensorBuffer vadOutputData = TensorBuffer.createFixedSize(new int[]{1, 2}, DataType.FLOAT32);
//                TensorBuffer vadInputData = TensorBuffer.createFixedSize(new int[]{1, 64, 64}, DataType.FLOAT32);
//                vadInputData.loadArray(randomData);
//                this.vadIntepreter.run(vadInputData.getBuffer(), vadOutputData.getBuffer());
//                long endTime = System.currentTimeMillis();
//                long duration = endTime - startTime;
//                Log.d("hao_performance", "each duration: " + duration);
//
//                s += duration;
//                index++;
//                i++;
//                Thread.sleep(300);
//            }
//            Log.d("hao_performance", "s: " + s);
//            Log.d("hao_performance", "index: " + index);
//
//
//        } catch (Exception e) {
//            e.printStackTrace();
//            Log.d("hao_performance", "index: " + e.getMessage());
//
//        }
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
//            preprocessingModel = FileUtil.loadMappedFile(this, "preprocessing.tflite");
//            preprocessing = new Interpreter(preprocessingModel, options);
//
//            voiceVerificationModel = FileUtil.loadMappedFile(this, "model.tflite");
//            voiceVerification = new Interpreter(voiceVerificationModel, options);
            fullModel = FileUtil.loadMappedFile(this, "v2-512.tflite");
            full = new Interpreter(fullModel, options);

            vadModel = FileUtil.loadMappedFile(this, "model.tflite");
            vadIntepreter = new Interpreter(vadModel, options);
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

        BufferedReader reader = null;
        InputStreamReader isFramesFile = null;
        try {
            int epoch = 0;
            isFramesFile = new InputStreamReader(getAssets()
                    .open("input_frame.txt"));
            reader = new BufferedReader(isFramesFile);
            String[] parts = null;
            String line = null;
            StringBuilder sb = new StringBuilder();
            double[] signal = null;
            while ((line = reader.readLine()) != null) {
                sb.append(line);
                // Split the line into an array of strings
                parts = sb.toString().split(" ");
                // Convert the strings to integers and print them
                signal = new double[parts.length];
                for (int i = 0; i < parts.length; i++) {
                    signal[i] = Double.parseDouble(parts[i]);
                }
                sb.setLength(0);
                dataList.add(signal);
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.d("hao_performance", "index: " + e.getMessage());

        } finally {
            try {
                isFramesFile.close();
                reader.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
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