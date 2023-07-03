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

import com.example.voiceverification.utils.FrameVad;
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
import java.util.Timer;
import java.util.TimerTask;

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
    short[] recordingBuffer = new short[BUFFER_SIZE];

    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;

    private static SharedPreferences prefs;
    private static Vibrator vibe;
    private static MediaPlayer mp;
    private static MediaPlayer mp_complete;
    private TextView txtSpeech;
    private static int numAudio = 0;
    private FrameVad frameVad = null;

    private Timer captureTimer;

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

            marblenetVadModel = FileUtil.loadMappedFile(this, "vad_marblenet_primary.tflite");
            marblenetVadIntepreter = new Interpreter(marblenetVadModel, options);

            preprocessingMfccModel = FileUtil.loadMappedFile(this, "preprocessing_mfcc_primary.tflite");
            preprocessingMfccIntepreter = new Interpreter(preprocessingMfccModel, options);

            this.frameVad = new FrameVad(preprocessingMfccIntepreter, marblenetVadIntepreter, (float) 0.5, (float) 0.01 , (float) (0.15 - 0.01) / 2, 0);


        } catch (IOException e){
            Log.e("tfliteSupport", "Error reading model", e);
        }

        requestMicrophonePermission();
        record = new AudioRecord(
                MediaRecorder.AudioSource.DEFAULT,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                BUFFER_SIZE * 2);
        captureTimer = new Timer();

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

    // Constants
    private static final int BUFFER_DURATION_MS = 10;
    private static final int TARGET_DURATION_MS = 150;
    private static final int BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION_MS / 1000;
    private static final int TARGET_SIZE = SAMPLE_RATE * TARGET_DURATION_MS / 1000;

    public void startCapture() {
        record.startRecording();
        startCaptureTimer();
    }

    private Integer speechDataIndex = 0;
    private short[] speechData = new short[32000];

    private static float[] softmax(float[] logits) {
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

    private boolean applyVAD() {
        /**
         * Start time MFCC
         */
        TensorBuffer mfccOutputData = TensorBuffer.createFixedSize(new int[]{1, 64, 16}, DataType.FLOAT32);
        TensorBuffer mfccInputData = TensorBuffer.createFixedSize(new int[]{2400}, DataType.FLOAT32);
        float[] recordingBufferFloat = new float[2400];
        for (int i = 0; i < this.recordingBuffer.length; i++)
            recordingBufferFloat[i] = (float) this.recordingBuffer[i];
        mfccInputData.loadArray(recordingBufferFloat);
        this.preprocessingMfccIntepreter.run(mfccInputData.getBuffer(), mfccOutputData.getBuffer());


        TensorBuffer vadOutputData = TensorBuffer.createFixedSize(new int[]{1, 2}, DataType.FLOAT32);
        TensorBuffer vadInputData = TensorBuffer.createFixedSize(new int[]{1, 64, 16}, DataType.FLOAT32);
        vadInputData.loadArray(mfccOutputData.getFloatArray());

        this.marblenetVadIntepreter.run(vadInputData.getBuffer(), vadOutputData.getBuffer());
        float[] probabilitySpeech = this.softmax(vadOutputData.getFloatArray());
        return probabilitySpeech[0] > probabilitySpeech[1];
    }

    private void startCaptureTimer() {
        TimerTask captureTask = new TimerTask() {
            @Override
            public void run() {
                int bytesRead = record.read(recordingBuffer, 0, BUFFER_SIZE);
                boolean isSpeech = applyVAD();

                if (isSpeech) {
                    int bytesToCopy = Math.min(bytesRead, RECORDING_LENGTH - speechDataIndex);
                    System.arraycopy(recordingBuffer, 0, speechData, speechDataIndex, bytesToCopy);
                    speechDataIndex += bytesToCopy;
                    if (speechDataIndex >= RECORDING_LENGTH) {
                        stopCapture();
                    }
                }
            }
        };
        captureTimer.scheduleAtFixedRate(captureTask, 0, 250);
    }

    public void stopCapture() {
        record.stop();
        stopCaptureTimer();
    }

    private void stopCaptureTimer() {
        captureTimer.cancel();
        captureTimer.purge();
    }

    private void processAudioDataWithVad() {
        speechDataIndex = 0;
        speechData = new short[speechData.length];
        while (true) {
            int bytesRead = record.read(recordingBuffer, 0, BUFFER_SIZE);
            float[] recordingBufferWithFloat = new float[recordingBuffer.length];
            for (int i = 0; i < recordingBuffer.length; i++)
                recordingBufferWithFloat[i] = (float) recordingBuffer[i];
            boolean isSpeech = frameVad.transcribe(recordingBufferWithFloat);

            if (isSpeech) {
                int bytesToCopy = Math.min(bytesRead, RECORDING_LENGTH - speechDataIndex);
                System.arraycopy(recordingBuffer, 0, speechData, speechDataIndex, bytesToCopy);
                speechDataIndex += bytesToCopy;
                if (speechDataIndex >= RECORDING_LENGTH) {
                    break;
                }
            }
        }

    }

    private float[] voiceToVec() {
        float[] inputBuffer = new float[RECORDING_LENGTH];
        record.startRecording();
        Log.e("tfliteSupport", "Start recording");
//        this.startCapture();

        long startTime = System.nanoTime();
        this.processAudioDataWithVad();
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