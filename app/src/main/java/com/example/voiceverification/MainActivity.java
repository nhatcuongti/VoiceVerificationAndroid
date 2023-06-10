package com.example.voiceverification;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Activity;
import android.app.KeyguardManager;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Vibrator;
import android.preference.PreferenceManager;
import android.telephony.PhoneStateListener;
import android.telephony.TelephonyManager;
import android.util.Log;
import android.view.KeyEvent;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
//import org.tensorflow.lite.flex.FlexDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

import com.example.voiceverification.utils.InputVadGenerator;
import com.example.voiceverification.utils.LockscreenService;
import com.example.voiceverification.utils.LockscreenUtils;
import com.example.voiceverification.utils.MfccModel;

public class MainActivity extends Activity implements LockscreenUtils.OnLockStatusChangedListener {
    MappedByteBuffer preprocessingModel;
    MappedByteBuffer voiceVerificationModel;
    MappedByteBuffer cosineSimilarityModel;
    MappedByteBuffer vadModel;
    MappedByteBuffer fullModel;

    Interpreter preprocessing = null;
    Interpreter voiceVerification = null;
    Interpreter cosineSimilarity = null;
    Interpreter vadIntepreter = null;
    Interpreter full = null;


    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 2000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);

    AudioRecord record = null;

    // Working variables.
    static short[] recordingBuffer = new short[RECORDING_LENGTH];

    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;

    private static boolean flag = true;
    private TextView txtSpeech;

    private static Vibrator vibe;
    private static MediaPlayer mp;
    private static MediaPlayer mp_y;
    private static MediaPlayer mp_n;

//    private static SharedPreferences prefs;
//    private static boolean isEnroll = true;
    private static boolean isStartServices = false;

    private static float[] audio_1 = null;
    private static float[] audio_2 = null;
    private static float[] audio_3 = null;
//    private static float[] audio_4 = null;
//    private static float[] audio_5 = null;

    private static LockscreenUtils mLockscreenUtils;

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

        setContentView(R.layout.activity_main);

        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);


        mLockscreenUtils = new LockscreenUtils();

        // unlock screen in case of app get killed by system
        if (getIntent() != null && getIntent().hasExtra("kill")
                && getIntent().getExtras().getInt("kill") == 1) {
//            enableKeyguard();
//            unlockHomeButton();
        } else {

            try {
                Log.d("tfliteSupport", "services");
                // disable keyguard
                disableKeyguard();
                Log.d("tfliteSupport", "services");

                // lock home button
                lockHomeButton();

                Log.d("tfliteSupport", "services");
                // start service for observing intents
                if (!isStartServices) {
                    startService(new Intent(this, LockscreenService.class));
                    isStartServices = true;
                }

                // listen the events get fired during the call
                StateListener phoneStateListener = new StateListener();
                TelephonyManager telephonyManager = (TelephonyManager) getSystemService(TELEPHONY_SERVICE);
                telephonyManager.listen(phoneStateListener,
                        PhoneStateListener.LISTEN_CALL_STATE);

            } catch (Exception e) {
                Log.d("tfliteSupport", ""+e);
            }

        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O_MR1)
        {
            setShowWhenLocked(true);
            setTurnScreenOn(false);

            KeyguardManager keyguardManager = (KeyguardManager) getSystemService(Context.KEYGUARD_SERVICE);
            if(keyguardManager!=null)
                keyguardManager.requestDismissKeyguard(this, null);
        }else {
            getWindow().setType(
                    WindowManager.LayoutParams.TYPE_KEYGUARD_DIALOG);
            getWindow().addFlags(
                    WindowManager.LayoutParams.FLAG_FULLSCREEN
                            | WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED
//                        | WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON
                            | WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD
            );
        }
        getWindow().getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_HIDE_NAVIGATION);




//        prefs = PreferenceManager.getDefaultSharedPreferences(this);
        boolean isEnroll = prefs.getBoolean("isEnroll", false);

        if (!isEnroll){
            int LAUNCH_ENROLL_ACTIVITY = 1;
            Intent i = new Intent(this, EnrollActivity.class);
            startActivityForResult(i, LAUNCH_ENROLL_ACTIVITY);
        }else{
            audio_1 = fromString(prefs.getString("audio_1", null));
            audio_2 = fromString(prefs.getString("audio_2", null));
            audio_3 = fromString(prefs.getString("audio_3", null));
//            audio_4 = fromString(prefs.getString("audio_4", null));
//            audio_5 = fromString(prefs.getString("audio_5", null));
        }



        Interpreter.Options options = new Interpreter.Options();

        NnApiDelegate nnApiDelegate = new NnApiDelegate();

        options.setNumThreads(4);





        try{

            cosineSimilarityModel = FileUtil.loadMappedFile(this, "cosine.tflite");
            cosineSimilarity = new Interpreter(cosineSimilarityModel, options);;
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
//        getSupportActionBar().hide();
        mp = MediaPlayer.create(this, R.raw.alert);
        mp_y = MediaPlayer.create(this, R.raw.success);
        mp_n = MediaPlayer.create(this, R.raw.fail);
        vibe = (Vibrator) this.getSystemService(Context.VIBRATOR_SERVICE);


        Log.d("tfliteSupport", "start");

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
//        if (preprocessing != null)
//            preprocessing.close();
//        if (voiceVerification != null)
//            voiceVerification.close();
//        if (cosineSimilarity != null)
//            cosineSimilarity.close();
//        if (record != null)
//            record.release();
//        record = null;
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
        Log.d("tfliteSupport", Arrays.toString(inputBuffer));

//        long startTime = System.nanoTime();
//        TensorBuffer outputBuffer1 =
//                TensorBuffer.createFixedSize(new int[]{10, 64, 402, 1}, DataType.FLOAT32);

        TensorBuffer outputBuffer2 =
                TensorBuffer.createFixedSize(new int[]{10, 512}, DataType.FLOAT32);

//        if ((null != preprocessing) && (null != voiceVerification)){
//            preprocessing.run(inputBuffer, outputBuffer1.getBuffer());
//            voiceVerification.run(outputBuffer1.getBuffer(), outputBuffer2.getBuffer());
//        }
//        long estimatedTime = System.nanoTime() - startTime;
//        Log.d("tfliteSupport", ""+estimatedTime);
        if (null != full){
            full.run(inputBuffer, outputBuffer2.getBuffer());
        }
//        Log.d("tfliteSupport", Arrays.toString(outputBuffer2.getFloatArray()));
        return outputBuffer2.getFloatArray();
    }

    private float verify(float[] v1, float[] v2){
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

    private static float[] fromString(String string) {
        String[] strings = string.replace("[", "").replace("]", "").split(", ");
        float result[] = new float[strings.length];
        for (int i = 0; i < result.length; ++i) {
            result[i] = Float.parseFloat(strings[i]);
        }
        return result;
    }

    public void onclickSpeak(View v){
        mp.start();
        vibe.vibrate(160);
        Toast.makeText(getBaseContext(), "Done", Toast.LENGTH_LONG).show();
        float[] vec = voiceToVec();
        float result = verify(vec, audio_1) + verify(vec, audio_2) + verify(vec, audio_3);
        Log.d("tfliteSupport", ""+ verify(vec, audio_1) + " " + verify(vec, audio_2) + " " + verify(vec, audio_3) + " " + result);

        if (result >= 2.0) {
            txtSpeech.setText("WELCOME");
            mp_y.start();
            unlockHomeButton();
        }else{
            txtSpeech.setText("TRY AGAIN");
            mp_n.start();
        }
    }

    public void onclickReset(View v){
        vibe.vibrate(160);
        int LAUNCH_ENROLL_ACTIVITY = 1;
        Intent i = new Intent(this, EnrollActivity.class);
        startActivityForResult(i, LAUNCH_ENROLL_ACTIVITY);
    }


    // Handle events of calls and unlock screen if necessary
    private class StateListener extends PhoneStateListener {
        @Override
        public void onCallStateChanged(int state, String incomingNumber) {

            super.onCallStateChanged(state, incomingNumber);
            switch (state) {
                case TelephonyManager.CALL_STATE_RINGING:
                    unlockHomeButton();
                    break;
                case TelephonyManager.CALL_STATE_OFFHOOK:
                    break;
                case TelephonyManager.CALL_STATE_IDLE:
                    break;
            }
        }
    };

    // Don't finish Activity on Back press
    @Override
    public void onBackPressed() {
        return;
    }

    // Handle button clicks
    @Override
    public boolean onKeyDown(int keyCode, android.view.KeyEvent event) {

        if ((keyCode == KeyEvent.KEYCODE_VOLUME_DOWN)
                || (keyCode == KeyEvent.KEYCODE_POWER)
                || (keyCode == KeyEvent.KEYCODE_VOLUME_UP)
                || (keyCode == KeyEvent.KEYCODE_CAMERA)) {
            return true;
        }
        if ((keyCode == KeyEvent.KEYCODE_HOME)) {

            return true;
        }

        return false;

    }

    // handle the key press events here itself
    public boolean dispatchKeyEvent(KeyEvent event) {
        if (event.getKeyCode() == KeyEvent.KEYCODE_VOLUME_UP
                || (event.getKeyCode() == KeyEvent.KEYCODE_VOLUME_DOWN)
                || (event.getKeyCode() == KeyEvent.KEYCODE_POWER)) {
            return false;
        }
        if ((event.getKeyCode() == KeyEvent.KEYCODE_HOME)) {

            return true;
        }
        return false;
    }

    // Lock home button
    public void lockHomeButton() {
        mLockscreenUtils.lock(MainActivity.this);
    }

    // Unlock home button and wait for its callback
    public void unlockHomeButton() {
        mLockscreenUtils.unlock();
    }

    // Simply unlock device when home button is successfully unlocked
    @Override
    public void onLockStatusChanged(boolean isLocked) {
        if (!isLocked) {
            unlockDevice();
        }
    }



    @Override
    protected void onStop() {
        super.onStop();
        unlockHomeButton();
    }

    @SuppressWarnings("deprecation")
    private void disableKeyguard() {
        KeyguardManager mKM = (KeyguardManager) getSystemService(KEYGUARD_SERVICE);
        KeyguardManager.KeyguardLock mKL = mKM.newKeyguardLock("IN");
        mKL.disableKeyguard();
    }

    @SuppressWarnings("deprecation")
    private void enableKeyguard() {
        KeyguardManager mKM = (KeyguardManager) getSystemService(KEYGUARD_SERVICE);
        KeyguardManager.KeyguardLock mKL = mKM.newKeyguardLock("IN");
        mKL.reenableKeyguard();
    }

    //Simply unlock device by finishing the activity
    private void unlockDevice()
    {
        finish();
    }


    @Override
    public void onWindowFocusChanged(boolean hasFocus) {
        super.onWindowFocusChanged(hasFocus);
        if (hasFocus) {
            getWindow().getDecorView().setSystemUiVisibility(
                    View.SYSTEM_UI_FLAG_LAYOUT_STABLE
                            | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                            | View.SYSTEM_UI_FLAG_FULLSCREEN
                            | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY);
        }
    }
}