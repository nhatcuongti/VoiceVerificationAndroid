package com.example.voiceverification;

import android.app.Activity;
import android.content.Context;
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
import android.view.KeyEvent;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;
import java.io.IOException;
import com.example.voiceverification.services.AudioProcessService;

public class VerifyActivity extends Activity {

    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 2000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);

    AudioRecord record = null;


    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;

    private static boolean flag = true;
    private TextView txtSpeech;

    private static Vibrator vibe;
    private static MediaPlayer mp;
    private static MediaPlayer mp_y;
    private static MediaPlayer mp_n;

    //    private static SharedPreferences prefs;
    private static boolean isStartServices = false;

    private static float[] audio_1 = null;
    private static float[] audio_2 = null;
    private static float[] audio_3 = null;

    AudioProcessService audioProcessService = null;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_verify);
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);

        audio_1 = fromString(prefs.getString("audio_1", null));
        audio_2 = fromString(prefs.getString("audio_2", null));
        audio_3 = fromString(prefs.getString("audio_3", null));




        try{
            this.audioProcessService = new AudioProcessService(this);
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
        if (record != null) record.release();
        super.onDestroy();
    }

    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(
                    new String[] {android.Manifest.permission.RECORD_AUDIO}, REQUEST_RECORD_AUDIO);
        }
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
        float[] vec = this.audioProcessService.voiceToVec(record);
        float result = this.audioProcessService.verify(vec, audio_1) + this.audioProcessService.verify(vec, audio_2) + this.audioProcessService.verify(vec, audio_3);
        if (result >= 2.0) {
            txtSpeech.setText("WELCOME");
            mp_y.start();
        }else{
            txtSpeech.setText("TRY AGAIN");
            mp_n.start();
        }
    }

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






    @Override
    protected void onStop() {
        super.onStop();
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