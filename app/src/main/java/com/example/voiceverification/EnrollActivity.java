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

import com.example.voiceverification.services.AudioProcessService;
import com.example.voiceverification.services.VadService;


import org.checkerframework.checker.units.qual.A;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.Timer;

public class EnrollActivity extends AppCompatActivity {
    AudioRecord record = null;
    // UI elements.
    private static final int REQUEST_RECORD_AUDIO = 13;
    private static SharedPreferences prefs;
    private static Vibrator vibe;
    private static MediaPlayer mp;
    private static MediaPlayer mp_complete;
    private TextView txtSpeech;
    private static int numAudio = 0;
    private static final int SAMPLE_RATE = 16000;


    AudioProcessService audioProcessService = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_enroll);

        try {
            numAudio = 0;
            prefs = PreferenceManager.getDefaultSharedPreferences(this);
            this.audioProcessService = new AudioProcessService(this);
            requestMicrophonePermission();
            record = new AudioRecord(
                    MediaRecorder.AudioSource.DEFAULT,
                    SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    BUFFER_SIZE * 2);
            if (record.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e("tfliteSupport", "Audio Record can't initialize!");
                return;
            }
            txtSpeech = (TextView) findViewById(R.id.txtSpeech);
            getSupportActionBar().hide();
            mp = MediaPlayer.create(this, R.raw.alert);
            mp_complete = MediaPlayer.create(this, R.raw.ting);
            vibe = (Vibrator) this.getSystemService(Context.VIBRATOR_SERVICE);
        } catch (Exception e) {

        }
    }

    @Override
    protected void onDestroy() {
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
    private static final int BUFFER_SIZE = SAMPLE_RATE * BUFFER_DURATION_MS / 1000;


    public void onclickEnroll(View v){
        mp.start();
        vibe.vibrate(160);
        Toast.makeText(getBaseContext(), "Done", Toast.LENGTH_LONG).show();
        float[] vec = this.audioProcessService.voiceToVec(record);
        prefs.edit().putString("audio_"+ ++numAudio, Arrays.toString(vec)).apply();

        txtSpeech.setText("Let's say "+ (3-numAudio) +" more time(s)");
        mp_complete.start();
        if (numAudio >= 3){
            prefs.edit().putBoolean("isEnroll", true).apply();
            Intent returnIntent = new Intent();
            setResult(Activity.RESULT_CANCELED, returnIntent);
            finish();
        }
    }


}