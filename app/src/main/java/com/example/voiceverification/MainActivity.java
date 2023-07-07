package com.example.voiceverification;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.PreferenceManager;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(this);
        boolean isEnroll = prefs.getBoolean("isEnroll", false);
        if (!isEnroll){
            int LAUNCH_ENROLL_ACTIVITY = 1;
            Intent i = new Intent(this, EnrollActivity.class);
            startActivityForResult(i, LAUNCH_ENROLL_ACTIVITY);
        }else{
            int LAUNCH_ENROLL_ACTIVITY = 1;
            Intent i = new Intent(this, VerifyActivity.class);
            startActivityForResult(i, LAUNCH_ENROLL_ACTIVITY);
        }
    }
}