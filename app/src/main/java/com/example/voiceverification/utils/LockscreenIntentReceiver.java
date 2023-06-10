package com.example.voiceverification.utils;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.util.Log;

import com.example.voiceverification.MainActivity;
import com.example.voiceverification.VerifyActivity;

import java.util.Arrays;

public class LockscreenIntentReceiver extends BroadcastReceiver {

	// Handle actions and display Lockscreen
	@Override
	public void onReceive(Context context, Intent intent) {
		Log.d("tfliteSupport", intent.getAction());
//		if (intent.getAction().equals(Intent.ACTION_SCREEN_OFF)
//				|| intent.getAction().equals(Intent.ACTION_SCREEN_ON)
//				|| intent.getAction().equals(Intent.ACTION_BOOT_COMPLETED)) {
////			Log.d("tfliteSupport", intent.getAction());
//			start_lockscreen(context);
//		}
		start_lockscreen(context);

	}

	// Display lock screen
	private void start_lockscreen(Context context) {
		Intent mIntent = new Intent(context, VerifyActivity.class);
		mIntent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
		context.startActivity(mIntent);
	}

}
