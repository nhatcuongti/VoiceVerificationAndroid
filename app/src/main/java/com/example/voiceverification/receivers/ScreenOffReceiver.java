package com.example.voiceverification.receivers;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import com.example.voiceverification.services.LockScreenService;

public class ScreenOffReceiver extends BroadcastReceiver {
	@Override
	public void onReceive(Context context, Intent intent) {
		if (Intent.ACTION_SCREEN_ON.equals(intent.getAction())) {
			// Screen is turned on, show your lock screen here
			Intent lockScreenIntent = new Intent(context, LockScreenService.class);
			context.startService(lockScreenIntent);
		}
	}
}
