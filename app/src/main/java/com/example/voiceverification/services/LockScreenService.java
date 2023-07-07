package com.example.voiceverification.services;

import android.app.Service;
import android.content.BroadcastReceiver;
import android.content.Intent;
import android.content.IntentFilter;
import android.content.res.Configuration;
import android.graphics.Color;
import android.graphics.PixelFormat;
import android.os.IBinder;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;

import com.example.voiceverification.R;
import com.example.voiceverification.VerifyActivity;
import com.example.voiceverification.receivers.ScreenOffReceiver;

public class LockScreenService extends Service {

	private WindowManager windowManager;
	private View lockScreenView;
	private BroadcastReceiver broadcastReceiver = new ScreenOffReceiver();


	@Override
	public IBinder onBind(Intent intent) {
		return null;
	}

	@Override
	public void onCreate() {
		super.onCreate();
		// Register the ScreenOnOffReceiver
		IntentFilter filter = new IntentFilter();
		filter.addAction(Intent.ACTION_SCREEN_ON);
		filter.addAction(Intent.ACTION_SCREEN_OFF);
		registerReceiver(broadcastReceiver, filter);

		windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);

		lockScreenView = LayoutInflater.from(this).inflate(R.layout.activity_verify, null);
		Button verifyButton = lockScreenView.findViewById(R.id.btnSpeak);
		verifyButton.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View view) {

			}
		});
	}

	boolean isViewAdded = false;

	// Register for Lockscreen event intents
	@Override
	public int onStartCommand(Intent intent, int flags, int startId) {
		if (!this.isViewAdded) {
			WindowManager.LayoutParams params = new WindowManager.LayoutParams(
					WindowManager.LayoutParams.MATCH_PARENT,
					WindowManager.LayoutParams.MATCH_PARENT,
					WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
					WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
							| WindowManager.LayoutParams.FLAG_NOT_TOUCH_MODAL
							| WindowManager.LayoutParams.FLAG_WATCH_OUTSIDE_TOUCH,
					PixelFormat.TRANSLUCENT);

			lockScreenView.setBackgroundColor(Color.WHITE);
			windowManager.addView(lockScreenView, params);
			this.isViewAdded = true;
		}

		return START_STICKY;
	}

	@Override
	public void onConfigurationChanged(Configuration newConfig) {
		super.onConfigurationChanged(newConfig);
	}

	@Override
	public void onDestroy() {
		super.onDestroy();
		Log.d("lock_screen_service", "onDestroy: " + "Hello");
		if (lockScreenView != null) {
			windowManager.removeView(lockScreenView);
			unregisterReceiver(broadcastReceiver);
		}
	}
}
