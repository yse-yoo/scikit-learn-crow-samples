import os
import cv2
import utils
import sound  # 音声再生モジュール
import time  # 時間の計測用

# 動画ファイルのパスを標準入力から取得
video_name = input("Enter the name of the video file (without extension): ").strip()
video_path = utils.get_video_path(video_name + ".mp4")

# 閾値
watch_time = 2.0
positive_ratio_limit = 0.6

# 動画ファイルが存在するかチェック
if not os.path.exists(video_path):
    print(f"Error: The file '{video_path}' does not exist.")
    exit(1)

# scikit-learnモデルの読み込み
model_name = "crow_classifier.pkl"
model = utils.load_model(model_name)

cap = cv2.VideoCapture(video_path)

paused = False  # 再生・停止状態を管理
positive_count = 0  # Positiveの検出回数
frame_count = 0  # 処理したフレーム数
start_time = time.time()  # 5秒間の計測開始

try:
    while cap.isOpened():  # 動画が正常に読み込まれている間
        if not paused:  # 再生中の場合のみフレームを取得
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read the frame. Exiting...")
                break

            # グレースケール変換とリサイズ
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64)).flatten().reshape(1, -1)

            # 予測の実行
            prediction = model.predict(resized)
            label = 'Positive' if prediction == 1 else ''

            # Positiveの検出をカウント
            if label == 'Positive':
                positive_count += 1

            # 総フレーム数を更新
            frame_count += 1

            # 結果と枠付きフレームを表示
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Video Prediction', frame)

            # 秒間の経過を確認
            elapsed_time = time.time() - start_time
            if elapsed_time >= watch_time:
                positive_ratio = positive_count / frame_count if frame_count > 0 else 0

                # Positiveの割合がリミット以上なら音声を再生
                if positive_ratio >= positive_ratio_limit:
                    sound.main()
                    # カウントとタイマーをリセット
                    positive_count = 0
                    frame_count = 0
                    start_time = time.time()  # 新しい計測開始

        # キー入力の確認
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESCキーで終了
            print("ESC pressed. Exiting...")
            break
        elif key == ord(' '):  # スペースキーで再生・停止を切り替え
            paused = not paused
            if paused:
                print("Video paused. Press SPACE to resume.")
            else:
                print("Video resumed.")

        # ウィンドウが閉じられた場合の確認
        if cv2.getWindowProperty('Video Prediction', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed. Exiting...")
            break

except KeyboardInterrupt:
    print("KeyboardInterrupt detected. Exiting gracefully...")

finally:
    # 動画とウィンドウの解放
    cap.release()
    cv2.destroyAllWindows()
    print("Resources released successfully.")
