import pygame
import threading

audio_path = "audios/ani_ge_bird_taka02.mp3"
# audio_path = "audios/ani_ge_bird_taka03.mp3"

def init():
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)

def play():
    pygame.mixer.music.play()
    print("Audio started...")

    # 別スレッドで再生を監視する
    threading.Thread(target=wait_for_audio_end, daemon=True).start()

def wait_for_audio_end():
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # 10 FPSで待機
    print("Audio finished.")

def main():
    init()
    play()

    while pygame.mixer.music.get_busy():
        print("Processing while audio is playing...")
        pygame.time.Clock().tick(1)  # 1 FPSで他の処理を実行
