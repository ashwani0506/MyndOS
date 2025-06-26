import simpleaudio as sa

def play_beep():
    wave_obj = sa.WaveObject.from_wave_file("assets/beep.wav")
    wave_obj.play()
