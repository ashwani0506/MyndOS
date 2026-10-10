"""Microphone check: records you once, then shows what each stage heard.

Run this when the voice loop isn't reacting. It puts the same five seconds of
audio through both recognisers, which is what separates the three failures that
look identical from the outside:

  nothing arrives          -> the device or Windows microphone permission
  audio arrives but quiet  -> mic gain, and the VAD thresholds in vad.py
  Whisper hears it, Vosk doesn't -> the wake-word stage specifically

Vosk is fed in the same half-second chunks as main.py, through the same
AcceptWaveform/Result cycle, so a wake word that fails here fails there too.

Run: python mic_check.py
"""

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import vosk

import vad

SAMPLERATE = 16000
SECONDS = 5
BLOCKSIZE = 8000  # main.py's chunk size, so this exercises the same path
PHRASE = "jarvis what time is it"


def main():
    din = sd.default.device[0]
    info = sd.query_devices(din)
    print(f"[device] {info['name']} (index {din}, native {int(info['default_samplerate'])}Hz)")
    print(f"[device] opening as {SAMPLERATE}Hz mono, which is what main.py asks for")

    print(f'\n>>> Say "{PHRASE}" clearly, starting now.')
    for n in (3, 2, 1):
        print(f"    {n}...", end="", flush=True)
        time.sleep(0.4)
    print(f" RECORDING {SECONDS}s -- speak!")

    audio = sd.rec(SECONDS * SAMPLERATE, samplerate=SAMPLERATE, channels=1,
                   dtype="int16", blocking=True).flatten()
    print("    done.\n")

    # ---- level ----------------------------------------------------------
    peak = int(np.abs(audio).max())
    overall = vad.rms(audio.tobytes())
    floor, loud = vad.levels(audio.tobytes(), SAMPLERATE)
    level = vad.threshold(floor, loud)
    loudest = max(
        vad.rms(audio[i:i + 480].tobytes()) for i in range(0, len(audio) - 480, 480)
    )
    print(f"[level] peak {peak} of 32768 ({100 * peak / 32768:.1f}% of full scale)")
    print(f"[level] overall rms {overall:.0f}, room {floor:.0f}, voice {loud:.0f}")
    print(f"[level] loudest 30ms frame {loudest:.0f}, vs vad threshold {level:.0f}")

    if peak == 0:
        print("\n  Nothing arrived at all. The device is wrong or Windows is")
        print("  blocking microphone access for Python (Settings > Privacy &")
        print("  security > Microphone). Nothing below will work until that does.")
        return
    if peak > 32000:
        print("\n  Clipping. The signal is hitting the ceiling and the loud parts")
        print("  are being squared off, which recognisers hear as distortion.")
        print("  Turn the Windows mic level or boost DOWN, not up.")
    if loudest < level:
        print("\n  Audio arrived, but nothing in it was loud enough to count as")
        print("  speech, so the VAD would never start recording a command.")
        print("  Raise the mic level in Windows, or lower MIN_RMS in vad.py.")

    # Kept so the same recording can be replayed against a different
    # recogniser without asking him to say it again -- which is the difference
    # between testing an idea in a second and in a minute.
    out = Path(__file__).parent / "cache" / "mic_check.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLERATE)
        w.writeframes(audio.tobytes())
    print(f"\n[saved] {out}")

    # ---- vosk, the wake word --------------------------------------------
    vosk.SetLogLevel(-1)
    rec = vosk.KaldiRecognizer(vosk.Model(lang="en-us"), SAMPLERATE)
    finals, partial = [], ""
    for i in range(0, len(audio), BLOCKSIZE):
        chunk = audio[i:i + BLOCKSIZE].tobytes()
        if rec.AcceptWaveform(chunk):
            finals.append(json.loads(rec.Result()).get("text", ""))
        else:
            partial = json.loads(rec.PartialResult()).get("partial", "")
    finals.append(json.loads(rec.FinalResult()).get("text", ""))
    heard = " ".join(t for t in finals if t.strip())

    print(f"\n[vosk] final:   {heard!r}")
    print(f"[vosk] partial: {partial!r}")
    # main.py only ever inspects Result(), so a word that shows up solely in a
    # partial is a word the loop never sees -- worth saying out loud.
    if "jarvis" in heard:
        print('[vosk] "jarvis" FOUND in a final result -- the loop would wake.')
    elif "jarvis" in partial:
        print('[vosk] "jarvis" is in the PARTIAL but not a final result. main.py')
        print("       only reads Result(), so it would not wake on this.")
    else:
        print('[vosk] "jarvis" not heard.')

    # ---- whisper, for comparison ----------------------------------------
    print("\n[whisper] loading...")
    from transcriber import Transcriber
    text = Transcriber().transcribe(audio)
    print(f"[whisper] {text!r}")

    # ---- verdict --------------------------------------------------------
    print("\n--- verdict ---")
    if "jarvis" in heard:
        print("  Wake word works. If the loop still doesn't react, the problem is")
        print("  in main.py's stream, not in recognition.")
    elif text.strip():
        print("  Whisper heard you and Vosk did not. The audio is fine and the")
        print("  wake-word stage is the problem -- which is the case for")
        print("  replacing Vosk with openWakeWord rather than tuning it.")
    else:
        print("  Neither heard anything. Treat this as a level problem first:")
        print("  check the [level] lines above before touching either model.")


if __name__ == "__main__":
    sys.exit(main())
