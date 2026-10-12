"""Microphone check: records you once, then shows what each stage heard.

Run this when the voice loop isn't reacting. It puts the same five seconds of
audio through both models, which is what separates the failures that look
identical from the outside:

  nothing arrives            -> the device, or Windows microphone permission
  arrives but clipping       -> boost set too high; the loud parts square off
  arrives but quiet          -> mic gain, and the thresholds in vad.py
  Whisper hears it, wake word doesn't -> the phrase, or the wake stage itself

It also saves the recording, so one capture can be replayed against a changed
threshold or a different model instead of being re-recorded each time.

Everything runs through the same WakeWord and the same block size as main.py,
so a phrase that fires here fires there.

Run: python mic_check.py
"""

import sys
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

import vad

SAMPLERATE = 16000
SECONDS = 5
BLOCKSIZE = 1280  # main.py's block size, so this exercises the same path
PHRASE = "hey jarvis what time is it"


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

    # ---- the wake word --------------------------------------------------
    # Scored through the same WakeWord the loop uses, in the same block size,
    # so a phrase that fires here fires there.
    from wakeword import WakeWord, PHRASE, THRESHOLD

    detector = WakeWord()
    best, fired = 0.0, False
    for i in range(0, len(audio), BLOCKSIZE):
        hit = detector.heard(audio[i:i + BLOCKSIZE].tobytes())
        best = max(best, detector.last_score)
        fired = fired or hit

    print(f"\n[wake] best score {best:.3f}, threshold {THRESHOLD}")
    if fired:
        print(f'[wake] {PHRASE!r} FOUND -- the loop would wake on this.')
    elif best > THRESHOLD * 0.6:
        print(f'[wake] close, but under the line. Say "{PHRASE}" as two clear')
        print("       words, and check the levels above for clipping.")
    else:
        print(f'[wake] {PHRASE!r} not heard. Note it needs the "hey".')

    # ---- whisper, for comparison ----------------------------------------
    print("\n[whisper] loading...")
    from transcriber import Transcriber
    text = Transcriber().transcribe(audio)
    print(f"[whisper] {text!r}")

    # ---- verdict --------------------------------------------------------
    print("\n--- verdict ---")
    if fired:
        print("  Wake word works. If the loop still doesn't react, the problem is")
        print("  in main.py's stream, not in recognition.")
    elif text.strip():
        print("  Whisper heard you and the wake-word model didn't fire. The audio")
        print(f'  is fine, so this is about the phrase: it must be "{PHRASE}",')
        print("  said as two words, and the levels above must not be clipping.")
    else:
        print("  Neither heard anything. Treat this as a level problem first:")
        print("  check the [level] lines above before touching either model.")


if __name__ == "__main__":
    sys.exit(main())
