import sounddevice as sd
import numpy as np
from queue import Queue

TARGET_FREQ = 350
FREQ_TOLERANCE = 100
STRENGTH_THRESHOLD = 450

FS = 44100
SIZE = 1024

ALPHA = 0.6

BIT_START = 2.0
BIT_END = 1.5
BIT_HIGH = 1.0
BIT_LOW = 0.5

ERROR = 0.25

audio_queue = Queue()


def callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


stream = sd.InputStream(
    samplerate=FS,
    channels=1,
    blocksize=SIZE,
    callback=callback
)

stream.start()

q = []
output = ""
receiving = False

time_step = SIZE / FS

print("Listening...")

while True:
    block = audio_queue.get()
    samples = block[:, 0]

    fft = np.fft.rfft(samples)
    fft_mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(samples), 1 / FS)

    band = np.where(
        (freqs >= TARGET_FREQ - FREQ_TOLERANCE) &
        (freqs <= TARGET_FREQ + FREQ_TOLERANCE)
    )

    energy = np.sum(fft_mag[band])

    if energy > STRENGTH_THRESHOLD:
        q.append(energy)

        # Print energy whenever the target band is detected
        #print(f"Energy: {energy:.2f}")

    elif q:
        peak = max(q)
        duration = 0

        for value in q:
            if value >= peak * ALPHA:
                duration += time_step

        #print(f"Duration: {duration:.2f}s")

        if abs(duration - BIT_START) <= ERROR:
            print("START")
            receiving = True

        elif receiving and abs(duration - BIT_HIGH) <= ERROR:
            output += "1"
            print("Bit: 1")

        elif receiving and abs(duration - BIT_LOW) <= ERROR:
            output += "0"
            print("Bit: 0")

        elif receiving and abs(duration - BIT_END) <= ERROR:
            print("END")
            print("Decoded:", output)

            receiving = False
            output = ""

        q.clear()