import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# ============================================================
# 1. CONFIGURATION
# ============================================================

TARGET_FREQ = 215
FREQ_TOLERANCE = 100
STRENGTH_THRESHOLD = 0.1

alpha = 0.6
FS = 44100
size = 1024

# Use the exact block duration
time_step = size / FS
error = 0.25

# Bit definitions
bit_start = 2.0
bit_end = 1.5
bit_high = 1.0
bit_low = 0.5


# ============================================================
# 2. LOAD AUDIO
# ============================================================

fs_read, data = wav.read("recording.wav")

print(f"Audio sample rate: {fs_read} Hz")
print(f"Expected sample rate: {FS} Hz")

# Make sure audio is mono
if data.ndim > 1:
    data = data[:, 0]

# Convert audio to float
if np.issubdtype(data.dtype, np.integer):
    info = np.iinfo(data.dtype)
    data = data.astype(np.float64) / max(abs(info.min), info.max)
else:
    data = data.astype(np.float64)

# Check sample rate
if fs_read != FS:
    print(
        f"WARNING: recording is {fs_read} Hz, "
        f"but decoder expects {FS} Hz."
    )

# Use actual recording sample rate for processing
FS = fs_read

time_step = size / FS


# ============================================================
# 3. PROCESS AUDIO
# ============================================================

num_blocks = len(data) // size

block_times = []
block_energies = []
calculated_pulses = []

current_q = []
q_start_time = None

# Decoder state
status = False
output = ""


def process_pulse(
    current_q,
    q_start_time,
    q_end_time,
    status,
    output
):
    """
    Process one detected pulse and return:
        pulse information
        updated status
        updated output
    """

    if not current_q:
        return None, status, output

    peak = max(current_q)

    # Duration based on alpha criterion
    duration = sum(
        time_step
        for val in current_q
        if val >= peak * alpha
    )

    # Raw duration
    raw_duration = len(current_q) * time_step

    event_type = "No Match"

    # --------------------------------------------------------
    # State machine
    # --------------------------------------------------------

    if bit_start - error <= duration <= bit_start + error:

        status = True
        event_type = "START"

    elif status and bit_high - error <= duration <= bit_high + error:

        output += "1"
        event_type = f"Bit '1'\nAccum: {output}"

    elif status and bit_low - error <= duration <= bit_low + error:

        output += "0"
        event_type = f"Bit '0'\nAccum: {output}"

    elif status and bit_end - error <= duration <= bit_end + error:

        event_type = f"END\nFinal: {output}"

        status = False
        output = ""

    pulse = {
        "start": q_start_time,
        "end": q_end_time,
        "raw_dur": raw_duration,
        "filt_dur": duration,
        "peak": peak,
        "active_thresh": peak * alpha,
        "event": event_type,
    }

    return pulse, status, output


# ============================================================
# 4. BLOCK-BY-BLOCK FFT PROCESSING
# ============================================================

for i in range(num_blocks):

    start_idx = i * size
    end_idx = start_idx + size

    samples = data[start_idx:end_idx]

    # Time at center of block
    t_center = (start_idx + size / 2) / FS
    block_times.append(t_center)

    # --------------------------------------------------------
    # FFT
    # --------------------------------------------------------

    fft = np.fft.rfft(samples)
    fft_mag = np.abs(fft)

    freqs = np.fft.rfftfreq(
        len(samples),
        d=1 / FS
    )

    # Target frequency band
    band = (
        (freqs >= TARGET_FREQ - FREQ_TOLERANCE) &
        (freqs <= TARGET_FREQ + FREQ_TOLERANCE)
    )

    energy = np.sum(fft_mag[band])

    block_energies.append(energy)

    # --------------------------------------------------------
    # Pulse detection
    # --------------------------------------------------------

    if energy > STRENGTH_THRESHOLD:

        # Start of a new pulse
        if len(current_q) == 0:
            q_start_time = start_idx / FS

        current_q.append(energy)

    elif current_q:

        # Pulse ended
        q_end_time = start_idx / FS

        pulse, status, output = process_pulse(
            current_q,
            q_start_time,
            q_end_time,
            status,
            output
        )

        if pulse is not None:
            calculated_pulses.append(pulse)

        current_q.clear()
        q_start_time = None


# ============================================================
# 5. PROCESS FINAL PULSE
# ============================================================

# If recording ends while a pulse is still active,
# process that pulse too.

if current_q:

    q_end_time = len(data) / FS

    pulse, status, output = process_pulse(
        current_q,
        q_start_time,
        q_end_time,
        status,
        output
    )

    if pulse is not None:
        calculated_pulses.append(pulse)


# ============================================================
# 6. PRINT DECODER RESULTS
# ============================================================

print("\n==============================")
print("DETECTED PULSES")
print("==============================")

for i, pulse in enumerate(calculated_pulses):

    print(
        f"\nPulse {i + 1}"
        f"\n  Start:      {pulse['start']:.3f}s"
        f"\n  End:        {pulse['end']:.3f}s"
        f"\n  Raw:        {pulse['raw_dur']:.3f}s"
        f"\n  Filtered:   {pulse['filt_dur']:.3f}s"
        f"\n  Peak:       {pulse['peak']:.3f}"
        f"\n  Alpha:      {pulse['active_thresh']:.3f}"
        f"\n  Event:      {pulse['event'].replace(chr(10), ' | ')}"
    )


# ============================================================
# 7. PLOT
# ============================================================

plt.figure(figsize=(14, 7))

plt.plot(
    block_times,
    block_energies,
    label=f"Target Energy ({TARGET_FREQ} Hz)",
    linewidth=1.5
)

plt.axhline(
    y=STRENGTH_THRESHOLD,
    linestyle="--",
    alpha=0.7,
    label=f"Base Threshold ({STRENGTH_THRESHOLD})"
)


# ------------------------------------------------------------
# Overlay detected pulses
# ------------------------------------------------------------

for idx, pulse in enumerate(calculated_pulses):

    # Detected window
    plt.axvspan(
        pulse["start"],
        pulse["end"],
        alpha=0.1,
        label="Detected Window" if idx == 0 else ""
    )

    # Dynamic alpha threshold
    plt.hlines(
        y=pulse["active_thresh"],
        xmin=pulse["start"],
        xmax=pulse["end"],
        linestyle="-",
        linewidth=2.5,
        label=(
            "Active Alpha Threshold (peak × alpha)"
            if idx == 0
            else ""
        )
    )

    # Label position
    mid_time = (
        pulse["start"] + pulse["end"]
    ) / 2

    label_text = (
        f"Filt: {pulse['filt_dur']:.3f}s\n"
        f"👉 {pulse['event']}"
    )

    is_valid_match = pulse["event"] != "No Match"

    bg_color = "orange" if is_valid_match else "white"

    plt.text(
        mid_time,
        pulse["peak"] + pulse["peak"] * 0.05,
        label_text,
        color="black",
        weight="bold",
        ha="center",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.4",
            fc=bg_color,
            ec="purple",
            alpha=0.9
        )
    )


# ============================================================
# 8. GRAPH FORMATTING
# ============================================================

plt.title(
    f"Updated Pulse Visualizer "
    f"({TARGET_FREQ} Hz) | Tolerance ±{error}s",
    fontsize=14
)

plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Energy Magnitude", fontsize=12)

plt.grid(
    True,
    linestyle=":",
    alpha=0.5
)

plt.legend(loc="upper right")

plt.tight_layout()
plt.show()