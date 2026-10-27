from __future__ import annotations
import argparse
import asyncio
import time
from dataclasses import dataclass

import numpy as np

HELLO_BYTE = b"\x0B"            # cap replies 0x87
START_BYTE = b"\xAA"
RX_UUID = "49535343-8841-43f4-a8d4-ecbe34729bb3"
NUM_CHANNELS = 8
SAMPLES_PER_FRAME = 25
SAMPLE_RATE = 250.0
SYNC_MARKER = 0xD2
COUNTER_MOD = 4096
CENTER = 64.0                      # 7-bit payload midpoint
H100_LABELS = ["FP1", "FP2", "O1", "O2", "C3", "Cz", "Fz", "C4"]


@dataclass
class Cycle:
    samples: np.ndarray            # (25, 8) centred counts
    counter: int                   # sample counter from the sync frame
    lost_cycles: int               # 100 ms slots with no EEG immediately before this cycle (the bridge interpolates across them)


class H100Parser:
    """Turns the raw notification bytes into complete 25-sample, 8-channel cycles. Resynchronises after any garbage or
    dropped bytes: a frame is only accepted if its header and payload bytes have the expected bit patterns."""

    def __init__(self):
        self.buffer = bytearray()
        self.frames: dict[int, np.ndarray] = {}
        self.pending_lost = 0
        self.last_counter: int | None = None
        self.total_slots = 0
        self.missing_slots = 0
        self.bad_bytes = 0
        self.dropped_cycles = 0

    def feed(self, data: bytes) -> list[Cycle]:
        self.buffer.extend(data)
        cycles: list[Cycle] = []
        buf = self.buffer
        i = 0
        while i < len(buf):
            marker = buf[i]
            if not 0xC0 <= marker <= 0xDF:
                self.bad_bytes += 1
                i += 1
                continue
            if len(buf) - i < 8:
                break                                            # header not complete yet
            header = buf[i + 1:i + 8]
            if not all(0x80 <= b < 0xC0 for b in header):
                self.bad_bytes += 1
                i += 1
                continue
            j = i + 8
            while j < len(buf) and buf[j] < 0x80:
                j += 1
            payload_len = j - (i + 8)
            known = marker == SYNC_MARKER or 0xC0 <= marker < 0xC0 + NUM_CHANNELS
            if known and payload_len >= SAMPLES_PER_FRAME:
                payload = bytes(buf[i + 8:i + 8 + SAMPLES_PER_FRAME])
                end = i + 8 + SAMPLES_PER_FRAME
            elif j >= len(buf):
                break                                            # payload may still be arriving
            else:
                payload, end = bytes(buf[i + 8:j]), j
            i = end
            if not known or payload_len != SAMPLES_PER_FRAME:
                continue                                         # other frame kinds (extra streams) are ignored
            if marker == SYNC_MARKER:
                counter = ((header[1] & 0x3F) << 6) | (header[0] & 0x3F)
                cycle = self._close_cycle(counter)
                if cycle is not None:
                    cycles.append(cycle)
            else:
                self.frames[marker - 0xC0] = np.frombuffer(payload, dtype=np.uint8).astype(np.float64) - CENTER
        del buf[:i]
        return cycles

    def _close_cycle(self, counter: int) -> Cycle | None:
        have = self.frames
        self.frames = {}
        if self.last_counter is not None:
            step = (counter - self.last_counter) % COUNTER_MOD
            if step % SAMPLES_PER_FRAME == 0 and step > SAMPLES_PER_FRAME:
                skipped = step // SAMPLES_PER_FRAME - 1       # whole sync frames that never arrived
                self.pending_lost += skipped
                self.total_slots += skipped
                self.missing_slots += skipped
        self.last_counter = counter
        self.total_slots += 1
        if len(have) < NUM_CHANNELS:
            self.dropped_cycles += 1
            self.missing_slots += 1
            self.pending_lost += 1
            return None
        samples = np.stack([have[k] for k in range(NUM_CHANNELS)], axis=1)
        lost, self.pending_lost = self.pending_lost, 0
        return Cycle(samples=samples, counter=counter, lost_cycles=lost)


def make_outlet(name: str, source_id: str):
    import pylsl
    info = pylsl.StreamInfo(name, "EEG", NUM_CHANNELS, SAMPLE_RATE, pylsl.cf_float32, source_id)
    channels = info.desc().append_child("channels")
    for label in H100_LABELS:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "counts (centred, unscaled)")
        ch.append_child_value("type", "EEG")
    info.desc().append_child_value("source", "q.neuro h100 over BLE, reverse-engineered decode (see eeg/h100_ble.py)")
    return pylsl.StreamOutlet(info, chunk_size=SAMPLES_PER_FRAME)


class H100StartError(RuntimeError):
    """The cap connected but never started streaming."""


class BridgeStats:
    def __init__(self):
        self.parser = H100Parser()
        self.samples = 0                 # samples pushed since the last report
        self.since = time.time()
        self.started = False
        self.prev_last: np.ndarray | None = None

    @property
    def missing_share(self) -> float:
        return 100.0 * self.parser.missing_slots / max(self.parser.total_slots, 1)


def is_h100(device, address: str | None = None) -> bool:
    return bool(device.name) and device.name.upper().startswith("H100") and (not address or device.address == address)


async def find_device(address: str | None = None, timeout: float = 30.0):
    from bleak import BleakScanner
    return await BleakScanner.find_device_by_filter(lambda d, a: is_h100(d, address), timeout=timeout)


async def stream_session(device, outlet, stats: BridgeStats, scale: float = 1.0, on_started=None) -> None:
    """One Bluetooth session: connect, say hello, start the stream, push every decoded cycle (gaps interpolated) to the
    LSL outlet until the link drops. Raises H100StartError if the cap never starts streaming."""
    from bleak import BleakClient

    def on_data(_, data: bytearray):
        for cycle in stats.parser.feed(bytes(data)):
            block = cycle.samples
            if cycle.lost_cycles and stats.prev_last is not None:
                n = SAMPLES_PER_FRAME * cycle.lost_cycles
                ramp = (np.arange(1, n + 1) / (n + 1))[:, None]
                block = np.vstack([stats.prev_last + (block[0] - stats.prev_last) * ramp, block])
            stats.prev_last = cycle.samples[-1]
            outlet.push_chunk((block * scale).astype(np.float32).tolist())
            stats.samples += block.shape[0]
            if not stats.started:
                stats.started = True
                if on_started is not None:
                    on_started()

    async with BleakClient(device, timeout=20) as client:
        for service in client.services:
            for ch in service.characteristics:
                if "notify" in ch.properties or "indicate" in ch.properties:
                    await client.start_notify(ch.uuid, on_data)
        await asyncio.sleep(0.8)
        for _ in range(5):
            await client.write_gatt_char(RX_UUID, HELLO_BYTE, response=False)
            await asyncio.sleep(0.7)
            await client.write_gatt_char(RX_UUID, START_BYTE, response=False)
            await asyncio.sleep(1.5)
            if stats.started:
                break
        if not stats.started:
            raise H100StartError("The cap did not start streaming. Switch it off and on, then try again.")
        while client.is_connected:
            await asyncio.sleep(0.5)


async def run(args) -> None:
    print("Scanning for the H100 (cap on, not connected to another device)...", flush=True)
    device = await find_device(args.address, timeout=30)
    if device is None:
        raise SystemExit("H100 not found. Switch the cap on, close the vendor software, and keep it within a metre.")
    print(f"Found {device.name}; connecting...", flush=True)
    outlet = make_outlet(args.name, f"h100-ble-{device.name}")
    stats = BridgeStats()

    async def report():
        while True:
            await asyncio.sleep(5)
            elapsed = time.time() - stats.since
            print(f"  {stats.samples / elapsed:6.1f} samples/s/ch, slots missing from the cap {stats.missing_share:.1f}% (interpolated), "
                  f"stray bytes {stats.parser.bad_bytes}", flush=True)
            stats.samples, stats.since = 0, time.time()

    reporter = asyncio.create_task(report())
    try:
        await stream_session(device, outlet, stats, args.scale,
                             on_started=lambda: print(f"Streaming: LSL '{args.name}' (type EEG), 8 ch @ 250 Hz, values = centred counts x {args.scale:g}. Ctrl+C to stop.", flush=True))
    except H100StartError as err:
        raise SystemExit(str(err))
    finally:
        reporter.cancel()
    print("Bluetooth link lost.", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the Q.Neuro H100 (BLE) as an LSL stream.")
    ap.add_argument("--name", default="H100_BLE", help="LSL stream name")
    ap.add_argument("--address", default=None, help="BLE address/UUID to pick one cap when several are in range")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply centred counts by this (microvolts per count, unknown until calibrated)")
    args = ap.parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()