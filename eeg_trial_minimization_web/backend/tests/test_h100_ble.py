import pytest
import numpy as np
from httpx import AsyncClient, ASGITransport
from backend.main import app
from backend.adapters.h100_ble_adapter import H100Parser, H100BleAdapter, H100_LABELS, SYNC_MARKER, Cycle

def test_h100_parser_feed_and_cycles():
    parser = H100Parser()
    assert parser.last_counter is None

    # Feed 8 channels (markers 0xC0 to 0xC7) + 1 sync frame (marker SYNC_MARKER 0xD2)
    # Each frame has: 1 marker byte + 7 header bytes (0x80..0xBF) + 25 payload bytes (< 0x80)
    for ch in range(8):
        marker = 0xC0 + ch
        header = bytes([0x80] * 7)
        payload = bytes([64 + ch] * 25) # center = 64
        frame_bytes = bytes([marker]) + header + payload
        cycles = parser.feed(frame_bytes)
        assert len(cycles) == 0  # Not closed yet until SYNC frame arrives

    # Now feed SYNC frame
    sync_marker = SYNC_MARKER
    # counter = 100: header[0] = 100 & 0x3F = 36 | 0x80 = 0xA4, header[1] = (100 >> 6) & 0x3F = 1 | 0x80 = 0x81
    sync_header = bytes([0x80 | (100 & 0x3F), 0x80 | ((100 >> 6) & 0x3F), 0x80, 0x80, 0x80, 0x80, 0x80])
    sync_payload = bytes([0] * 25)
    sync_frame_bytes = bytes([sync_marker]) + sync_header + sync_payload

    cycles = parser.feed(sync_frame_bytes)
    assert len(cycles) == 1
    c = cycles[0]
    assert c.counter == 100
    assert c.lost_cycles == 0
    assert c.samples.shape == (25, 8)
    # Channel 0: 64 - 64.0 = 0.0
    assert np.allclose(c.samples[:, 0], 0.0)
    # Channel 7: 71 - 64.0 = 7.0
    assert np.allclose(c.samples[:, 7], 7.0)

def test_h100_parser_garbage_tolerance():
    parser = H100Parser()
    # Feed arbitrary invalid bytes (must not crash, bad_bytes counter increments)
    cycles = parser.feed(b"\x00\x01\x02\xFF\xAA\xBB")
    assert len(cycles) == 0
    assert parser.bad_bytes > 0

@pytest.mark.asyncio
async def test_hardware_gain_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.post("/api/hardware/gain", json={"gain": 2.5})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert data["gain"] == 2.5

@pytest.mark.asyncio
async def test_ble_adapters_listed():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        res = await ac.get("/api/hardware/adapters")
        assert res.status_code == 200
        adapters = res.json()
        ids = [a["id"] for a in adapters]
        assert "h100_ble" in ids

@pytest.mark.asyncio
async def test_hardware_filter_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Set filter to clean
        res = await ac.post("/api/hardware/filter", json={"mode": "clean"})
        assert res.status_code == 200
        assert res.json()["filter_mode"] == "clean"

        # Check filter
        res2 = await ac.get("/api/hardware/filter")
        assert res2.status_code == 200
        assert res2.json()["filter_mode"] == "clean"

        # Set to raw
        res3 = await ac.post("/api/hardware/filter", json={"mode": "raw"})
        assert res3.status_code == 200
        assert res3.json()["filter_mode"] == "raw"

def test_h100_adapter_filter_attenuation():
    adapter = H100BleAdapter()
    assert adapter.filters_available is True
    adapter.set_filter_mode("clean")

    # Generate 50 Hz noise (30 uV) + 10 Hz alpha wave (10 uV)
    fs = 250.0
    t = np.arange(25) / fs
    noise_50 = 30.0 * np.sin(2 * np.pi * 50.0 * t)[:, None]
    clean_10 = 10.0 * np.sin(2 * np.pi * 10.0 * t)[:, None]
    samples = np.repeat(noise_50 + clean_10, 8, axis=1)

    c = Cycle(samples=samples, counter=1, lost_cycles=0)
    adapter._process_cycle(c)
    assert len(adapter.chunk_queue) == 1
    chunk = adapter.chunk_queue.popleft()
    out = np.array(chunk["samples"]) # (8, 25)
    # The 50 Hz powerline hum should be attenuated by the notch filter
    assert np.std(out[0]) < np.std(samples[:, 0])
