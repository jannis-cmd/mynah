from __future__ import annotations

from hashlib import sha256

from app import ble_sync


class FakeTransport(ble_sync.BleTransport):
    def __init__(self, objects: dict[str, bytes], manifest_text: str) -> None:
        super().__init__()
        self._objects = objects
        self._manifest_text = manifest_text
        self._fetch_id = ""
        self._fetch_offset = 0
        self._fetch_len = 0
        self._committed = ""
        self._connected = False

    async def resolve_address(self, address: str | None, name_prefix: str, scan_timeout_s: float) -> str:  # noqa: ARG002
        return address or "AA:BB:CC:DD:EE:FF"

    async def connect(self, address: str) -> None:  # noqa: ARG002
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    async def read(self, uuid: str) -> bytes:
        assert self._connected
        if uuid == ble_sync.CHAR_UUID_DEVICE_INFO:
            return b"device_id=testwearable|fw=0.7.0|cap=hr,audio,ble_sync"
        if uuid == ble_sync.CHAR_UUID_STATUS:
            return b"status=ok"
        if uuid == ble_sync.CHAR_UUID_MANIFEST:
            return self._manifest_text.encode("utf-8")
        if uuid == ble_sync.CHAR_UUID_FETCH_DATA:
            data = self._objects[self._fetch_id]
            return data[self._fetch_offset : self._fetch_offset + self._fetch_len]
        if uuid == ble_sync.CHAR_UUID_WIPE_CONFIRM:
            return f"ok|commit={self._committed}".encode("utf-8")
        raise AssertionError(f"unexpected read uuid {uuid}")

    async def write(self, uuid: str, data: bytes) -> None:
        assert self._connected
        if uuid == ble_sync.CHAR_UUID_FETCH_REQ:
            text = data.decode("ascii")
            object_id, offset, length = text.split("|", 2)
            self._fetch_id = object_id
            self._fetch_offset = int(offset)
            self._fetch_len = int(length)
            return
        if uuid == ble_sync.CHAR_UUID_COMMIT_SYNC:
            self._committed = data.decode("ascii")
            return
        if uuid == ble_sync.CHAR_UUID_TIME_SYNC:
            int(data.decode("ascii"))
            return
        raise AssertionError(f"unexpected write uuid {uuid}")


def test_parse_manifest():
    hr_payload = b"ts_ms,bpm,quality,red,ir\n1700000000000,70,100,123,456\n"
    audio_payload = b"\x01\x02\x03\x04"
    manifest = "\n".join(
        [
            f"hr.csv|{len(hr_payload)}|{sha256(hr_payload).hexdigest()}",
            f"audio_1.pcm|{len(audio_payload)}|{sha256(audio_payload).hexdigest()}|1700000001000|1700000002000|1000|10|16000|1200",
        ]
    )
    out = ble_sync.parse_manifest(manifest)
    assert len(out) == 2
    assert out[0].object_id == "hr.csv"
    assert out[0].kind == "hr"
    assert out[1].object_id == "audio_1.pcm"
    assert out[1].kind == "audio"
    assert int(out[1].meta["duration_ms"]) == 1000


def test_sync_wearable_ble_with_fake_transport():
    hr_payload = b"ts_ms,bpm,quality,red,ir\n1700000000000,70,100,123,456\n"
    audio_payload = b"\x01\x02\x03\x04\x05\x06"
    hr_sha = sha256(hr_payload).hexdigest()
    audio_sha = sha256(audio_payload).hexdigest()
    manifest = "\n".join(
        [
            f"hr.csv|{len(hr_payload)}|{hr_sha}",
            f"audio_2.pcm|{len(audio_payload)}|{audio_sha}|1700000003000|1700000005000|2000|20|32000|850",
        ]
    )
    transport = FakeTransport(objects={"hr.csv": hr_payload, "audio_2.pcm": audio_payload}, manifest_text=manifest)
    # Run async path directly with fake transport to avoid bleak dependency in tests.
    import asyncio

    result = asyncio.run(
        ble_sync.sync_wearable_ble(
            address="AA:BB:CC:DD:EE:FF",
            name_prefix="MYNAH-WEARABLE",
            chunk_size=3,
            transport=transport,
            scan_timeout_s=1.0,
        )
    )
    assert result["address"] == "AA:BB:CC:DD:EE:FF"
    assert len(result["objects"]) == 2
    assert result["objects"][0]["object_id"] == "hr.csv"
    assert result["objects"][1]["object_id"] == "audio_2.pcm"
    assert "hr.csv" in result["committed_ids"]
    assert "audio_2.pcm" in result["committed_ids"]
