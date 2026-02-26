from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any


def _uuid16(v: int) -> str:
    return f"0000{v:04x}-0000-1000-8000-00805f9b34fb"


CHAR_UUID_DEVICE_INFO = _uuid16(0xFFF1)
CHAR_UUID_STATUS = _uuid16(0xFFF2)
CHAR_UUID_MANIFEST = _uuid16(0xFFF3)
CHAR_UUID_FETCH_REQ = _uuid16(0xFFF4)
CHAR_UUID_FETCH_DATA = _uuid16(0xFFF5)
CHAR_UUID_COMMIT_SYNC = _uuid16(0xFFF6)
CHAR_UUID_WIPE_CONFIRM = _uuid16(0xFFF7)
CHAR_UUID_TIME_SYNC = _uuid16(0xFFF8)

HR_OBJECT_ID = "hr.csv"


@dataclass
class ManifestObject:
    object_id: str
    size: int
    sha256_hex: str
    kind: str
    meta: dict[str, Any]


def parse_manifest(manifest_text: str) -> list[ManifestObject]:
    objects: list[ManifestObject] = []
    for raw_line in manifest_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        object_id = parts[0]
        try:
            size = int(parts[1])
        except ValueError:
            continue
        sha = parts[2].lower()
        if len(sha) != 64:
            continue

        if object_id == HR_OBJECT_ID:
            objects.append(
                ManifestObject(
                    object_id=object_id,
                    size=size,
                    sha256_hex=sha,
                    kind="hr",
                    meta={},
                )
            )
            continue

        if object_id.startswith("audio_") and len(parts) >= 9:
            try:
                start_ms = int(parts[3])
                end_ms = int(parts[4])
                duration_ms = int(parts[5])
                frames = int(parts[6])
                samples = int(parts[7])
                avg_rms_x100 = int(parts[8])
            except ValueError:
                continue
            objects.append(
                ManifestObject(
                    object_id=object_id,
                    size=size,
                    sha256_hex=sha,
                    kind="audio",
                    meta={
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_ms": duration_ms,
                        "frames": frames,
                        "samples": samples,
                        "avg_rms_x100": avg_rms_x100,
                    },
                )
            )
    return objects


class BleTransport:
    def __init__(self) -> None:
        self._client = None

    async def resolve_address(self, address: str | None, name_prefix: str, scan_timeout_s: float) -> str:
        if address:
            return address
        try:
            from bleak import BleakScanner
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("bleak not available") from exc
        devices = await BleakScanner.discover(timeout=scan_timeout_s)
        for d in devices:
            name = (d.name or "").strip()
            if name.startswith(name_prefix):
                return d.address
        raise RuntimeError(f"wearable not found (name prefix: {name_prefix})")

    async def connect(self, address: str) -> None:
        try:
            from bleak import BleakClient
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("bleak not available") from exc
        self._client = BleakClient(address)
        ok = await self._client.connect()
        if not ok:
            raise RuntimeError(f"ble connect failed: {address}")

    async def disconnect(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            finally:
                self._client = None

    async def read(self, uuid: str) -> bytes:
        if self._client is None:
            raise RuntimeError("ble not connected")
        return bytes(await self._client.read_gatt_char(uuid))

    async def write(self, uuid: str, data: bytes) -> None:
        if self._client is None:
            raise RuntimeError("ble not connected")
        await self._client.write_gatt_char(uuid, data, response=True)


async def sync_wearable_ble(
    *,
    address: str | None,
    name_prefix: str,
    chunk_size: int,
    scan_timeout_s: float = 8.0,
    transport: BleTransport | None = None,
) -> dict[str, Any]:
    if chunk_size <= 0 or chunk_size > 240:
        raise ValueError("chunk_size must be between 1 and 240")

    t = transport or BleTransport()
    resolved = await t.resolve_address(address, name_prefix, scan_timeout_s)
    await t.connect(resolved)
    try:
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        await t.write(CHAR_UUID_TIME_SYNC, str(now_ms).encode("ascii"))

        device_info = (await t.read(CHAR_UUID_DEVICE_INFO)).decode("utf-8", errors="replace")
        status_before = (await t.read(CHAR_UUID_STATUS)).decode("utf-8", errors="replace")
        manifest_text = (await t.read(CHAR_UUID_MANIFEST)).decode("utf-8", errors="replace")
        manifest = parse_manifest(manifest_text)

        objects_out: list[dict[str, Any]] = []
        committed_ids: list[str] = []
        for obj in manifest:
            buf = bytearray()
            offset = 0
            while offset < obj.size:
                req_len = min(chunk_size, obj.size - offset)
                req = f"{obj.object_id}|{offset}|{req_len}".encode("ascii")
                await t.write(CHAR_UUID_FETCH_REQ, req)
                chunk = await t.read(CHAR_UUID_FETCH_DATA)
                if not chunk:
                    break
                buf.extend(chunk)
                offset += len(chunk)

            if len(buf) != obj.size:
                raise RuntimeError(f"object size mismatch for {obj.object_id}: got={len(buf)} expected={obj.size}")
            actual_sha = sha256(bytes(buf)).hexdigest()
            if actual_sha != obj.sha256_hex:
                raise RuntimeError(f"sha mismatch for {obj.object_id}: got={actual_sha} expected={obj.sha256_hex}")

            committed_ids.append(obj.object_id)
            objects_out.append(
                {
                    "object_id": obj.object_id,
                    "kind": obj.kind,
                    "size": obj.size,
                    "sha256": actual_sha,
                    "meta": obj.meta,
                    "payload": bytes(buf),
                }
            )

        if committed_ids:
            await t.write(CHAR_UUID_COMMIT_SYNC, ",".join(committed_ids).encode("ascii"))
        wipe_confirm = (await t.read(CHAR_UUID_WIPE_CONFIRM)).decode("utf-8", errors="replace")
        status_after = (await t.read(CHAR_UUID_STATUS)).decode("utf-8", errors="replace")

        return {
            "address": resolved,
            "device_info": device_info,
            "status_before": status_before,
            "status_after": status_after,
            "manifest_text": manifest_text,
            "wipe_confirm": wipe_confirm,
            "objects": objects_out,
            "committed_ids": committed_ids,
        }
    finally:
        await t.disconnect()


def sync_wearable_ble_blocking(
    *,
    address: str | None,
    name_prefix: str,
    chunk_size: int,
    scan_timeout_s: float = 8.0,
) -> dict[str, Any]:
    return asyncio.run(
        sync_wearable_ble(
            address=address,
            name_prefix=name_prefix,
            chunk_size=chunk_size,
            scan_timeout_s=scan_timeout_s,
        )
    )
