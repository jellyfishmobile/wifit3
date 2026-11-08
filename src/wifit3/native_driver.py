"""Built-in Wi-Fi driver: monitors the phone's own Wi-Fi through a monitor interface plus an
AF_PACKET socket, instead of a USB card driven over bulk endpoints.

The USB-only design (PyUSB + per-chip register mini-drivers) needs an externally attached dongle.
On phones the Wi-Fi radio is built into the SoC's PCIe block and already driven by the platform's
kernel driver; there is no USB device to claim. This driver keeps the same ``Driver`` contract the
array/sink/campaign layers expect and maps it onto the ordinary Linux path:

* ``connect``       -> create ``mon0`` with ``iw`` and open an AF_PACKET socket on it
* ``set_channel``   -> ``iw dev mon0 set channel N``
* ``inject_frame``  -> ``send()`` on the same socket
* ``close``         -> drop the reader, close the socket, delete ``mon0``

Frames arrive with their radiotap prefix; the header is split off here and its antenna-signal
field feeds the ``rssi`` argument of the shared parser, mirroring what the chip mini-drivers do
with their per-chip RX descriptors.
"""
from __future__ import annotations

import asyncio
import logging
import socket
import struct
import subprocess
from typing import Callable, ClassVar, List, Optional

import usb.core

from wifit3.chips.driver import Driver, FakeMacSupport
from wifit3.dot11.parser import WlanFrameParser
from wifit3.models.device_id import DeviceID

logger = logging.getLogger(__name__)

MON_IFACE = "mon0"

# Radiotap present-bit -> field size, in the fixed order of the header (only fields up to the
# antenna signal are walked; everything after it is irrelevant for the scanner).
_RT_FIELDS: ClassVar = ((8, 1, 1, 2, 2, 1))


def _iw(*args: str, timeout: float = 10.0) -> str:
    """Run ``iw`` and return stdout; empty string on failure (caller treats it as 'unknown')."""
    try:
        proc = subprocess.run(("iw", *args), capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug("iw %s failed: %s", " ".join(args), e)
        return ""
    return proc.stdout.decode(errors="replace")


def _split_radiotap(buf: bytes) -> tuple[bytes, Optional[int]]:
    """``(mpdu, rssi)`` from a socket read: strip the radiotap prefix when present.

    Without a radiotap prefix (some firmwares hand the plain MPDU to AF_PACKET) the buffer is
    returned unchanged with ``rssi=None`` so the caller can fall back to the parser default."""
    if len(buf) >= 8 and buf[0] == 0 and buf[1] == 0:
        hdr_len = struct.unpack_from("<H", buf, 2)[0]
        if 8 <= hdr_len <= len(buf):
            rssi = _radiotap_rssi(buf[:hdr_len])
            return buf[hdr_len:], rssi
    return buf, None


def _radiotap_rssi(hdr: bytes) -> Optional[int]:
    """Walk the present bitmask to the Antenna Signal field (bit 5); None when absent."""
    present = struct.unpack_from("<I", hdr, 4)[0]
    off = 8
    for bit, size in enumerate(_RT_FIELDS):
        if present & (1 << bit):
            if bit == 5 and off + size <= len(hdr):
                return hdr[off] if hdr[off] < 128 else hdr[off] - 256
            off += size
    return None


class NativeDriver(Driver):
    """The ``Driver`` contract over the platform-managed interface; no USB round-trips."""

    SUPPORTED_CHANNELS: ClassVar[List[int]] = Driver.CHANNELS_2G + Driver.CHANNELS_5G
    FAKE_MAC: ClassVar[FakeMacSupport] = FakeMacSupport.NONE

    def __init__(self, phy: str = "phy0", parent: str = "wlan0") -> None:
        super().__init__()
        self.phy = phy
        self.parent = parent
        self.is_warm = False
        self.mac_address = None
        self._rx_callback: Optional[Callable[..., None]] = None
        self._disconnect_callback: Optional[Callable[[], None]] = None
        self._sock: Optional[socket.socket] = None
        self._seq = 0

    @classmethod
    def from_usb_device(cls, dev: usb.core.Device, id_entry: DeviceID) -> "NativeDriver":
        """Not used for the built-in radio (there is no USB handle to adopt); kept for the ABC."""
        extras = id_entry.extras or {}
        return cls(phy=extras.get("phy", "phy0"), parent=extras.get("parent", "wlan0"))

    def register_rx_callback(self, cb: Callable[..., None]) -> None:
        self._rx_callback = cb

    def register_disconnect_callback(self, cb: Callable[[], None]) -> None:
        self._disconnect_callback = cb

    async def connect(self, progress_cb: Optional[Callable[[float, str], None]] = None) -> bool:
        loop = asyncio.get_running_loop()
        if progress_cb:
            progress_cb(0.2, "creating monitor interface")
        await loop.run_in_executor(None, self._ensure_monitor_iface)
        try:
            self.mac_address = await loop.run_in_executor(None, self._read_mac)
        except OSError:
            pass
        if progress_cb:
            progress_cb(0.6, "opening capture socket")
        sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003))
        sock.bind((MON_IFACE,))
        sock.setblocking(False)
        self._sock = sock
        loop.add_reader(sock.fileno(), self._on_readable)
        self.is_warm = True
        logger.info("native driver up: %s via %s (mac %s)", MON_IFACE, self.phy, self.mac_address)
        return True

    def _ensure_monitor_iface(self) -> None:
        if MON_IFACE in _iw("dev"):
            return
        _iw("phy", self.phy, "interface", "add", MON_IFACE, "type", "monitor")

    def _read_mac(self) -> Optional[str]:
        try:
            with open(f"/sys/class/net/{self.parent}/address") as f:
                return f.read().strip().lower()
        except OSError:
            return None

    async def set_channel(self, channel: int, scan: bool = False) -> bool:
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: _iw(
            "dev", MON_IFACE, "set", "channel", str(channel)))
        return out != "" or MON_IFACE in _iw("dev")

    def _on_readable(self) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            while True:
                try:
                    buf = sock.recv(65536)
                except BlockingIOError:
                    return
                except OSError as e:
                    self._lost(e)
                    return
                if buf:
                    self._dispatch(buf)
        except Exception as e:  # a dead socket must surface once, then stay quiet
            self._lost(e)

    def _lost(self, exc: Exception) -> None:
        logger.error("native driver lost the capture socket: %s", exc)
        if self._disconnect_callback:
            try:
                self._disconnect_callback()
            except Exception:
                logger.exception("disconnect callback raised")

    def _dispatch(self, buf: bytes) -> None:
        mpdu, rssi = _split_radiotap(buf)
        if len(mpdu) == 10 and mpdu[0] == 0xD4:
            self.record_ack(mpdu)
            return
        cb = self._rx_callback
        if cb is None:
            return
        parsed = WlanFrameParser.parse_80211_frame(mpdu, rssi if rssi is not None else -100)
        if parsed:
            try:
                cb(parsed)
            except Exception:
                logger.exception("RX callback raised")

    async def _inject_frame(self, frame_bytes: bytes) -> bool:
        if self._sock is None:
            return False
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._sock.send, frame_bytes)
        except OSError:
            return False
        return True

    def _stamp_tx_seq(self, frame_bytes: bytes) -> bytes:
        if len(frame_bytes) < 26:
            return frame_bytes
        self._seq = (self._seq + 1) & 0xFFF
        seq = struct.pack("<H", self._seq << 4)
        return frame_bytes[:24] + seq + frame_bytes[26:]

    async def _enable_rx_acks(self) -> None:
        """No-op: the monitor tap already admits ACK control frames to the socket."""

    async def _disable_rx_acks(self) -> None:
        """No-op inverse of ``_enable_rx_acks``."""

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                loop.remove_reader(sock.fileno())
            except Exception:
                pass
            sock.close()
        await loop.run_in_executor(None, lambda: _iw("dev", MON_IFACE, "del"))
        logger.info("native driver closed: %s released", MON_IFACE)
