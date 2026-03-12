from __future__ import annotations

from dataclasses import dataclass
import socket
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from calibration.transforms import pose_xyzrotvec_to_matrix

try:
    import rtde.rtde as ur_rtde
except Exception as exc:  # pragma: no cover - optional dependency
    ur_rtde = None
    UR_RTDE_IMPORT_ERROR = exc
else:
    UR_RTDE_IMPORT_ERROR = None


UR_RTDE_DEFAULT_PORT = 30004
UR_RTDE_OUTPUT_NAMES = ["timestamp", "actual_TCP_pose"]
UR_RTDE_OUTPUT_TYPES = ["DOUBLE", "VECTOR6D"]

UR_KNOWN_PORTS: Dict[int, str] = {
    29999: "Dashboard Server",
    30001: "Primary Client",
    30002: "Secondary Client",
    30003: "Realtime Client",
    30004: "RTDE",
    30011: "Primary Client (read-only)",
    30012: "Secondary Client (read-only)",
    30013: "Realtime Client (read-only)",
    30020: "Interpreter Mode",
}


@dataclass(frozen=True)
class URPortStatus:
    port: int
    name: str
    is_open: bool


@dataclass(frozen=True)
class URTCPPoseSample:
    timestamp_s: Optional[float]
    pose_vector: np.ndarray
    transform: np.ndarray


class URRTDEClient:
    """UR RTDE wrapper that follows the official Python client library workflow."""

    def __init__(self) -> None:
        self._client: Optional[Any] = None
        self._host: str = ""
        self._port: int = UR_RTDE_DEFAULT_PORT
        self._frequency_hz: float = 10.0
        self._controller_version: Optional[Tuple[int, ...]] = None

    @staticmethod
    def is_sdk_available() -> bool:
        return ur_rtde is not None

    @staticmethod
    def sdk_error_message() -> str:
        if UR_RTDE_IMPORT_ERROR is None:
            return ""
        return str(UR_RTDE_IMPORT_ERROR)

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def controller_version(self) -> Optional[Tuple[int, ...]]:
        return self._controller_version

    @property
    def is_connected(self) -> bool:
        return self._client is not None and bool(self._client.is_connected())

    def scan_known_ports(
        self,
        host: str,
        timeout_s: float = 0.25,
        ports: Optional[Sequence[int]] = None,
    ) -> List[URPortStatus]:
        statuses: List[URPortStatus] = []
        for port in ports if ports is not None else UR_KNOWN_PORTS.keys():
            is_open = False
            try:
                with socket.create_connection((host, int(port)), timeout=timeout_s):
                    is_open = True
            except OSError:
                is_open = False
            statuses.append(URPortStatus(port=int(port), name=UR_KNOWN_PORTS.get(int(port), "Unknown"), is_open=is_open))
        return statuses

    def connect(self, host: str, port: int = UR_RTDE_DEFAULT_PORT, frequency_hz: float = 10.0) -> Tuple[int, ...]:
        if ur_rtde is None:
            raise RuntimeError(
                "未安装 UR 官方 RTDE Python Client Library。"
                f" 当前导入错误: {self.sdk_error_message()}"
            )
        if int(port) != UR_RTDE_DEFAULT_PORT:
            raise ValueError(
                "当前实现基于 UR 官方 RTDE Python Client Library，读取 TCP 位姿仅支持 RTDE 端口 30004。"
            )

        self.disconnect()

        client = ur_rtde.RTDE(host, int(port))
        try:
            client.connect()
            controller_version = tuple(int(value) for value in client.get_controller_version())
            if not client.send_output_setup(
                UR_RTDE_OUTPUT_NAMES,
                UR_RTDE_OUTPUT_TYPES,
                frequency=float(frequency_hz),
            ):
                raise RuntimeError("RTDE 输出配置失败，无法订阅 actual_TCP_pose。")
            if not client.send_start():
                raise RuntimeError("RTDE 数据同步启动失败。")
        except Exception:
            try:
                client.disconnect()
            except Exception:
                pass
            raise

        self._client = client
        self._host = str(host)
        self._port = int(port)
        self._frequency_hz = float(frequency_hz)
        self._controller_version = controller_version
        return controller_version

    def disconnect(self) -> None:
        if self._client is None:
            return
        try:
            if self._client.is_connected():
                try:
                    self._client.send_pause()
                except Exception:
                    pass
        finally:
            try:
                self._client.disconnect()
            finally:
                self._client = None
                self._controller_version = None

    def read_tcp_pose(self, blocking: bool = True) -> Optional[URTCPPoseSample]:
        if not self.is_connected or self._client is None:
            raise RuntimeError("UR RTDE 尚未连接。")

        try:
            if blocking or not hasattr(self._client, "receive_buffered"):
                state = self._client.receive()
            else:
                state = self._client.receive_buffered()
        except Exception as exc:
            self.disconnect()
            raise RuntimeError(f"RTDE 读取失败: {exc}") from exc

        if state is None:
            return None

        pose_vector = getattr(state, "actual_TCP_pose", None)
        if pose_vector is None:
            raise RuntimeError("RTDE 数据包中缺少 actual_TCP_pose 字段。")

        pose = np.asarray(pose_vector, dtype=np.float64).reshape(6)
        transform = pose_xyzrotvec_to_matrix(*pose.tolist())
        timestamp = getattr(state, "timestamp", None)
        return URTCPPoseSample(
            timestamp_s=float(timestamp) if timestamp is not None else None,
            pose_vector=pose,
            transform=transform,
        )

    @staticmethod
    def describe_open_ports(port_statuses: Iterable[URPortStatus]) -> str:
        open_ports = [f"{item.port}({item.name})" for item in port_statuses if item.is_open]
        return ", ".join(open_ports) if open_ports else "未发现开放的 UR 常用端口"
