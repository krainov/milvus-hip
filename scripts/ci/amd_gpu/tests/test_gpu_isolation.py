from __future__ import annotations

import signal
import unittest
from unittest import mock

from scripts.ci.amd_gpu.gpu_isolation import (
    container_id_for_pid,
    parse_amd_processes,
    parse_nvidia_processes,
    stop_processes,
)


class GpuIsolationTest(unittest.TestCase):
    def test_parses_nvidia_compute_processes(self) -> None:
        self.assertEqual(
            [{"pid": 123, "name": "milvus", "memory_mib": "4160"}],
            parse_nvidia_processes("123, milvus, 4160\n"),
        )

    def test_parses_amd_compute_processes(self) -> None:
        self.assertEqual(
            [{"pid": 456, "name": "milvus", "memory_bytes": "5844430848"}],
            parse_amd_processes(
                '{"system":{"PID456":"milvus, 1, 5844430848, 0, unknown"}}'
            ),
        )

    def test_finds_docker_container_from_systemd_cgroup(self) -> None:
        container_id = "a" * 64
        with mock.patch(
            "scripts.ci.amd_gpu.gpu_isolation.Path.read_text",
            return_value=f"0::/system.slice/docker-{container_id}.scope\n",
        ):
            self.assertEqual(container_id, container_id_for_pid(123))

    def test_invalid_amd_output_has_no_processes(self) -> None:
        self.assertEqual([], parse_amd_processes("not json"))

    def test_stops_owning_docker_container_without_removing_it(self) -> None:
        container_id = "b" * 64
        with (
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.protected_pids",
                return_value=set(),
            ),
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.container_id_for_pid",
                return_value=container_id,
            ),
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.run_command",
                return_value={"returncode": 0, "stdout": container_id, "stderr": ""},
            ) as run_command,
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.wait_for_gpu_idle",
                return_value=[],
            ),
        ):
            actions = stop_processes(
                "nvidia", [{"pid": 123, "name": "milvus"}], 30
            )

        run_command.assert_called_once_with(
            ["docker", "stop", "--time", "30", container_id],
            timeout=45,
        )
        self.assertEqual("docker-stop", actions[0]["action"])

    def test_terminates_host_process_without_touching_files(self) -> None:
        with (
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.protected_pids",
                return_value=set(),
            ),
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.container_id_for_pid",
                return_value=None,
            ),
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.wait_for_gpu_idle",
                return_value=[],
            ),
            mock.patch(
                "scripts.ci.amd_gpu.gpu_isolation.os.kill"
            ) as kill,
        ):
            actions = stop_processes(
                "amd", [{"pid": 456, "name": "milvus"}], 30
            )

        kill.assert_called_once_with(456, signal.SIGTERM)
        self.assertEqual("sigterm", actions[0]["action"])


if __name__ == "__main__":
    unittest.main()
