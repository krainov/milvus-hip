from __future__ import annotations

import unittest

from scripts.ci.amd_gpu.common import retry_operation


class RetryOperationTest(unittest.TestCase):
    def test_retries_matching_failure(self) -> None:
        calls = 0

        def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("node not match")
            return "ok"

        result, retries = retry_operation(
            operation,
            is_retryable=lambda error: "node not match" in str(error),
            attempts=3,
            delay_s=0,
        )

        self.assertEqual("ok", result)
        self.assertEqual(2, retries)

    def test_does_not_retry_other_failure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid schema"):
            retry_operation(
                lambda: (_ for _ in ()).throw(RuntimeError("invalid schema")),
                is_retryable=lambda error: "node not match" in str(error),
                attempts=3,
                delay_s=0,
            )


if __name__ == "__main__":
    unittest.main()
