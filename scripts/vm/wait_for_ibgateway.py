from __future__ import annotations

import argparse
import socket
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait until the IB Gateway socket is reachable.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4002)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    deadline = time.time() + int(args.timeout)
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect((args.host, args.port))
            sock.close()
            return 0
        except OSError:
            time.sleep(2.0)
        finally:
            sock.close()
    print(f"Timed out waiting for IB Gateway at {args.host}:{args.port}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
