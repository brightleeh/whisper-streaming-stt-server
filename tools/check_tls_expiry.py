#!/usr/bin/env python3
import argparse
import datetime as dt
import ssl
import sys


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Fail if a TLS certificate expires within N days."
    )
    parser.add_argument("cert_path", help="Path to PEM-encoded certificate")
    parser.add_argument(
        "--warn-days",
        type=int,
        default=14,
        help="Fail if certificate expires within this many days (default: 14)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _ssl = getattr(ssl, "_ssl", None)
    if _ssl is None:
        print("SSL backend not available for certificate decoding.", file=sys.stderr)
        return 2
    decode_cert = getattr(_ssl, "_test_decode_cert", None)
    if decode_cert is None:
        print(
            "Certificate decoding is not supported by this SSL backend.",
            file=sys.stderr,
        )
        return 2
    info = decode_cert(args.cert_path)
    not_after = info.get("notAfter")
    if not not_after:
        print("Unable to read certificate expiry", file=sys.stderr)
        return 2
    expiry = dt.datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
    remaining = expiry - dt.datetime.utcnow()
    days_left = remaining.days
    print(f"TLS certificate expires on {expiry.isoformat()} ({days_left} days left)")
    if days_left <= args.warn_days:
        print(
            f"TLS certificate expires within {args.warn_days} days.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
