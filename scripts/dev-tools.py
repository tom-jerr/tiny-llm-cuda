import argparse

import pytest


def test(args):
    if args.func:
        pytest.main(
            ["-v", f"tests/unittests/test_{args.func}.py", "-W", "ignore"] + args.remainders
        )
    else:
        pytest.main(["-v", "tests/unittests/"] + args.remainders)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--func", type=str, required=False)
    test_parser.add_argument("remainders", nargs="*")
    test_parser.set_defaults(test_parser=True)
    args = parser.parse_args()
    if hasattr(args, "test_parser"):
        test(args)


if __name__ == "__main__":
    main()
