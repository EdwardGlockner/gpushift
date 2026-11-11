"""Console entry point. `gpushift hook` runs before every agent shell command, so
it skips importing the full CLI (psutil, sqlite, ...)."""

import sys


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "hook":
        from gpushift.hook import main as hook_main

        sys.exit(hook_main(sys.argv[2] if len(sys.argv) > 2 else "pretooluse"))
    from gpushift.cli import main as cli_main

    cli_main()
