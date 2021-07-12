"""Entry point to program."""

import sys
from driver import Driver


def main(args):
    """Main func."""
    driver = Driver()
    driver.run(args)


if __name__ == "__main__":
    main(sys.argv[1:])