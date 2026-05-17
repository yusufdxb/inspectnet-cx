import subprocess
import sys

COMMANDS = (
    ("pytest", "-q"),
    ("ruff", "check", "src", "tests", "scripts"),
    (sys.executable, "-m", "build"),
)


def main() -> None:
    for command in COMMANDS:
        print("+", " ".join(command))
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
