from .cli import Bioimageio


def main():
    cli = Bioimageio()  # pyright: ignore[reportCallIssue]
    cli.run()


if __name__ == "__main__":
    main()
