"""Application entry points."""

from pkg import config as cfg
from pkg.config import load_config

from . import utils


class BaseService:
    def start(self):
        return self.ready()

    def ready(self):
        return True


class App(BaseService):
    """Main application."""

    name: str = "app"
    retries = 3

    def __init__(self, path):
        self.path = path
        self.config = None

    def initialize_app(self):
        self.config = load_config(self.path)
        cfg.validate(self.config, strict=True)
        self.start()
        return utils.helper(len(self.path))

    @staticmethod
    def build(path):
        return App(path)

    async def fetch(self, url: str, timeout: float = 1.0) -> bytes:  # noqa: ARG002
        def inner(x):
            return utils.helper(x)

        return inner(url)

    def restart(self):
        super().start()


def main():
    app = App.build("config.json")
    app.initialize_app()
    print(compute(nested(1)))


def compute(x):
    return x


def nested(y):
    return y


app_instance = App("x")
app_instance.initialize_app()
