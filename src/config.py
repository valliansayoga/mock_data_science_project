import tomli
from pathlib import Path
from rich.table import Table
from rich.console import Console


root_dir = Path(__file__).resolve().parents[1]
filepath = root_dir / "config.toml"


class dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = dotdict() or d = dotdict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = dotdict(value)
            self[key] = value


class Config(dotdict):
    def __init__(self, filepath=filepath):
        with open(filepath, "rb") as file:
            data = tomli.load(file)
        super().__init__(data)

        # if getattr(getattr(self, "paths"), "db"):

        try:
            db_uri = (Path(self.paths.pythonpath) / Path(self.paths.db)).as_posix()
            self.general.db_uri = self.general.db_uri.replace("...", db_uri)
        except KeyError:
            print("No database URI found in toml file")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        table = Table(title="Config")
        table.add_column("Key", style="white bold")
        table.add_column("Value", style="white bold")
        for key, value in self.items():
            table.add_row(key, f"{value!r}", style="white")
        console = Console()
        console.print(table)
        return ""


def dotdict_test():
    test_case = {
        "a": "nice",
        "b": "very nice",
        "c": {"d": "quite nice"},
    }

    dd_test = dotdict(test_case)

    assert dd_test.a == test_case["a"], "Value is not the same!"
    assert dd_test.b == test_case["b"], "Value is not the same!"

    try:
        assert dd_test.c.d == test_case["c"]["d"], "Value is not the same!"
    except AttributeError:
        raise AttributeError("Recursive apply failed.")


def get_config():
    """Returns a new instance of Config"""
    return Config()


if __name__ == "__main__":
    dotdict_test()
    cfg = Config()
    repr(cfg)
