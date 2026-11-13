from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from typer.core import TyperGroup, TyperOption

if TYPE_CHECKING:
    from typer._click.core import Context
    from typer._click.parser import _OptionParser

__all__ = [
    "InputPositionDirpaths",
    "OptionEatAll",
    "expand_position_dirpaths",
    "install_eat_all_positions",
]


def expand_position_dirpaths(patterns: list[str]) -> list[Path]:
    """Expand position paths, plate roots, and globs into position directory paths.

    Patterns are selected as [`open_positions`][iohub.ngff.open_positions]
    selects them: plate roots select all positions in the plate, file matches
    are skipped, and a position named twice is listed once.
    Raise ``typer.BadParameter`` if a pattern matches nothing,
    matches a directory that is not a position or plate, or if no position is found.
    """
    from iohub.ngff import open_positions
    from iohub.ngff.nodes import _store_dirpath

    try:
        with open_positions(patterns) as positions:
            return [_store_dirpath(position) for position in positions]
    except (FileNotFoundError, ValueError) as err:
        raise typer.BadParameter(str(err)) from err


class OptionEatAll(TyperOption):
    """Collect values after a list option until the next option.

    For example, ``-i a b c`` gives ``["a", "b", "c"]``. This also accepts
    paths expanded by the shell from an unquoted glob.
    """

    def add_to_parser(self, parser: _OptionParser, ctx: Context) -> None:
        super().add_to_parser(parser, ctx)
        for opt_name in self.opts:
            registered = parser._long_opt.get(opt_name) or parser._short_opt.get(opt_name)
            if registered is None:
                continue
            append_one = registered.process

            def eat_all(value, state, _append=append_one, _prefixes=registered.prefixes):
                _append(value, state)
                # Stop before the next option so the parser can handle it.
                while state.rargs and not any(state.rargs[0].startswith(p) for p in _prefixes):
                    _append(state.rargs.pop(0), state)

            registered.process = eat_all
            break


def install_eat_all_positions(group: TyperGroup) -> None:
    """Apply ``OptionEatAll`` to position options in a group's immediate commands.

    Call after ``typer.main.get_command``. Options are matched by the parameter
    name ``input_position_dirpaths``.
    """
    for command in group.commands.values():
        for param in command.params:
            if isinstance(param, TyperOption) and param.name == "input_position_dirpaths":
                param.__class__ = OptionEatAll


InputPositionDirpaths = Annotated[
    list[str],
    typer.Option(
        "--input-position-dirpaths",
        "-i",
        help=(
            "Position paths, plate roots, or globs. A plate root selects all "
            "positions. One -i accepts multiple paths, up to the next option."
        ),
    ),
]
