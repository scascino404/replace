"""Placement instance stored as flat tensors.

A design is a set of rectangular *objects* (standard cells, macros, IO pads)
connected by *nets*. Every net is a set of *pins*; every pin sits on one object
at a fixed offset from the object's center. Positions always refer to object
centers.
"""

from dataclasses import dataclass, replace

import torch


@dataclass
class Design:
    name: str
    die: tuple[float, float, float, float]  # placement region (xl, yl, xh, yh)
    row_height: float
    size: torch.Tensor  # (n, 2) object width and height
    pos: torch.Tensor  # (n, 2) object centers (inputs only matter for fixed objects)
    fixed: torch.Tensor  # (n,) True for objects the placer may not move
    pin_obj: torch.Tensor  # (p,) object owning each pin
    pin_net: torch.Tensor  # (p,) net of each pin
    pin_offset: torch.Tensor  # (p, 2) pin location relative to its object's center
    num_nets: int

    @property
    def num_objects(self) -> int:
        return self.size.shape[0]

    @property
    def num_pins(self) -> int:
        return self.pin_obj.shape[0]

    @property
    def movable(self) -> torch.Tensor:
        return ~self.fixed

    @property
    def macro(self) -> torch.Tensor:
        """Objects taller than a standard-cell row."""
        return self.size[:, 1] > self.row_height * 1.5

    @property
    def die_width(self) -> float:
        return self.die[2] - self.die[0]

    @property
    def die_height(self) -> float:
        return self.die[3] - self.die[1]

    def pin_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """(p, 2) absolute pin positions for object centers `pos`."""
        return pos[self.pin_obj] + self.pin_offset

    def with_pos(self, pos: torch.Tensor) -> "Design":
        return replace(self, pos=pos)

    def summary(self) -> str:
        mov, mac = self.movable, self.macro
        return (
            f"{self.name}: {int((mov & ~mac).sum())} cells, "
            f"{int((mov & mac).sum())} movable / {int((self.fixed & mac).sum())} fixed macros, "
            f"{int((self.fixed & ~mac).sum())} pads, {self.num_nets} nets, {self.num_pins} pins, "
            f"die {self.die_width:.0f} x {self.die_height:.0f}"
        )
