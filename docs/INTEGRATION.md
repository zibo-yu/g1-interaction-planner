# Integration contract

The MuJoCo harness should remain generator-agnostic.

## Upstream contract

Future planner/generator code should produce one of:

1. complete MuJoCo `qpos[T, nq]`; or
2. root pose + named joint positions, which an adapter converts to `qpos`.

Never rely on column index coincidence between KiMoDo and MuJoCo.

## Downstream contract

This harness ends at visualization/reference validation.

The future tracker should receive a semantically identical whole-body reference,
but its own interface may include joint velocity, root velocity, phase variables,
contact flags, or future reference windows.

## Recommended next classes

```python
class WholeBodyGenerator:
    def generate(self, task_space_plan, robot_state) -> MotionReference:
        ...

class KimodoAdapter:
    def to_motion_clip(self, kimodo_output) -> MotionClip:
        ...
```

Keep `KimodoAdapter` separate from `playback.py`.
