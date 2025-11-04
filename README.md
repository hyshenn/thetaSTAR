# Theta* Pathfinding

This project is similar to the A* pathfinding algorithm, but it uses [Theta* pathfinding](https://en.wikipedia.org/wiki/Theta*) to find more direct and natural paths. It currently can handle: walking, jumping, and uneven terrain.

## How It Works

- Finds smooth paths from one point to another.
- Skips unnecessary nodes using line-of-sight checks.
- Handles walking on slabs and stepping up small heights.
- Jumps automatically when needed.

## Usage Examples

Move the player to a target position:

```python
from your_script import path_walk_to

target_position = (100, 65, 200)
path_walk_to(goal=target_position)
```
## Precompute a path and reuse it:
```py
from your_script import path_find, path_walk_to

start = (50, 64, 50)
goal = (100, 65, 200)
path = path_find(start, goal)
path_walk_to(path=path)
```
