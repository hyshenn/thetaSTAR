# Theta* Pathfinding

This project is similar to the A* pathfinding algorithm, but it uses [Theta* pathfinding](https://en.wikipedia.org/wiki/Theta*) to find more direct and natural paths. It currently can handle: walking, jumping, and uneven terrain.

## How It Works

- Finds smooth paths from one point to another.
- Skips unnecessary nodes using line-of-sight checks.
- Handles walking on slabs and stepping up small heights.
- Jumps automatically when needed.
 
## path_walk_to parameters

goal – The target position to walk to. Can be (x, y, z). (Optional)
path – A precomputed path from path_find. If provided, the function will use it instead of computing a new path. (Optional)
distance – How close the player needs to be to a path point before moving to the next one. Lower values make the player follow the path more precisely. (Optional, default value: 1)
look_ahead – How many steps ahead of the current position the player considers for smoothing movement. Higher values create smoother motion but may react slower to obstacles. (Optional, default value: 1)

## path_find parameters

start – Starting position (x, y, z).  
goal – Target position (x, y, z) to pathfind.

## Usage Examples

Move the player to a target position:

```python
from your_script import path_walk_to

target_position = (100, 65, 200)
path_walk_to(goal=target_position)
```
Precompute a path and reuse it:
```py
from your_script import path_find, path_walk_to

start = (50, 64, 50)
goal = (100, 65, 200)
path = path_find(start, goal)
path_walk_to(path=path)
```
