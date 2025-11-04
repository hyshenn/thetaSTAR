from system.lib import minescript
import math, heapq, threading, time
from concurrent.futures import ThreadPoolExecutor
from rotation import look

AIRLIKE = {"minecraft:air", "minecraft:light"}
LIQUIDS = {"minecraft:water", "minecraft:lava"}
IGNORE_BLOCKS = {"air","water","lava","carpet","stairs","sign","flower","torch","ladder","vine","grass","snow","sapling"}
MAX_VERTICAL_STEP = 1
EYE_HEIGHT = 1.62
MAX_WORKERS = 25

def is_relevant(block_id: str) -> bool:
    if not block_id:
        return False
    bid = block_id.lower()
    if any(n in bid for n in IGNORE_BLOCKS):
        return False
    return "slab" in bid and "double" not in bid and ("bottom" in bid or "type=bottom" in bid or "_bottom" in bid)

def get_block(x, y, z, cache):
    key = (x, y, z)
    if key not in cache:
        cache[key] = minescript.getblock(x, y, z)
    return cache[key]

def LOS(start, end, cache) -> bool:
    x0, y0, z0 = start
    x1, y1, z1 = end
    y0 += EYE_HEIGHT
    dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
    steps = max(abs(dx), abs(dy), abs(dz))
    if steps == 0:
        return True
    dx /= steps
    dy /= steps
    dz /= steps
    for i in range(1, int(steps) + 1):
        xi, yi, zi = x0 + dx * i, y0 + dy * i, z0 + dz * i
        local_y = yi - math.floor(yi)
        bx, by, bz = int(math.floor(xi)), int(math.floor(yi)), int(math.floor(zi))
        block = get_block(bx, by, bz, cache)
        if not block or block.lower() in AIRLIKE:
            continue
        bid = block.lower()
        if bid in IGNORE_BLOCKS:
            continue
        if is_relevant(bid) and local_y > 0.8:
            continue
        return False
    return True


class Node:
    __slots__ = ("pos","parent","G","H")
    def __init__(self,pos, parent=None):
        self.pos = tuple(map(math.floor,pos))
        self.parent = parent
        self.G = 0
        self.H = 0

    @property
    def F(self):
        return self.G + 1.5*self.H

    def __lt__(self,other):
        return self.F < other.F

    def heuristic(self, goal):
        px, py, pz = self.pos
        gx, gy, gz = goal
        return abs(px-gx) + abs(py-gy) + abs(pz-gz)

    def is_walkable(self, pos, cache):
        nx,ny,nz = pos
        cx,cy,cz = self.pos
        dx,dy,dz = nx-cx, ny-cy, nz-cz
        if abs(dy) > MAX_VERTICAL_STEP:
            return False
        below = get_block(nx, ny-1, nz, cache)
        current = get_block(nx, ny, nz, cache)
        above = get_block(nx, ny+1, nz, cache)
        if below in AIRLIKE.union(LIQUIDS):
            return False
        if current not in AIRLIKE and not is_relevant(current):
            return False
        if above not in AIRLIKE:
            return False
        if abs(dx)+abs(dz) == 2:
            block_x = get_block(cx+dx, cy, cz, cache)
            block_z = get_block(cx, cy, cz+dz, cache)
            if block_x not in AIRLIKE and not is_relevant(block_x):
                return False
            if block_z not in AIRLIKE and not is_relevant(block_z):
                return False
        return True

    def neighbors(self, cache):
        x,y,z = self.pos
        candidates = [(x+dx, y+dy, z+dz)
            for dx in [-1,0,1]
            for dy in [-1,0,1] if abs(dy) <= MAX_VERTICAL_STEP
            for dz in [-1,0,1] if (dx,dy,dz)!=(0,0,0)
            ]
        result = []
        def check(p):
            return p if self.is_walkable(p, cache) else None
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as e:
            for r in e.map(check, candidates):
                if r:
                    result.append(r)
        return result

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

def path_find(start, goal):
    start_time = time.time()
    start = tuple(map(math.floor, start))
    goal = tuple(map(math.floor, goal))
    start_node = Node(start)
    start_node.H = start_node.heuristic(goal)
    open_heap = [(start_node.F, start_node)]
    closed = set()
    node_map = {start: start_node}
    cache = {}
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current.pos in closed:
            continue
        if current.pos == goal:
            dist = sum(abs(a-b) for a,b in zip(goal, start))
            minescript.echo(f"Pathfinding took {time.time()-start_time:.3f}s | Nodes: {len(reconstruct_path(current))} | Distance: {dist}")
            return reconstruct_path(current)
        closed.add(current.pos)
        for npos in current.neighbors(cache):
            if npos in closed:
                continue
            if current.parent and LOS(current.parent.pos, npos, cache):
                tentative_G = current.parent.G + math.sqrt(sum((a-b)**2 for a,b in zip(current.parent.pos, npos)))
                parent_candidate = current.parent
            else:
                tentative_G = current.G + math.sqrt(sum((a-b)**2 for a,b in zip(current.pos, npos)))
                parent_candidate = current
            neighbor = node_map.get(npos)
            if not neighbor:
                neighbor = Node(npos, parent_candidate)
                neighbor.G = tentative_G
                neighbor.H = neighbor.heuristic(goal)
                node_map[npos] = neighbor
                heapq.heappush(open_heap, (neighbor.F, neighbor))
            elif tentative_G < neighbor.G:
                neighbor.parent = parent_candidate
                neighbor.G = tentative_G
                heapq.heappush(open_heap, (neighbor.F, neighbor))
    raise ValueError("No path found")


def jump_loop(path_ref):
    last_jump_time = 0
    while True:
        time.sleep(0.01)
        now = time.time()
        if now - last_jump_time < 0.25 or not path_ref or not path_ref[0]:
            continue
        px, py, pz = map(float, minescript.player_position())
        foot_y = math.floor(py)
        nearest_index = min(
            range(len(path_ref[0])),
            key=lambda i: (px - (path_ref[0][i][0]+0.5))**2 + (pz - (path_ref[0][i][2]+0.5))**2
        )
        nxt = next((p for p in path_ref[0][nearest_index:] if p[1] > math.floor(py)), None)
        if not nxt:
            continue
        dx, dy, dz = px - nxt[0], py - nxt[1], pz - nxt[2]
        dy = nxt[1] - py
        if dx*dx + dy*dy + dz*dz <= 4 and dy > 0:
            block_below = minescript.getblock(math.floor(px), foot_y-1, math.floor(pz))
            if block_below not in AIRLIKE:
                minescript.player_press_jump(True)
                time.sleep(0.35)
                minescript.player_press_jump(False)
                last_jump_time = time.time()

def path_walk_to(goal=None, path=None, distance: float=1, look_ahead: int=1):
    if not getattr(path_walk_to, "_jump_running", False):
        path_ref = [None]
        path_walk_to._path_ref = path_ref
        threading.Thread(target=jump_loop, args=(path_ref,), daemon=True).start()
        path_walk_to._jump_running = True
    else:
        path_ref = path_walk_to._path_ref
    if path is None:
        if goal is None:
            return
        path = path_find(tuple(map(float, minescript.player_position())), goal)
    path_ref[0] = path
    current_index = 0
    forward_velocity = strafe_velocity = 0.0
    ACCEL = 0.16
    MIN_THRESHOLD = 0.01

    while current_index < len(path):
        px, py, pz = map(float, minescript.player_position())
        lx = lz = ly = 0.0
        count = 0
        for i in range(current_index, min(current_index + look_ahead, len(path))):
            x, y, z = path[i]
            lx += x + 0.5
            ly += y
            lz += z + 0.5
            count += 1
        lx /= count
        ly /= count
        lz /= count
        cx, cy, cz = path[current_index]
        cx += 0.5
        cz += 0.5
        cy -= 1
        dx, dy, dz = cx - px, cy - (py - 1), cz - pz
        if dx*dx + dy*dy + dz*dz <= distance**2:
            current_index += 1
            continue
        dx_look, dy_look, dz_look = lx - px, ly - py, lz - pz
        yaw = math.degrees(math.atan2(dz_look, dx_look)) - 90
        pitch = -math.degrees(math.atan2(dy_look, math.sqrt(dx_look**2 + dz_look**2)))
        look(yaw, pitch)
        yaw_rad = math.radians(yaw + 90)
        pitch_rad = math.radians(pitch)
        forward_dir = (math.cos(pitch_rad)*math.cos(yaw_rad), -math.sin(pitch_rad), math.cos(pitch_rad)*math.sin(yaw_rad))
        right_dir = (math.sin(yaw_rad), 0, -math.cos(yaw_rad))
        length = math.sqrt(dx_look**2 + dy_look**2 + dz_look**2)
        if length == 0:
            continue
        move_vec = (dx_look/length, dy_look/length, dz_look/length)
        forward_target = sum(f*m for f, m in zip(forward_dir, move_vec))
        strafe_target = sum(r*m for r, m in zip(right_dir, move_vec))
        forward_velocity += (forward_target - forward_velocity) * ACCEL
        strafe_velocity += (strafe_target - strafe_velocity) * ACCEL
        minescript.player_press_forward(forward_velocity > MIN_THRESHOLD)
        minescript.player_press_backward(forward_velocity < -MIN_THRESHOLD)
        minescript.player_press_left(strafe_velocity < -MIN_THRESHOLD)
        minescript.player_press_right(strafe_velocity > MIN_THRESHOLD)
    minescript.player_press_forward(False)
    minescript.player_press_backward(False)
    minescript.player_press_left(False)
    minescript.player_press_right(False)


