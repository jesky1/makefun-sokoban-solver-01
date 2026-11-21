// Push-level A* Sokoban solver with room decomposition.
// Search is over pushes, not moves: a state is the sorted box squares plus the
// canonical player square (the lowest-index square of the region the player can
// reach without moving a box). The heuristic is a minimum-cost matching
// (Hungarian) between boxes and goals over exact lone-box push distances
// (reverse push-BFS per goal), restricted to pairs inside one floor region:
// the floor graph's articulation points are the room doors and corridors, a
// box never changes region in these levels, and the restriction turns the
// bound into a sum of per-room bounds. When the restriction admits no matching
// for the initial state it is dropped. Pruning: static dead squares (no
// lone-box push path to any goal), the 2x2 block rule, and an iterative freeze
// check. Budgets are expansion counts only (never the clock) and iteration
// orders are fixed, so runs are deterministic.
//
// Levels from this generator are a hallway plus rooms hanging off it through
// single doors, and every box stays in its room, so the whole-level search is
// a product of independent room searches. solveDecomposed therefore splits
// the level at the doors (block tree of components and articulation squares),
// solves each room on its own play area with the same push A*, and splices
// the push chains; the walk between pushes is reconstructed on the full
// level. A room's solution is accepted only when the player can still reach
// the room's entry square afterwards (otherwise it is re-solved with that as
// part of the goal test). Fallback ladder per instance: room decomposition,
// optimal whole-level push A*, weighted push A*, move-level A*, empty answer.
// Every instance always gets a solution file.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Fixed direction order: up, down, left, right.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LET = ['u', 'd', 'l', 'r'];
const INF = 1e9;

const CAP_OPT = 500000;
const CAP_WEIGHTED = 500000;
const CAP_MOVE = 400000;
const CAP_ROOM = 1000000;

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const n = w * h;
  const wall = new Uint8Array(n).fill(1);
  const goal = new Uint8Array(n);
  const goals = [];
  let boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') { goal[i] = 1; goals.push(i); }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  // Neighbour table; -1 = off board.
  const nbr = new Int32Array(n * 4).fill(-1);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = y * w + x;
    for (let d = 0; d < 4; d++) {
      const nx = x + DX[d], ny = y + DY[d];
      if (nx >= 0 && nx < w && ny >= 0 && ny < h) nbr[i * 4 + d] = ny * w + nx;
    }
  }
  return { w, h, n, wall, goal, goals, boxes, player, nbr };
}

// Per-goal lone-box push distances: reverse BFS in push space from the goal.
// dist[g][s] = min pushes to bring a lone box from s to goal g (the player may
// stand on any non-wall square), or -1 when impossible.
function pushDistances(L) {
  const { n, wall, goals, nbr } = L;
  const perGoal = [];
  for (const g of goals) {
    const dist = new Int32Array(n).fill(-1);
    dist[g] = 0;
    const queue = [g];
    let head = 0;
    while (head < queue.length) {
      const b = queue[head++];
      // A forward push in direction d moves a box from s to b where s = b - d
      // and the player stood at s - d. Reverse: predecessors s with landing b.
      for (let d = 0; d < 4; d++) {
        const s = nbr[b * 4 + (d ^ 1)]; // square opposite the push direction
        if (s < 0 || wall[s] || dist[s] >= 0) continue;
        const stand = nbr[s * 4 + (d ^ 1)];
        if (stand < 0 || wall[stand]) continue;
        dist[s] = dist[b] + 1;
        queue.push(s);
      }
    }
    perGoal.push(dist);
  }
  const minDist = new Int32Array(n).fill(-1);
  for (let s = 0; s < n; s++) {
    let m = -1;
    for (const dist of perGoal) {
      const v = dist[s];
      if (v >= 0 && (m < 0 || v < m)) m = v;
    }
    minDist[s] = m;
  }
  return { perGoal, minDist };
}

// Floor analysis: articulation points (Tarjan), the components left when
// they are removed, and regions — components merged across short articulation
// chains when one side holds goals but no boxes (or boxes but no goals).
// A room's goal alcove can hide behind a chain of up to 3 articulation
// squares, while two different rooms are always separated by a door, at
// least one hallway square and another door (and no two doors share a
// column), which takes at least 4 — so 3 never merges two rooms.
// Articulation squares themselves (doors, hallway, corridors) get region -1:
// a box or goal there is a wildcard for the matching restriction.
function analyze(L) {
  const { n, wall, nbr } = L;
  const disc = new Int32Array(n).fill(-1);
  const low = new Int32Array(n);
  const isArt = new Uint8Array(n);
  let time = 0;
  for (let root = 0; root < n; root++) {
    if (wall[root] || disc[root] >= 0) continue;
    // Iterative DFS; a frame is [node, parent, nextDir, treeChildren].
    const stack = [[root, -1, 0, 0]];
    disc[root] = low[root] = time++;
    while (stack.length) {
      const fr = stack[stack.length - 1];
      const v = fr[0], parent = fr[1];
      if (fr[2] < 4) {
        const d = fr[2]++;
        const to = nbr[v * 4 + d];
        if (to < 0 || wall[to] || to === parent) continue;
        if (disc[to] >= 0) { if (disc[to] < low[v]) low[v] = disc[to]; continue; }
        disc[to] = low[to] = time++;
        fr[3]++;
        stack.push([to, v, 0, 0]);
      } else {
        stack.pop();
        if (parent >= 0) {
          if (low[v] < low[parent]) low[parent] = low[v];
          if (low[v] >= disc[parent]) isArt[parent] = 1;
        } else if (fr[3] > 1) {
          isArt[v] = 1;
        }
      }
    }
  }
  const raw = new Int32Array(n).fill(-1);
  let nc = 0;
  const queue = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    if (wall[s] || isArt[s] || raw[s] >= 0) continue;
    let head = 0, tail = 0;
    queue[tail++] = s;
    raw[s] = nc;
    while (head < tail) {
      const c = queue[head++];
      for (let d = 0; d < 4; d++) {
        const q = nbr[c * 4 + d];
        if (q < 0 || wall[q] || isArt[q] || raw[q] >= 0) continue;
        raw[q] = nc;
        queue[tail++] = q;
      }
    }
    nc++;
  }
  const region = new Int32Array(n).fill(-1);
  if (nc === 0) return { region, raw, isArt, nc };
  // Union-find merging of needy components into adjacent givers.
  const parentU = new Int32Array(nc);
  for (let i = 0; i < nc; i++) parentU[i] = i;
  const find = (x) => {
    while (parentU[x] !== x) { parentU[x] = parentU[parentU[x]]; x = parentU[x]; }
    return x;
  };
  const boxCnt = new Int32Array(nc);
  const goalCnt = new Int32Array(nc);
  for (const b of L.boxes) if (raw[b] >= 0) boxCnt[raw[b]]++;
  for (const g of L.goals) if (raw[g] >= 0) goalCnt[raw[g]]++;
  // Adjacency: component pairs joined by an articulation path of at most 3
  // articulation squares (bipartite BFS over component/articulation nodes).
  const artIndex = new Int32Array(n).fill(-1);
  const artList = [];
  for (let s = 0; s < n; s++) if (isArt[s]) { artIndex[s] = artList.length; artList.push(s); }
  const compArts = Array.from({ length: nc }, () => []);
  const artComps = Array.from({ length: artList.length }, () => []);
  for (let ai = 0; ai < artList.length; ai++) {
    const s = artList[ai];
    for (let d = 0; d < 4; d++) {
      const q = nbr[s * 4 + d];
      if (q >= 0 && raw[q] >= 0 && !artComps[ai].includes(raw[q])) {
        artComps[ai].push(raw[q]);
        compArts[raw[q]].push(ai);
      }
    }
  }
  const artArts = Array.from({ length: artList.length }, () => []);
  for (let ai = 0; ai < artList.length; ai++) {
    const s = artList[ai];
    for (let d = 0; d < 4; d++) {
      const q = nbr[s * 4 + d];
      if (q >= 0 && artIndex[q] >= 0 && !artArts[ai].includes(artIndex[q])) artArts[ai].push(artIndex[q]);
    }
  }
  const adj = new Map();
  const addAdj = (a, b) => {
    if (a === b) return;
    const k = a < b ? a * nc + b : b * nc + a;
    adj.set(k, true);
  };
  for (let c = 0; c < nc; c++) {
    const seenArt = new Set();
    let frontier = compArts[c].slice();
    for (const ai of frontier) seenArt.add(ai);
    for (let depth = 1; depth <= 3; depth++) {
      for (const ai of frontier) for (const oc of artComps[ai]) addAdj(c, oc);
      if (depth === 3) break;
      const next = [];
      for (const ai of frontier) for (const aj of artArts[ai]) {
        if (!seenArt.has(aj)) { seenArt.add(aj); next.push(aj); }
      }
      frontier = next;
    }
  }
  const adjList = [...adj.keys()].map((k) => [(k / nc) | 0, k % nc]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let changed = true;
  while (changed) {
    changed = false;
    const rb = new Int32Array(nc), rg = new Int32Array(nc);
    for (let c = 0; c < nc; c++) { rb[find(c)] += boxCnt[c]; rg[find(c)] += goalCnt[c]; }
    for (const [a, b] of adjList) {
      const ra = find(a), rb2 = find(b);
      if (ra === rb2) continue;
      const needy = (r) => (rb[r] === 0 && rg[r] > 0) || (rg[r] === 0 && rb[r] > 0);
      const gives = (r, other) => (rb[other] === 0 && rg[other] > 0 && rb[r] > 0) || (rg[other] === 0 && rb[other] > 0 && rg[r] > 0);
      if ((needy(ra) && gives(rb2, ra)) || (needy(rb2) && gives(ra, rb2))) {
        parentU[rb2] = ra;
        changed = true;
      }
    }
  }
  for (let s = 0; s < n; s++) if (raw[s] >= 0) region[s] = find(raw[s]);
  return { region, raw, isArt, nc, artIndex, compArts, artComps };
}

// Hungarian algorithm for the assignment problem, O(n^3), on a flat cost
// array with INF for impossible assignments. Returns the minimum total cost,
// or INF when no perfect matching exists. Scratch buffers are allocated once
// by the factory so the per-call cost is pure arithmetic.
function makeHungarian(n) {
  const u = new Float64Array(n + 1);
  const v = new Float64Array(n + 1);
  const p = new Int32Array(n + 1); // p[j] = worker assigned to job j
  const way = new Int32Array(n + 1);
  const minv = new Float64Array(n + 1);
  const used = new Uint8Array(n + 1);
  return (cost) => {
    u.fill(0); v.fill(0); p.fill(0);
    for (let i = 1; i <= n; i++) {
      p[0] = i;
      let j0 = 0;
      minv.fill(Infinity);
      used.fill(0);
      do {
        used[j0] = 1;
        const i0 = p[j0];
        let delta = Infinity, j1 = 0;
        for (let j = 1; j <= n; j++) {
          if (used[j]) continue;
          const cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (let j = 0; j <= n; j++) {
          if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
          else minv[j] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do {
        const j1 = way[j0];
        p[j0] = p[j1];
        j0 = j1;
      } while (j0 !== 0);
    }
    let total = 0;
    for (let j = 1; j <= n; j++) {
      if (p[j] === 0) return INF;
      const c = cost[(p[j] - 1) * n + (j - 1)];
      if (c >= INF) return INF;
      total += c;
    }
    return total;
  };
}

// Rebuild the full move string for a push chain (packed boxFrom * 4 + dir)
// on a level, BFS-walking the player to each stand square.
function reconstructMoves(L, chain) {
  const { wall, nbr } = L;
  const boxes = L.boxes.slice();
  const boxSet = new Set(boxes);
  let player = L.player;
  let moves = '';
  const parent = new Int32Array(L.n);
  const pdir = new Int32Array(L.n);
  const seen = new Int32Array(L.n);
  const queue = new Int32Array(L.n);
  let seenStamp = 0;
  for (const packed of chain) {
    const boxFrom = (packed / 4) | 0;
    const d = packed % 4;
    const stand = nbr[boxFrom * 4 + (d ^ 1)];
    seenStamp++;
    let head = 0, tail = 0;
    queue[tail++] = player;
    seen[player] = seenStamp;
    parent[player] = -1;
    while (head < tail) {
      const c = queue[head++];
      if (c === stand) break;
      for (let dd = 0; dd < 4; dd++) {
        const q = nbr[c * 4 + dd];
        if (q < 0 || wall[q] || boxSet.has(q) || seen[q] === seenStamp) continue;
        seen[q] = seenStamp;
        parent[q] = c;
        pdir[q] = dd;
        queue[tail++] = q;
      }
    }
    let walk = '';
    for (let c = stand; parent[c] >= 0; c = parent[c]) walk = LET[pdir[c]] + walk;
    moves += walk + LET[d];
    const land = nbr[boxFrom * 4 + d];
    boxSet.delete(boxFrom);
    boxSet.add(land);
    boxes[boxes.indexOf(boxFrom)] = land;
    player = boxFrom;
  }
  return moves;
}

// ---------- push-level A* ----------
// weight = 1: optimal. weight > 1: weighted A* fallback (admissibility given
// up for speed; the answer is still replayed-legal by construction).
// comp = region ids for the matching restriction, or null. exitSquare >= 0
// additionally requires the player's region to reach that square in the goal
// state. Returns { moves, chain, expansions } or null when the cap is
// exhausted / the level is provably unsolvable.
function pushSearch(L, D, weight, cap, useMatching, comp, exitSquare = -1) {
  const { n, wall, goal, goals, nbr } = L;
  const nb = L.boxes.length;
  if (nb === 0) return { moves: '', chain: [], expansions: 0 };
  if (goals.length < nb) return null;
  const minDist = D.minDist;
  for (const b of L.boxes) if (minDist[b] < 0) return null; // box on a dead square

  let compR = comp;
  const costBuf = useMatching ? new Float64Array(nb * nb) : null;
  const hungarian = useMatching ? makeHungarian(nb) : null;
  // Restricted per-box minimum distances for the sum fallback.
  let minDistR = null;
  if (compR && !useMatching) {
    minDistR = new Int32Array(n).fill(-1);
    for (let s = 0; s < n; s++) {
      let m = -1;
      for (let j = 0; j < goals.length; j++) {
        if (compR[s] >= 0 && compR[goals[j]] >= 0 && compR[s] !== compR[goals[j]]) continue;
        const v = D.perGoal[j][s];
        if (v >= 0 && (m < 0 || v < m)) m = v;
      }
      minDistR[s] = m;
    }
  }
  const heuristic = (boxes) => {
    if (useMatching) {
      for (let i = 0; i < nb; i++) {
        const b = boxes[i];
        const cb = compR ? compR[b] : -1;
        for (let j = 0; j < nb; j++) {
          let c = D.perGoal[j][b];
          if (c < 0) c = INF;
          else if (cb >= 0 && compR[goals[j]] >= 0 && compR[goals[j]] !== cb) c = INF;
          costBuf[i * nb + j] = c;
        }
      }
      return hungarian(costBuf);
    }
    const md = compR && minDistR ? minDistR : minDist;
    let s = 0;
    for (let i = 0; i < nb; i++) {
      const d = md[boxes[i]];
      if (d < 0) return INF;
      s += d;
    }
    return s;
  };

  // Scratch buffers reused across expansions (generation stamps avoid clearing).
  const boxStamp = new Int32Array(n);
  let stamp = 0;
  // Separate occupancy stamps for deadlock checks on candidate box sets, so
  // generation never clobbers the current state's boxStamp.
  const deadStamp = new Int32Array(n);
  let dstamp = 0;
  const reachStamp = new Int32Array(n);
  let rstamp = 0;
  const queue = new Int32Array(n);

  // Flood fill the player's reachable region with boxes as obstacles.
  // Returns the canonical (minimum) square of the region; reachability is
  // then membership in reachStamp === rstamp.
  const flood = (from, boxes) => {
    stamp++;
    for (let i = 0; i < boxes.length; i++) boxStamp[boxes[i]] = stamp;
    rstamp++;
    let head = 0, tail = 0, minSq = from;
    queue[tail++] = from;
    reachStamp[from] = rstamp;
    while (head < tail) {
      const c = queue[head++];
      if (c < minSq) minSq = c;
      for (let d = 0; d < 4; d++) {
        const q = nbr[c * 4 + d];
        if (q < 0 || wall[q] || boxStamp[q] === stamp || reachStamp[q] === rstamp) continue;
        reachStamp[q] = rstamp;
        queue[tail++] = q;
      }
    }
    return minSq;
  };

  const isGoal = (boxes) => {
    for (let i = 0; i < boxes.length; i++) if (!goal[boxes[i]]) return false;
    return true;
  };

  // 2x2 block rule: any 2x2 of walls/boxes holding a non-goal box is dead.
  // Only the four 2x2 squares containing the moved box can newly form.
  // Occupancy is read from deadStamp/dstamp (the candidate box set).
  const block22 = (moved) => {
    const mx = moved % L.w, my = (moved / L.w) | 0;
    for (let oy = -1; oy <= 0; oy++) for (let ox = -1; ox <= 0; ox++) {
      const x0 = mx + ox, y0 = my + oy;
      if (x0 < 0 || y0 < 0 || x0 + 1 >= L.w || y0 + 1 >= L.h) continue;
      const cells = [y0 * L.w + x0, y0 * L.w + x0 + 1, (y0 + 1) * L.w + x0, (y0 + 1) * L.w + x0 + 1];
      let allSolid = true, badBox = false;
      for (const c of cells) {
        if (wall[c]) continue;
        if (deadStamp[c] === dstamp) { if (!goal[c]) badBox = true; continue; }
        allSolid = false; break;
      }
      if (allSolid && badBox) return true;
    }
    return false;
  };

  // Freeze check: iteratively mark non-goal boxes that cannot move on either
  // axis because a neighbour on each axis is a wall or an already-frozen box.
  const frozen = new Uint8Array(n);
  const hasFrozenDeadlock = (boxes) => {
    for (let i = 0; i < boxes.length; i++) frozen[boxes[i]] = 0;
    let changed = true;
    while (changed) {
      changed = false;
      for (let i = 0; i < boxes.length; i++) {
        const b = boxes[i];
        if (goal[b] || frozen[b]) continue;
        let axisBlocked = true;
        for (let axis = 0; axis < 2 && axisBlocked; axis++) {
          const d1 = axis === 0 ? 2 : 0; // left / up
          const d2 = axis === 0 ? 3 : 1; // right / down
          const s1 = nbr[b * 4 + d1], s2 = nbr[b * 4 + d2];
          const blocked = (s) => s < 0 || wall[s] || (deadStamp[s] === dstamp && frozen[s] === 1);
          if (!blocked(s1) && !blocked(s2)) axisBlocked = false;
        }
        if (axisBlocked) { frozen[b] = 1; changed = true; }
      }
    }
    for (let i = 0; i < boxes.length; i++) if (frozen[boxes[i]] && !goal[boxes[i]]) return true;
    return false;
  };

  const encode = (canon, boxes) => String.fromCharCode(canon, ...boxes);
  const decode = (key) => {
    const bs = new Array(key.length - 1);
    for (let i = 1; i < key.length; i++) bs[i - 1] = key.charCodeAt(i);
    return bs;
  };

  // Node storage: the key string carries the box squares; nodePush packs
  // boxFrom * 4 + dir for the push that created the node (-1 at the root).
  const nodeKey = [encode(flood(L.player, L.boxes), L.boxes)];
  const nodePlayer = [L.player]; // actual player square (box's old square after a push)
  const nodeParent = [-1];
  const nodePush = [-1];
  const nodeG = [0];
  let h0 = heuristic(L.boxes);
  if (h0 >= INF && compR) { compR = null; h0 = heuristic(L.boxes); }
  if (h0 >= INF) return null;
  const best = new Map([[nodeKey[0], 0]]);

  // Binary heap ordered by (f asc, g desc, insertion asc): deterministic.
  const heapF = [], heapG = [], heapNode = [], heapCnt = [];
  let counter = 0;
  const less = (i, j) => {
    if (heapF[i] !== heapF[j]) return heapF[i] < heapF[j];
    if (heapG[i] !== heapG[j]) return heapG[i] > heapG[j];
    return heapCnt[i] < heapCnt[j];
  };
  const swap = (i, j) => {
    [heapF[i], heapF[j]] = [heapF[j], heapF[i]];
    [heapG[i], heapG[j]] = [heapG[j], heapG[i]];
    [heapNode[i], heapNode[j]] = [heapNode[j], heapNode[i]];
    [heapCnt[i], heapCnt[j]] = [heapCnt[j], heapCnt[i]];
  };
  const hpush = (f, g, node) => {
    let i = heapF.length;
    heapF.push(f); heapG.push(g); heapNode.push(node); heapCnt.push(counter++);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(i, p)) break;
      swap(i, p); i = p;
    }
  };
  const hpop = () => {
    const top = heapNode[0];
    const lf = heapF.pop(), lg = heapG.pop(), ln = heapNode.pop(), lc = heapCnt.pop();
    if (heapF.length > 0) {
      heapF[0] = lf; heapG[0] = lg; heapNode[0] = ln; heapCnt[0] = lc;
      for (let i = 0; ;) {
        let c = 2 * i + 1;
        if (c >= heapF.length) break;
        if (c + 1 < heapF.length && less(c + 1, c)) c++;
        if (!less(c, i)) break;
        swap(i, c); i = c;
      }
    }
    return top;
  };
  hpush(h0 * weight, 0, 0);

  let expansions = 0;
  while (heapF.length > 0) {
    const s = hpop();
    if (best.get(nodeKey[s]) !== s) continue; // superseded by a cheaper path
    const boxes = decode(nodeKey[s]);
    if (isGoal(boxes)) {
      let ok = true;
      if (exitSquare >= 0) {
        flood(nodePlayer[s], boxes);
        ok = reachStamp[exitSquare] === rstamp;
      }
      if (ok) {
        const chain = [];
        for (let t = s; t > 0; t = nodeParent[t]) chain.push(nodePush[t]);
        chain.reverse();
        return { moves: reconstructMoves(L, chain), chain, expansions };
      }
      // Goal squares covered but the player cannot leave: keep searching.
    }
    if (expansions >= cap) return null;
    expansions++;
    const g = nodeG[s];
    const player = nodePlayer[s];
    flood(player, boxes); // reachStamp/boxStamp now describe this state
    const curStamp = stamp, curReach = rstamp;
    // Generate candidate pushes in fixed order (box index, direction), then
    // order them by heuristic delta for a better dive.
    const cands = [];
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const stand = nbr[b * 4 + (d ^ 1)];
        const land = nbr[b * 4 + d];
        if (stand < 0 || land < 0) continue;
        if (reachStamp[stand] !== curReach) continue;
        if (wall[land] || boxStamp[land] === curStamp) continue;
        if (!goal[land] && minDist[land] < 0) continue; // static dead square
        const nb2 = boxes.slice();
        nb2[bi] = land;
        nb2.sort((a, z) => a - z);
        // Deadlock checks read occupancy from deadStamp: stamp nb2 there.
        dstamp++;
        for (let i = 0; i < nb2.length; i++) deadStamp[nb2[i]] = dstamp;
        let dead = false;
        if (!goal[land] && block22(land)) dead = true;
        if (!dead && hasFrozenDeadlock(nb2)) dead = true;
        if (dead) continue;
        const hh = heuristic(nb2);
        if (hh >= INF) continue;
        cands.push({ nb2, hh, d, b });
      }
    }
    cands.sort((a, z) => (a.hh - z.hh) || (a.b - z.b) || (a.d - z.d));
    for (const c of cands) {
      const canon = flood(c.b, c.nb2); // player ends on the box's old square
      const key = encode(canon, c.nb2);
      const ng = g + 1;
      const prev = best.get(key);
      if (prev !== undefined && nodeG[prev] <= ng) continue;
      const t = nodeKey.length;
      nodeKey.push(key);
      nodePlayer.push(c.b);
      nodeParent.push(s);
      nodePush.push(c.b * 4 + c.d);
      nodeG.push(ng);
      best.set(key, t);
      hpush(ng + c.hh * weight, ng, t);
    }
  }
  return null;
}

// ---------- room decomposition ----------
// Split the level into rooms via the block tree (components + articulation
// squares), solve each room separately on its play area, and splice the push
// chains. Returns { moves, expansions } or null when the level does not
// decompose cleanly (caller falls back to the whole-level ladder).
function solveDecomposed(L, A) {
  const { n, wall, goal, nbr } = L;
  const { raw, isArt, nc, region } = A;
  if (nc === 0) return null;
  // Block tree: node ids 0..nc-1 for components, nc+ai for articulation
  // squares (ai = index into artSquares).
  const artSquares = [];
  const artId = new Int32Array(n).fill(-1);
  for (let s = 0; s < n; s++) if (isArt[s]) { artId[s] = artSquares.length; artSquares.push(s); }
  const total = nc + artSquares.length;
  const adjT = Array.from({ length: total }, () => []);
  const compSquares = Array.from({ length: nc }, () => []);
  for (let s = 0; s < n; s++) if (raw[s] >= 0) compSquares[raw[s]].push(s);
  for (let ai = 0; ai < artSquares.length; ai++) {
    const s = artSquares[ai];
    const seen = new Set();
    for (let d = 0; d < 4; d++) {
      const q = nbr[s * 4 + d];
      if (q >= 0 && raw[q] >= 0 && !seen.has(raw[q])) {
        seen.add(raw[q]);
        adjT[nc + ai].push(raw[q]);
        adjT[raw[q]].push(nc + ai);
      }
      // Articulation chains (the hallway, alcove necks) need art-art edges,
      // or whole stretches of corridor are unreachable in the tree.
      if (q >= 0 && isArt[q] && artId[q] > ai) {
        adjT[nc + ai].push(nc + artId[q]);
        adjT[nc + artId[q]].push(nc + ai);
      }
    }
  }
  // Root the block tree at the player's node.
  const playerNode = isArt[L.player] ? nc + artId[L.player] : raw[L.player];
  if (playerNode < 0) return null;
  const parentT = new Int32Array(total).fill(-2);
  const depthT = new Int32Array(total);
  parentT[playerNode] = -1;
  const order = [playerNode];
  for (let i = 0; i < order.length; i++) {
    const v = order[i];
    for (const to of adjT[v]) {
      if (parentT[to] !== -2) continue;
      parentT[to] = v;
      depthT[to] = depthT[v] + 1;
      order.push(to);
    }
  }
  if (order.length < total) {
    // Disconnected floor (should not happen); only proceed if every
    // box/goal component was reached.
    const reached = new Set(order);
    for (const b of L.boxes) if (raw[b] >= 0 && !reached.has(raw[b])) return null;
    for (const g of L.goals) if (raw[g] >= 0 && !reached.has(raw[g])) return null;
  }
  // Regions that hold boxes, in deterministic order (by first box square).
  const regionOfComp = new Int32Array(nc).fill(-1);
  for (let c = 0; c < nc; c++) regionOfComp[c] = compSquares[c].length ? region[compSquares[c][0]] : -1;
  const roomRegions = [];
  const seenRegion = new Set();
  for (const b of L.boxes) {
    const r = raw[b] >= 0 ? regionOfComp[raw[b]] : -1;
    if (r >= 0 && !seenRegion.has(r)) { seenRegion.add(r); roomRegions.push(r); }
  }
  // A box on an articulation square belongs to the room whose door or
  // interior chain it sits on; that is resolved per room below via play
  // areas. First pass: rooms from component boxes only.
  if (roomRegions.length === 0) return null;
  const rooms = [];
  const covered = new Uint8Array(n);
  let totalExpansions = 0;
  for (const r of roomRegions) {
    // Shallowest component of the region (closest to the player in the tree).
    let shallow = -1;
    for (let c = 0; c < nc; c++) {
      if (regionOfComp[c] !== r) continue;
      if (parentT[c] === -2) return null;
      if (shallow < 0 || depthT[c] < depthT[shallow]) shallow = c;
    }
    if (shallow < 0) return null;
    if (shallow === playerNode) return null; // player starts inside the room
    const doorNode = parentT[shallow];
    if (doorNode < nc) return null; // parent of a component is an articulation node
    const doorSquare = artSquares[doorNode - nc];
    // Play area: the subtree beyond the door (components + articulation
    // squares), plus the door square, plus the entry square on the player
    // side of the door.
    const inArea = new Uint8Array(n);
    const stack = [shallow];
    const seenNode = new Set([shallow, doorNode]);
    while (stack.length) {
      const v = stack.pop();
      if (v < nc) for (const s of compSquares[v]) inArea[s] = 1;
      else inArea[artSquares[v - nc]] = 1;
      for (const to of adjT[v]) {
        if (to === doorNode || seenNode.has(to)) continue;
        seenNode.add(to);
        stack.push(to);
      }
    }
    inArea[doorSquare] = 1;
    // Entry square: BFS from the player's start over squares outside the
    // play area until a square next to the play area is reached (the room's
    // neck). The entry is the box-free square on that approach path closest
    // to the room — a box sitting in the neck is pushed in from there — and
    // the path squares from the entry to the neck join the play area.
    let entrySquare = -1;
    if (inArea[L.player]) {
      entrySquare = L.player;
    } else {
      const par = new Int32Array(n).fill(-2);
      const oq = [L.player];
      par[L.player] = -1;
      let target = -1;
      for (let head = 0; head < oq.length; head++) {
        const c = oq[head];
        let adjIn = false;
        for (let d = 0; d < 4; d++) {
          const s = nbr[c * 4 + d];
          if (s >= 0 && inArea[s]) { adjIn = true; break; }
        }
        if (adjIn) { target = c; break; }
        for (let d = 0; d < 4; d++) {
          const s = nbr[c * 4 + d];
          if (s < 0 || wall[s] || inArea[s] || par[s] !== -2) continue;
          par[s] = c;
          oq.push(s);
        }
      }
      if (target < 0) return null;
      const path = [];
      for (let c = target; c >= 0; c = par[c]) path.push(c);
      // path runs target -> ... -> player; scan from the target backwards.
      for (const s of path) {
        inArea[s] = 1;
        if (!L.boxes.includes(s)) { entrySquare = s; break; }
      }
      if (entrySquare < 0) return null;
    }
    const roomBoxes = L.boxes.filter((b) => inArea[b]);
    const roomGoals = L.goals.filter((g) => inArea[g]);
    if (roomBoxes.length === 0 || roomBoxes.length !== roomGoals.length) return null;
    for (const b of roomBoxes) covered[b] = 1;
    for (const g of roomGoals) covered[g] = 1;
    rooms.push({ inArea, entrySquare, roomBoxes, roomGoals });
  }
  // Every box and goal must be covered by exactly one room.
  for (const b of L.boxes) if (!covered[b]) return null;
  for (const g of L.goals) if (!covered[g]) return null;
  for (let i = 0; i < rooms.length; i++) {
    for (let j = i + 1; j < rooms.length; j++) {
      for (const b of rooms[i].roomBoxes) if (rooms[j].inArea[b]) return null;
    }
  }
  // Solve the rooms in order, tracking the global box state for exit checks.
  const globalBoxes = L.boxes.slice();
  let globalPlayer = L.player;
  const chain = [];
  for (const room of rooms) {
    const subWall = new Uint8Array(n).fill(1);
    for (let s = 0; s < n; s++) if (room.inArea[s]) subWall[s] = 0;
    const sub = {
      w: L.w, h: L.h, n,
      wall: subWall,
      goal,
      goals: room.roomGoals,
      boxes: room.roomBoxes,
      player: room.entrySquare,
      nbr,
    };
    const D = pushDistances(sub);
    const useMatching = room.roomBoxes.length <= 16;
    let r = pushSearch(sub, D, 1, CAP_ROOM, useMatching, null, -1);
    if (!r) return null;
    totalExpansions += r.expansions;
    // Exit check on the global state after this room's pushes: the player
    // (on the last pushed box's old square) must reach the entry square.
    const after = globalBoxes.slice();
    for (const packed of r.chain) {
      const bf = (packed / 4) | 0;
      const land = nbr[bf * 4 + (packed % 4)];
      after[after.indexOf(bf)] = land;
    }
    const playerAfter = r.chain.length ? (r.chain[r.chain.length - 1] / 4) | 0 : globalPlayer;
    if (!canReach(L, after, playerAfter, room.entrySquare)) {
      r = pushSearch(sub, D, 1, CAP_ROOM, useMatching, null, room.entrySquare);
      if (!r) return null;
      totalExpansions += r.expansions;
      for (const packed of r.chain) {
        const bf = (packed / 4) | 0;
        const land = nbr[bf * 4 + (packed % 4)];
        globalBoxes[globalBoxes.indexOf(bf)] = land;
      }
      globalPlayer = r.chain.length ? (r.chain[r.chain.length - 1] / 4) | 0 : globalPlayer;
    } else {
      for (let i = 0; i < after.length; i++) globalBoxes[i] = after[i];
      globalPlayer = playerAfter;
    }
    chain.push(...r.chain);
  }
  return { moves: reconstructMoves(L, chain), expansions: totalExpansions, chain };
}

// Can the player, standing at `from` with `boxes` placed, reach `target`?
function canReach(L, boxes, from, target) {
  if (from === target) return true;
  const boxSet = new Set(boxes);
  const seen = new Uint8Array(L.n);
  const queue = [from];
  seen[from] = 1;
  while (queue.length) {
    const c = queue.pop();
    for (let d = 0; d < 4; d++) {
      const q = L.nbr[c * 4 + d];
      if (q < 0 || L.wall[q] || boxSet.has(q) || seen[q]) continue;
      if (q === target) return true;
      seen[q] = 1;
      queue.push(q);
    }
  }
  return false;
}

// ---------- move-level A* fallback (baseline style, larger cap) ----------
function moveSearch(L, cap) {
  const { wall, goal, goals, nbr } = L;
  const dist = (b) => {
    let m = INF;
    for (const g of goals) {
      const d = Math.abs((g % L.w) - (b % L.w)) + Math.abs(((g / L.w) | 0) - ((b / L.w) | 0));
      if (d < m) m = d;
    }
    return m;
  };
  const est = (bs) => bs.reduce((a, b) => a + dist(b), 0);
  const key = (p, bs) => String.fromCharCode(p, ...bs);
  const sp = [L.player];
  const sb = [L.boxes.slice()];
  const sg = [0];
  const parent = [-1];
  const step = [''];
  const best = new Map([[key(L.player, L.boxes), 0]]);
  const heap = [[est(L.boxes), 0, 0]];
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] > b[1] || (a[1] === b[1] && a[2] < b[2])));
  const push = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const pop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  let expanded = 0;
  while (heap.length > 0 && expanded < cap) {
    const [, g, s] = pop();
    if (best.get(key(sp[s], sb[s])) !== s) continue;
    const bs = sb[s];
    if (bs.every((b) => goal[b])) {
      let moves = '';
      for (let t = s; t > 0; t = parent[t]) moves = step[t] + moves;
      return moves;
    }
    expanded++;
    const p = sp[s];
    for (let d = 0; d < 4; d++) {
      const q = nbr[p * 4 + d];
      if (q < 0 || wall[q]) continue;
      let nb = bs;
      let move = LET[d];
      const at = bs.indexOf(q);
      if (at >= 0) {
        const beyond = nbr[q * 4 + d];
        if (beyond < 0 || wall[beyond] || bs.includes(beyond)) continue;
        nb = bs.slice();
        nb[at] = beyond;
        nb.sort((a, z) => a - z);
        move = LET[d].toUpperCase();
      }
      const k = key(q, nb);
      const seen = best.get(k);
      if (seen !== undefined && sg[seen] <= g + 1) continue;
      const t = sp.length;
      sp.push(q); sb.push(nb); sg.push(g + 1); parent.push(s); step.push(move);
      best.set(k, t);
      push([g + 1 + est(nb), g + 1, t]);
    }
  }
  return '';
}

function solve(text) {
  try {
    const L = parseLevel(text);
    if (L.player < 0) return '';
    const A = analyze(L);
    const dec = solveDecomposed(L, A);
    if (dec) return dec.moves;
    const D = pushDistances(L);
    const useMatching = L.boxes.length <= 16 && L.goals.length === L.boxes.length;
    let r = pushSearch(L, D, 1, CAP_OPT, useMatching, A.region);
    if (r) return r.moves;
    r = pushSearch(L, D, 3, CAP_WEIGHTED, useMatching, A.region);
    if (r) return r.moves;
    return moveSearch(L, CAP_MOVE);
  } catch {
    return '';
  }
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  try {
    moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
