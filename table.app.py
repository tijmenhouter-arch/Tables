import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.patches as patches
from matplotlib.figure import Figure
from scipy.optimize import minimize

st.set_page_config(page_title="Classroom table layout", layout="wide")

SEATS_PER_TABLE = 4
SEAT_EDGE_MARGIN = 0.35  # m, first/last seat from the table's short edge
WALL_STEP = 0.10         # m, resolution of candidate positions for new sockets

# Existing layout of Arch hall R
ORIGINAL_LAYOUT = dict(
    table_w=0.70,   # table size in the existing layout
    table_l=2.85,
    rows=[
        # y = centre line of the row, x = left edge of each table in that row,
        # x_min = where the row starts (only used for the first gap arrow)
        dict(y=6.075, x_min=0.0, x=[2.12, 3.61, 5.07, 6.57, 8.03, 9.34]),
        dict(y=1.425, x_min=2.2, x=[2.87, 4.30, 5.79, 7.16, 9.01]),
    ],
)


# ----------------------------------------------------------------------------
# Geometry & optimisation
# ----------------------------------------------------------------------------
def seats_for_table(x_left, y_center, table_w, table_l):
    y_off = np.linspace(-table_l / 2 + SEAT_EDGE_MARGIN,
                        table_l / 2 - SEAT_EDGE_MARGIN, SEATS_PER_TABLE)
    return np.column_stack([np.full(SEATS_PER_TABLE, x_left + table_w / 2),
                            y_center + y_off])


def tables_that_fit(span, table_w, min_gap):
    """Max number of tables in a row if every table has >= min_gap in front of it."""
    return max(0, int(np.floor((span + 1e-9) / (table_w + min_gap))))

def split_tables(n_total, caps):
    """Spread n_total tables over the rows in proportion to each row's capacity."""
    counts = [0] * len(caps)
    for _ in range(min(n_total, sum(caps))):
        open_rows = [i for i, c in enumerate(caps) if counts[i] < c]
        i = min(open_rows, key=lambda i: counts[i] / caps[i])
        counts[i] += 1
    return counts


def optimize_row(n, x_min, x_max, y_center, sockets, reach,
                 table_w, table_l, min_gap, max_gap, priority):
    """Returns the x_left of each table in the row."""
    if n == 0:
        return []

    free = (x_max - x_min) - n * table_w          # space to divide over n gaps
    gaps0 = np.full(n, free / n)

    # priority 0 = only even spacing, 1 = only power coverage
    w_cable = 1000.0 * priority
    w_uniform = 700.0 * (1.0 - priority)

    def positions(gaps):
        xs, cur = [], x_min
        for g in gaps:
            cur += g
            xs.append(cur)
            cur += table_w
        return xs

    def objective(gaps):
        uniformity = np.var(gaps) * w_uniform
        if len(sockets) == 0:
            return uniformity
        seats = np.vstack([seats_for_table(x, y_center, table_w, table_l)
                           for x in positions(gaps)])
        d = np.linalg.norm(seats[:, None, :] - sockets[None, :, :], axis=2).min(axis=1)
        # capped "magnetism": beyond 0.5 m short, a seat stops pulling the tables
        shortfall = np.minimum(np.maximum(0.0, d - reach), 0.5)
        return np.sum(shortfall ** 2) * w_cable + uniformity

    ub = max(max_gap, free / n)
    res = minimize(objective, gaps0, method="SLSQP",
                   bounds=[(min_gap, ub)] * n,
                   constraints=({"type": "eq", "fun": lambda g: np.sum(g) - free},))
    gaps = res.x if res.success else gaps0
    return positions(gaps)


def propose_new_sockets(seats, existing, room_l, room_w, reach, n_new):
    """Greedy: place each new wall socket where it powers the most unpowered seats."""
    if n_new <= 0 or len(seats) == 0:
        return np.empty((0, 2))

    if len(existing):
        d = np.linalg.norm(seats[:, None, :] - existing[None, :, :], axis=2).min(axis=1)
        uncovered = seats[d > reach]
    else:
        uncovered = seats.copy()

    nx, ny = int(room_l / WALL_STEP) + 1, int(room_w / WALL_STEP) + 1
    xs, ys = np.linspace(0, room_l, nx), np.linspace(0, room_w, ny)
    cands = np.vstack([
        np.column_stack([xs, np.zeros(nx)]),
        np.column_stack([xs, np.full(nx, room_w)]),
        np.column_stack([np.zeros(ny), ys]),
        np.column_stack([np.full(ny, room_l), ys]),
    ])

    placed = []
    for _ in range(n_new):
        if len(uncovered) == 0:
            break
        d = np.linalg.norm(uncovered[None, :, :] - cands[:, None, :], axis=2)  # (cand, seat)
        covered = d <= reach
        counts = covered.sum(axis=1)
        best = counts.argmax()
        if counts[best] == 0:
            break
        placed.append(cands[best])
        uncovered = uncovered[~covered[best]]
    return np.array(placed).reshape(-1, 2)


def build_layout(p):
    x_end = p["room_l"] - p["lect_d"]  # tables stop where the lecturer space starts
    rows = [
        dict(y=p["room_w"] - p["table_l"] / 2, x_min=0.0, x_max=10),    # north row
        dict(y=p["table_l"] / 2, x_min=p["ent_w"], x_max=9.76),           # south row
    ]
    sockets = p["sockets"]
    for r in rows:
        caps = [tables_that_fit(r["x_max"] - r["x_min"], p["table_w"], p["min_gap"]) for r in rows]
        for r, n in zip(rows, split_tables(p["n_tables"], caps)):
            r["n"] = n
            r["x"] = optimize_row(r["n"], r["x_min"], r["x_max"], r["y"], sockets, p["reach"],
                              p["table_w"], p["table_l"], p["min_gap"], p["max_gap"],
                              p["priority"])

    seat_blocks = [seats_for_table(x, r["y"], p["table_w"], p["table_l"])
                   for r in rows for x in r["x"]]
    seats = np.vstack(seat_blocks) if seat_blocks else np.empty((0, 2))

    new = propose_new_sockets(seats, sockets, p["room_l"], p["room_w"], p["reach"], p["n_new"])
    all_sockets = np.vstack([sockets, new]) if len(new) else sockets
    if len(seats) and len(all_sockets):
        powered = np.linalg.norm(seats[:, None, :] - all_sockets[None, :, :],
                                 axis=2).min(axis=1) <= p["reach"]
    else:
        powered = np.zeros(len(seats), dtype=bool)
    return dict(rows=rows, seats=seats, powered=powered, new=new, x_end=x_end)

def build_original_layout(p, x_end):
    """Existing layout of the hall, scored against the existing sockets only."""
    o = ORIGINAL_LAYOUT
    rows = [dict(y=r["y"], x_min=r["x_min"], x=list(r["x"]), n=len(r["x"])) for r in o["rows"]]
    blocks = [seats_for_table(x, r["y"], o["table_w"], o["table_l"])
              for r in rows for x in r["x"]]
    seats = np.vstack(blocks) if blocks else np.empty((0, 2))
    sockets = p["sockets"]
    if len(seats) and len(sockets):
        powered = np.linalg.norm(seats[:, None, :] - sockets[None, :, :],
                                 axis=2).min(axis=1) <= p["reach"]
    else:
        powered = np.zeros(len(seats), dtype=bool)
    return dict(rows=rows, seats=seats, powered=powered, new=np.empty((0, 2)), x_end=x_end)

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
def draw(p, lay):
    fig = Figure(figsize=(9, 6))
    ax = fig.subplots()
    ax.set_aspect("equal")
    L, W, tw, tl = p["room_l"], p["room_w"], p["table_w"], p["table_l"]

    ax.add_patch(patches.Rectangle((0, 0), L, W, lw=2, ec="black", fc="none"))

    # Lecturer space
    ax.add_patch(patches.Rectangle((lay["x_end"], 0), p["lect_d"], W, lw=1, ec="gray",
                                   fc="#d9d9d9", alpha=0.5,
                                   label=f"Lecturer space ({p['lect_d'] * W:.1f} m²)"))
    ax.text(lay["x_end"] + p["lect_d"] / 2, W / 2, "Lecturer space", rotation=90,
            va="center", ha="center", fontsize=10, color="#333333")

    # Entrance
    ax.add_patch(patches.Rectangle((0, 0), p["ent_w"], p["ent_d"], lw=1, ec="gray",
                                   fc="#d9d9d9", alpha=0.5,
                                   label=f"Entrance ({p['ent_w'] * p['ent_d']:.1f} m²)"))
    ax.text(p["ent_w"] / 2, p["ent_d"] / 2, "Entrance", va="center", ha="center",
            fontsize=10, color="#333333")

    # Tables + gap markers
    for r in lay["rows"]:
        prev = r["x_min"]
        for x in r["x"]:
            ax.add_patch(patches.Rectangle((x, r["y"] - tl / 2), tw, tl, lw=1.5,
                                           ec="black", fc="#8B5A2B", alpha=0.8))
            gap = x - prev
            if gap > 0.1:
                ax.annotate("", xy=(x, r["y"]), xytext=(prev, r["y"]),
                            arrowprops=dict(arrowstyle="<|-|>", color="#999999",
                                            shrinkA=0, shrinkB=0, lw=0.8, alpha=0.8))
                ax.text(prev + gap / 2, r["y"] + 0.15, f"{gap:.2f}m", ha="center",
                        va="bottom", fontsize=7, color="#555555",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))
            prev = x + tw

    # Sockets + reach
    for i, s in enumerate(p["sockets"]):
        ax.add_patch(patches.Circle(s, p["reach"], color="royalblue", alpha=0.2))
        ax.plot(*s, "s", color="blue", ms=8, mec="black",
                label="Existing socket" if i == 0 else None)
    for i, s in enumerate(lay["new"]):
        ax.add_patch(patches.Circle(s, p["reach"], color="darkorange", alpha=0.3))
        ax.plot(*s, "*", color="gold", ms=16, mec="black",
                label="Proposed socket" if i == 0 else None)

    # Seats
    seats, powered = lay["seats"], lay["powered"]
    if len(seats):
        ax.scatter(*seats[powered].T, c="lime", edgecolors="black", s=40, zorder=5)
        ax.scatter(*seats[~powered].T, c="red", edgecolors="black", s=40, zorder=5)
    ax.plot([], [], "o", color="lime", mec="black", label="Powered seat")
    ax.plot([], [], "o", color="red", mec="black", label="Unpowered seat")

    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, W + 0.5)
    ax.set_xticks(np.arange(0, L + 1, 1))
    ax.set_yticks(np.arange(0, W + 1, 1))
    ax.grid(True, ls="--", alpha=0.5)
    ax.set_xlabel("Room length (X) [m]")
    ax.set_ylabel("Room width (Y) [m]")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Main settings")
    min_gap = st.slider("Minimum space between tables [m]", 0.50, 1.50, 0.80, 0.05,
                        help="Smaller value = more tables fit in a row.")
    
    show_original = st.toggle("Original layout Arch hall R", value=True)
    tables_slot = st.container()
    reach = st.slider("Cable length [m]", 0.5, 4.0, 1.80, 0.1)

    st.markdown("**Existing power sockets** (x, y in m)")
    sockets_df = st.data_editor(
        pd.DataFrame({"x": [2.5, 2.5, 9.5, 9.5], "y": [0.0, 7.5, 0.0, 7.5]}),
        num_rows="dynamic", hide_index=True, width="stretch",
        column_config={"x": st.column_config.NumberColumn("x [m]", format="%.2f"),
                       "y": st.column_config.NumberColumn("y [m]", format="%.2f")},
    )

    add_new = st.toggle("Add new sockets", value=False)
    n_new = st.number_input("Number of new sockets", 1, 20, 2) if add_new else 0
    
    priority = st.slider("Priority: even spacing ↔ power coverage", 0, 100, 70,
                         help="0 = spread tables evenly, ignore sockets. "
                              "100 = move tables towards sockets, ignore spacing.") / 100

    with st.expander("Room & furniture dimensions"):
        room_l = st.number_input("Classroom length (X) [m]", 4.0, 40.0, 12.0, 0.5)
        room_w = st.number_input("Classroom width (Y) [m]", 3.0, 30.0, 7.5, 0.5)
        table_w = st.number_input("Table width [m]", 0.3, 2.0, 0.70, 0.05)
        table_l = st.number_input("Table length [m]", 1.0, 5.0, 2.85, 0.05)
        max_gap = st.number_input("Maximum space between tables [m]", 0.5, 5.0, 1.50, 0.1)
        st.markdown("**Lecturer space** (full width, at the far end)")
        lect_d = st.number_input("Lecturer space depth [m]", 0.0, 10.0, 1.96, 0.05)
        st.markdown("**Entrance** (bottom-left corner)")
        ent_w = st.number_input("Entrance width (X) [m]", 0.0, 10.0, 2.20, 0.1)
        ent_d = st.number_input("Entrance depth (Y) [m]", 0.0, 10.0, 3.75, 0.1)

x_end = room_l - lect_d
max_tables = (tables_that_fit(x_end, table_w, min_gap)
              + tables_that_fit(x_end - ent_w, table_w, min_gap))

with tables_slot:
    use_max = st.checkbox("Use maximum number of tables", value=False)
    n_tables_input = st.number_input("Number of tables", 1, 100, 11, disabled=use_max)
    n_tables_wanted = max_tables if use_max else min(n_tables_input, max_tables)
    if not use_max and n_tables_input > max_tables:
        st.caption(f"Only {max_tables} fit with the current minimum space, so {max_tables} are used.")

params = dict(
    n_tables=int(n_tables_wanted), min_gap=min_gap,
    reach=reach, n_new=int(n_new), priority=priority,
    sockets=sockets_df.dropna().to_numpy(dtype=float).reshape(-1, 2),
    room_l=room_l, room_w=room_w, table_w=table_w, table_l=table_l, max_gap=max_gap,
    lect_d=lect_d, ent_w=ent_w, ent_d=ent_d,
)

st.title("Classroom table layout optimiser")

if 2 * table_l > room_w:
    st.error("The two table rows don't fit in the classroom width. "
             "Reduce the table length or increase the classroom width.")
elif lect_d + ent_w >= room_l:
    st.error("Lecturer space and entrance don't fit in the classroom length.")
else:
    lay = build_layout(params)
    n_tables = sum(r["n"] for r in lay["rows"])
    n_seats = len(lay["seats"])
    n_powered = int(lay["powered"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tables", n_tables)
    c2.metric("Seats", n_seats)
    c3.metric("Powered seats", f"{n_powered} / {n_seats}",
              f"{100 * n_powered / max(n_seats, 1):.0f}%", delta_color="off")
    c4.metric("New sockets placed", len(lay["new"]))

    st.subheader("Optimised layout")
    plot_col, info_col = st.columns([3, 2])
    plot_col.pyplot(draw(params, lay), width="stretch")
    show_original = st.toggle("Original layout Arch hall R", value=True)
    if show_original:
        st.divider()
        st.subheader("Original layout: Arch hall R")
        orig = build_original_layout(params, lay["x_end"])
        p_orig = {**params, "table_w": ORIGINAL_LAYOUT["table_w"], "table_l": ORIGINAL_LAYOUT["table_l"]}
        o_tables = sum(r["n"] for r in orig["rows"])
        o_seats = len(orig["seats"])
        o_powered = int(orig["powered"].sum())
        pct_orig = 100 * o_powered / max(o_seats, 1)
        pct_opt = 100 * n_powered / max(n_seats, 1)
    
        orig_plot, orig_info = st.columns([3, 2])
        orig_plot.pyplot(draw(p_orig, orig), width="stretch")
        with orig_info:
            st.metric("Tables", o_tables, int(n_tables - o_tables), delta_color="off")
            st.metric("Seats", o_seats, int(n_seats - o_seats), delta_color="off")
            st.metric("Powered seats", f"{o_powered} / {o_seats}", f"{pct_orig:.0f}%", delta_color="off")
            st.metric("Powered seats, optimised vs original", f"{pct_opt:.0f}%",
                      f"{pct_opt - pct_orig:+.0f} percentage points")
            st.caption("Original layout is scored with the existing sockets only. "
                       "Differences next to Tables and Seats are optimised minus original.")

    if len(lay["new"]):
        st.caption("Proposed socket positions: " + " · ".join(
            f"#{i + 1} (x={x:.2f}, y={y:.2f})" for i, (x, y) in enumerate(lay["new"])))
