import time
from typing import List

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, visualize: bool, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.visualize = visualize
        self.fig, self.ax = [], []
        self.exit = False

        if not visualize:
            return

        # Uncomment to make nice pgf plots:

        # import matplotlib
        # matplotlib.use("pgf")
        # matplotlib.rcParams.update({
            # "pgf.texsystem": "pdflatex",
            # 'font.family': 'serif',
            # 'text.usetex': True,
            # 'pgf.rcfonts': False,
        # })


        # Simulation
        fig, ax = plt.subplots()
        self.fig.append(fig)
        self.ax.append(ax)

        # Statistics
        fig, axes = plt.subplots(2, 1, sharex=True)
        plt.xlabel("Ticks")
        axes[0].set(title="Tasks per tick", ylabel="Tasks")
        axes[1].set(title="Communication units per tick", ylabel="Communication Units")
        self.fig.append(fig)
        self.ax.append(axes)

        [fig.canvas.mpl_connect("close_event", self._handle_close) for fig in self.fig]
        plt.ion()
        plt.show()

    def simulation(
        self, robots: List["Robot"], tasks: List["Task"], edges: List["Edge"]
    ) -> None:
        self.ax[0].set_aspect("equal", "box")
        self.ax[0].axis([0, self.width, 0, self.height])

        self._robots(robots)
        self._tasks(tasks)
        self._edges(edges)

        # Uncomment to make a plot

        # self.fig[0].tight_layout()
        # img_path = "../../tex2/img/area_1.pgf"
        # print(f"Written figure to '{img_path}'")
        # self.fig[0].savefig(img_path)
        # exit(1)

        self.fig[0].canvas.draw()
        self.fig[0].canvas.start_event_loop(1e-7)
        self.ax[0].clear()

    def statistics(self, statistics: "Statistics") -> None:
        tasks_x, tasks_y = zip(*statistics._tasks)
        com_x, com_y = zip(*statistics._com_units)

        self.ax[1][0].plot(tasks_x, tasks_y, color="C0")
        self.ax[1][1].plot(com_x, com_y, color="C0")

        self.fig[1].canvas.update()
        self.fig[1].canvas.flush_events()

    def _handle_close(self, event) -> None:
        self.exit = True

    def _robots(self, robots: List["Robot"]) -> None:
        circles = []
        for r in robots:
            #Uncomment to make a plot

            # self.ax[0].text(
            # r.x,
            # r.y,
            # f"{r.id, r.capacity, r.org_speed}",
            # fontsize=10,
            # )
            circles.append(
                plt.Circle(
                    xy=(r.x, r.y),
                    radius=10,
                    facecolor="grey",
                    # edgecolor="k",
                    alpha=0.5,
                    zorder=1,
                )
            )

        [self.ax[0].add_artist(c) for c in circles]

    def _tasks(self, tasks: List["Task"]) -> None:
        xs, ys = zip(*[(task.x, task.y) for task in tasks])
        self.ax[0].scatter(xs, ys, c="k", marker="x", zorder=3)

        circles = []
        for t in tasks:
            self.ax[0].text(t.x + 5, t.y + 5, str(t.id), fontsize=10)
            circles.append(
                plt.Circle(
                    xy=(t.x, t.y),
                    radius=t.r,
                    facecolor="none",
                    edgecolor="k",
                    # alpha=0.5,
                    zorder=1,
                )
            )

        [self.ax[0].add_artist(c) for c in circles]

    def _edges(self, edges: List["Edge"]) -> None:
        for e in edges:
            self.ax[0].plot(*zip(*(e.r1.pos(), e.r2.pos())), c="k", zorder=1, alpha=0.5)
