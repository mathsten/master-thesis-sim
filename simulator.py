import time
import random
from collections import defaultdict
# random.seed(0)
import statistics
from typing import List, Tuple

import numpy as np

# np.random.seed(0)
import matplotlib.pyplot as plt
#from pynput.keyboard import Key, Listener, KeyCode

# from robot import Robot, Task, Edge, Message, Counter
# from visualization import Visualization
# from statistics import Statistics

if __name__ != "__main__":
    from sim.robot import Robot, Task, Edge, Message, Counter
    from sim.visualization import Visualization
    from sim.statistics import Statistics


class Simulator:
    def __init__(
        self,
        caption: str,
        width: int,
        height: int,
        robots: "Robot",
        tasks: "Robot",
        method: str,
        end: int = 100_0000,
        fps: int = 1,
        speed: int = 1,
        collect_data_rate: int = 100,
        viz_rate: int = 1000,
        visualize: bool = True,
        threshold = 50,
        task_behavior_threshold = 10,
        auction_timer = None,
        noise=0,
    ) -> None:

        self.robots = robots
        [robot.set_method(method) for robot in self.robots]
        [robot.set_threshold(threshold) for robot in self.robots]
        self.threshold = threshold
        [robot.set_task_behavior_threshold(task_behavior_threshold) for robot in
                self.robots]
        # [robot.set_auction_timer(auction_timer) for robot in self.robots]
        if method == "call off dynamic" or method == "call out dynamic" or method == "broadcast dynamic":
            [robot.set_method(method) for robot in self.robots]

        self.tasks = tasks
        self.noise = noise
        self.counter = Counter()
        self.solved_tasks = []
        self.edges = []
        self.width = width
        self.height = height
        self.fps = fps
        self.method = method
        self.end = end
        self.collect_data_rate = collect_data_rate
        self.viz_rate = viz_rate
        self.statistics = Statistics()
        self.visualize = Visualization(visualize, self.width, self.height)
        self.auction_timer = auction_timer
        self.pause = False
        self.speed = speed
        self.once = False
        self.engaged_tasks = []
        self.engaged_in_tasks = []

    def _exit(self) -> bool:
        return self.visualize.exit

    def _robot_within_task(self) -> bool:
        any_robot_within = False
        for robot in self.robots:
            robot.within_task = []

        tasks = np.array([np.array([complex(task.x, task.y) for task in self.tasks])])
        robots = np.array(
            [np.array([complex(robot.x, robot.y) for robot in self.robots])]
        )
        distances = abs(robots.T - tasks)

        if np.any(distances < self.tasks[0].r):
            indices = np.argwhere(distances < self.tasks[0].r)

            for (i, j) in indices:
                self.robots[i].within_task.append(self.tasks[j])

                any_robot_within = True
        return any_robot_within

    def _generate_edges(self) -> None:
        self.edges = []
        for robot in self.robots:
            robot.edges = []

        z = np.array([[complex(robot.x, robot.y) for robot in self.robots]])
        distances = abs(z.T - z)
        distances = np.triu(distances)
        indices = np.argwhere(distances < self.robots[0].r)

        for (j, i) in indices:
            if j >= i:
                continue

            edge = Edge(noise=self.noise, r1=self.robots[i], r2=self.robots[j],d=distances[j, i],
                    counter=self.counter)
            self.robots[i].edges.append(edge)
            self.robots[j].edges.append(edge)
            self.edges.append(edge)

    def _request_help(self, msg_type: "Message") -> None:
        for robot in self.robots:
            if not robot.within_task:
                continue

            if robot.state in ["helping out", "awaiting help", "solving task"]:
                continue

            robot.task = robot.within_task[0]

            if (
                msg_type == "broadcast" 
                or msg_type == "broadcast dynamic"
                or msg_type == "call out"
                or msg_type == "call off" 
                or msg_type == "call off dynamic" 
                or msg_type == "call out dynamic"
            ):
                robot.send(Message(type="broadcast", data=robot.within_task[0]))
                robot.state = "awaiting help"

            elif msg_type == "quorum":
                robot.send(Message(type="broadcast", data=robot.within_task[0]))
                robot.state = "awaiting help"

            elif msg_type == "auction" or msg_type == "auction timer":
                robot.announce_task()
                robot.auction_task()
                # robot.send(Message(type="auction", data=robot))
                # robot.state = "awaiting help"
            elif msg_type in "random":
                robot.task = robot.within_task[0]

            elif msg_type == "baseline":
                robot.task = robot.within_task[0]
                robot.state = "awaiting help"
                # robot.state = "helping out"
            else:
                robot.state = "awaiting help"

            # robot.state = "awaiting response"

    def _solve_task(self, i):
        for task in self.tasks:
            # robots = 0
            robots = []
            robot_cap = 0
            for robot in self.robots:
                if task != robot.task:
                    continue

                if task not in robot.within_task:
                    continue
                # if task not in robot.within_task or robot.task != task:
                    # continue
                # robot.state = "helping out"
                # if robot.method == "quorum":
                    # print(robot.id, robot.task, robot.state)
                # if robot.method == "quorum" and robot.state == "search":
                    # continue

                if robot.state == "helping out" and robot.method != "random":
                    robot.state = "solving task"
                # robots += 1
                robots.append(robot)
                robot_cap += robot.capacity

            # Spawn new task
            # if len(robots) >= task.cap:
            if robot_cap >= task.cap:
                x = random.randint(0, self.width)
                y = random.randint(0, self.height)

                self.tasks.append(Task(x=x, y=y, r=task.r, cap=task.cap))
                self.solved_tasks.append(task)
                self.tasks.remove(task)
                
                if (self.method == "auction" 
                    or self.method == "auction timer"
                    or self.method == "call off"
                    or self.method == "call off dynamic"
                    or self.method == "quorum"
                ):
                    for robot in self.robots:
                        if robot.task == task:
                            robot.reset()
                else:
                    for robot in robots:
                    # for robot in self.robots:
                        # if robot.task == task and robot.method != "random":
                        if robot.method != "random":
                            robot.reset()

                            if robot.method == "task behavior":
                                robot.completed_tasks.append(i)
                                robot.task_behavior_counter(reset=True)
                            # if (robot.state == "awaiting help" or
                                # robot.state == "solving task" or
                                # robot.state == "awaiting bids" or
                                # robot.state == "helping out"):
                                # robot.reset()

    def _quorum_sensing(self, i):
        self._generate_edges()
        for robot in self.robots:
            robot.quorum_sensing(i)

    def _collect_data(self, i: int) -> None:
        self.statistics.log_tasks(i, len(self.solved_tasks))
        self.statistics.log_com_units(i, self.counter.i)  # edge.com_units())
        # self.statistics.log_tasks(i, self.solved_tasks)
        # self.statistics.log_com_units(i, self.counter.i)# Edge.com_units())

    def score(self, n) -> float:
        scores = list(zip(*self.statistics._tasks))[1]
        return statistics.mean(scores[-n:])
        # return self.statistics._tasks[-1][1]

    def coms(self, n) -> float:
        coms = list(zip(*self.statistics._com_units))[1]
        return statistics.mean(coms[-n:])
        # return self.statistics._com_units[-1][1]

    def run(self):
        # pause = False
        # with Listener(on_press=self.on_press) as listener:
        qqs = []
        if True:
            # i = 0
            # while True:
            t1 = time.time()
            for i in range(self.end):
                # if self.pause:
                # i -= 1
                # listener.join()
                # continue

                # i += 1

                # Logic
                # print("----")
                # [print(robot.id, robot.state, robot.timer) for robot in self.robots]
                # [print(robot.quorum) for robot in self.robots]
                [robot.move(w=self.width, h=self.height) for robot in self.robots]

                if self.method == "quorum":
                    self._quorum_sensing(i)
                    # print([r.quorum for r in self.robots])
                    # qqs.append(np.mean([r.quorum for r in self.robots]))
                    # print("--")
                elif self.method == "task behavior":
                    [robot.task_behavior_counter() for robot in self.robots]

                if self._robot_within_task():
                    if self.method not in ["random", "baseline"]:
                    # if self.method in ["auction", "broadcast", "quorum", "task behavior", "auction timer"]:
                        self._generate_edges()
                    self._request_help(msg_type=str(self.method))

                    # print("----")
                    # print(self.method, self.threshold, [robot.quorum for robot in self.robots])
                    # print(np.mean([robot.quorum for robot in self.robots]),
                            # np.std([robot.quorum for robot in self.robots]))


                    dd = defaultdict(int)
                    for task in [r.task for r in self.robots]:
                        if task is None:
                            continue
                        dd[task.id] += 1

                    self.engaged_tasks.append(len(dd.keys()))
                    self.engaged_in_tasks.append(np.mean([v for v in
                        dd.values()]))

                        # print(key, int(100*value/len(self.robots)))

                    # [print(robot.id, robot.state, robot.task, robot._task_behavior_counter) for robot in self.robots]
                    # self._request_help(msg_type="broadcast")
                    # if self.method == "auction":
                        # [robot.auction_task() for robot in self.robots]
                    self._solve_task(i)

                    for robot in self.robots:
                        robot.have_bid = False
                        if robot.state == "awaiting bids":
                            robot.reset()
                        # if (robot.state == "awaiting help" and robot.task not in robot.within_task):
                            #print(f"\n\n\n\n {robot}, {robot.task}, {robot.within_task} \n WHAT THE F**K!!!!\n\n\n\n")


                # [print(robot.state) for robot in self.robots]
                # Data collection rate
                if i % self.collect_data_rate == 0:
                    self._collect_data(i)
                    # t2 = time.time()
                    # print(t2 - t1)
                    # t1 = t2

                if self.visualize.visualize:
                    if i % self.speed == 0:
                        self._generate_edges()
                        self.visualize.simulation(self.robots, self.tasks, self.edges)
                        # print(np.mean(qqs))

                        if i % self.speed * 2 == 0:
                            self.visualize.statistics(self.statistics)

                if self._exit():
                    break

    def on_press(self, key):
        if key == Key.space:
            self.pause = not self.pause

        if key == KeyCode.from_char("+"):
            self.speed += 1
            # print(self.speed)

        if key == KeyCode.from_char("-"):
            self.speed -= 1
            if self.speed <= 0:
                self.speed = 1
            # print(self.speed)


# def initiate_robots(n: int, r: int, x: int, y: int) -> List["Robot"]:
# return [Robot(x, y, r) for _ in range(n)]

def initiate_robots(
    n: int,
    r: int,
    x: int = None,
    y: int = None,
    init_random: bool = False,
    timer: int = 0,
    speed: int = 50,
    capacity: int = 1,
) -> List["Robot"]:
    if init_random is True:
        robots = []
        for _ in range(n):
            x = random.randint(1, 999)
            y = random.randint(1, 999)
            # capacity = np.around(np.random.uniform(0.5,1),1)
            # speed = random.randint(25, 50)
            robots.append(Robot(x=x, y=y, r=r, timer=timer, speed=speed,
                capacity = capacity, noise_lvl=0))
        return robots

    return [Robot(x=x, y=y, r=r, timer=timer, speed=speed, capacity=capacity) for _ in range(n)]


def initiate_tasks(n: int, r: int, cap: int, bounds: Tuple[int]) -> List["Task"]:
    tasks = []
    for _ in range(n):
        x = random.randint(0, bounds[0])
        y = random.randint(0, bounds[1])
        tasks.append(Task(x=x, y=y, r=r, cap=cap))
    return tasks


if __name__ == "__main__":
    from robot import Robot, Task, Edge, Message, Counter
    from visualization import Visualization
    from statistics import Statistics

    sim = Simulator(
        # caption="Simulation",
        # width=1000,
        # height=1000,
        # robots=initiate_robots(n=50, r=0, x=500, y=500, init_random=True,
            # timer=0, speed=50),
        # tasks=initiate_tasks(n=20, r=200, cap=2, bounds=(1000, 1000)),
        # collect_data_rate=50,
        # visualize=True,
        # speed=1,
        # method="broadcast",
        caption="Simulation",
        width=1000,
        height=1000,
        robots=initiate_robots(n=15, r=400, x=500, y=500, init_random=True,
            timer=30, speed=50),
        tasks=initiate_tasks(n=3, r=50, cap=3, bounds=(1000, 1000)),
        collect_data_rate=10,
        visualize=True,
        speed=1,
        method="auction",
        threshold=8,
        task_behavior_threshold=10,
    )
    sim.run()
