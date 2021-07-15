import math
import time
from copy import deepcopy
from collections import namedtuple
from typing import Tuple, Optional

import numpy as np
# np.random.seed(0)

# Message = namedtuple("Message", ["type", "data"])
class Message():
    def __init__(self, type, data):
        self.type = type
        self.data = data


class Robot:
    __last_id = 0

    def __init__(self, x: int, y: int, r: int, timer=0, speed=2, threshold=10,
            capacity = 1, noise_lvl = 0) -> None:
        self.id = Robot.__last_id
        Robot.__last_id += 1
        self.capacity = capacity
        self.noise_lvl = noise_lvl
        self.x = x
        self.y = y
        self.set_timer(timer)
        # self.timer_off = timer_off
        self.theta = np.random.random() * 2 * np.pi
        self.r = r
        self.within_task = []
        self.task = None
        self.edges = []
        self.bids = []
        self.org_speed = speed
        self.speed = speed
        self.last_state = "search"
        self.state_count = 0
        self.state = "search"  # awaiting response / solving task
        self.have_bid = False
        self.collisions = [1000000]
        self.threshold = threshold
        self.quorum = 0
        self.completed_tasks = []
        self._task_behavior_counter = 0
        self._task_behavior_threshold = 10

    def __hash__(self):
        return self.id

    def __eq__(self, other: "Robot") -> bool:
        return self.id == other.id

    def __repr__(self):
        return f"r{self.id}"

    def set_timer(self, timer):
        if timer == "dynamic":
            # self.timer = self.r/self.speed
            self.timer = math.ceil(self.r/self.speed) + 1
        else:
            self.timer = timer

    def set_auction_timer(self, timer):
        self.auction_timer = timer

    def set_method(self, method):
        self.method = method
        if self.method == "random":
            self.timer = None

        elif self.method == "call off dynamic" or self.method == "call out dynamic" or self.method == "broadcast dynamic":
            self.set_timer("dynamic")
            # self.timer = "dynamic"

    def set_threshold(self, threshold):
        self.threshold = threshold
        self.com_threshold = 100

    def set_task_behavior_threshold(self, threshold):
        self._task_behavior_threshold = threshold

    def task_behavior_counter(self, reset=False):
        if reset:
            self._task_behavior_counter = 0
        else:
            self._task_behavior_counter += 1

    def _count_state(self):
        if self.state == "search":
            self.state_count = 0
            return

        if self.state == self.last_state:
            self.state_count += 1

        else:
            self.last_state = self.state
            self.state_count = 0

        if self.timer is None and self.method == "random":
            return

        elif self.timer and self.state_count > self.timer and self.method != "auction":
            self.reset()

    def pos(self) -> Tuple[float]:
        return (self.x, self.y)

    def reset(self, rand_angle=True) -> None:
        if rand_angle:
            self.theta += np.pi

        self.within_task = []
        self.task = None
        self.edges = []
        self.bids = []
        self.state = "search"
        self.state_count = 0
        self.speed  = self.org_speed*2
        # self.speed = 8

    def move(self, w: int, h: int) -> None:
        self._count_state()
        # if self.state == "awaiting help" and self.task is None:
            # self.reset()
        # if not self.task:
            # self.theta += np.random.normal() / 10

        if self.state in ["solving task", "awaiting help", "awaiting bids"]:
            return

        x = self.x + self.speed * np.cos(self.theta)
        y = self.y + self.speed * np.sin(self.theta)

        x, y = self._restrict_movement(x, y, w, h, method="random_angle")
        self.x, self.y = x, y
        self.speed = self.org_speed
        # self.speed = 2

    def _restrict_movement(
        self, x: float, y: float, w: int, h: int, method: Optional[str] = None
    ) -> Tuple[float]:

        if method is "random_angle":
            if x < 0:
                self.theta = np.random.normal() * np.pi - np.pi / 2
            elif x > w:
                self.theta = np.random.normal() * np.pi + np.pi / 2
            if y < 0:
                self.theta = np.random.normal() * np.pi + np.pi
            elif y > h:
                self.theta = np.random.normal() * np.pi - np.pi

        x = 0 if x < 0 else w if x > w else x
        y = 0 if y < 0 else h if y > h else y

        return x, y

    def _set_task(self, task: "Task") -> None:
        #print(self, "helping out", task) 
        self.task = deepcopy(task)
        noise = self.noise_lvl * np.random.random(2)
        dx = (task.x + noise[0]) - self.x
        dy = (task.y + noise[1]) - self.y
        self.theta = np.arctan2(dy, dx)
        self.state = "helping out"

    def _select_edge(self, robot: "Robot") -> "Edge":
        for edge in self.edges:
            if edge.r1 is robot or edge.r2 is robot:
                return edge

    def auction_task(self) -> None:
        if not self.state == "awaiting bids":
            if not self.bids:
                return

        if self.state in ["awaiting help", "helping out", "solving task"]:
            return

        if len(self.bids) < self.task.cap-1:
            if self.method == "auction timer":
                #print("awaiting help")
                self.state = "awaiting help"
            else:
                self.reset(rand_angle=False)
            return

        bids = {}
        robot_edge = {}
        for bid in self.bids:
            robot = bid.r1 if bid.r1 != self else bid.r2
            bids[robot] = robot.capacity / (bid.length / self.org_speed)
            robot_edge[robot] = bid

        remaining_help = self.task.cap - self.capacity
        top_n_bids = []
        for robot, score in sorted(bids.items(), key=lambda x: x[1], reverse=True):
            if remaining_help <= 0:
                break

            top_n_bids.append(robot_edge[robot])
            remaining_help -= robot.capacity


        # top_n_bids = sorted(self.bids, key=lambda x: x.length,
                # reverse=True)[-(self.task.cap)+1:]

        
        #print(self, "auction to", [b.r1 if b.r1 != self else b.r2 for b in top_n_bids ], self.task)

        for winner in top_n_bids:
            # if not self.bids:
                # return
            #print(self.id, len(self.bids), self.within_task)
            msg = Message(type="winner", data=[winner, self.task])
            # msg = Message(type="broadcast", data=self.within_task)
            self.send(msg, edge=winner)

        self.state = "awaiting help"

        # self.bids = []

    def announce_task(self):
        if self.state == "helping out":
            return

        msg = Message(type="announcement", data=self)
        [self.send(msg, edge) for edge in self.edges]
        self.state = "awaiting bids"
        #print(self, "awaiting bids", [e.r1 if e.r1 is not self else e.r2 for e in self.edges], self.task)

    # def _cancel_bids(self):
        # #print("cancel")
        # for bid in self.bids:
            # msg = Message(type="reset", data=bid)
            # self.send(message=msg, edge=bid)

    def _task_behavior(self):
        if self._task_behavior_counter > self._task_behavior_threshold:
            return True

        return False

    def quorum_sensing(self, i):
        # if self.state != "search":
            # return


        any_cols = False
        for edge in self.edges:
            if edge.length <= self.com_threshold:
                # self.collisions.append(i)
                any_cols = True
                break

        if not any_cols:
            self.quorum += 1
            return


        self.quorum -= 1 
        if self.quorum < 0:
            self.quorum = 0

        return

        cols = self.collisions[-5:]
        quorum = []


        for c1, c2 in zip(cols[:-1], cols[1:]):
            quorum.append(c2 - c1)

        self.quorum = sum(quorum)/len(quorum)
        # print(self.collisions)
        # print(self.quorum)


    def send(self, message: "Message", edge: Optional["Edge"] = None) -> None:
        if edge:
            edge.send(self, message)
        else:
            for edge in self.edges:
                edge.send(self, message)

    def receive(self, message: "Message") -> None:
        if self.method == "auction" or self.method == "auction timer":
            if message.type == "announcement":
                if self.state in ["awaiting bids","helping out","solving task"]:
                    return

                if self.state == "awaiting help":
                    if message.data.task != self.task:
                        return

                # if self.have_bid:
                    # return

                edge = self._select_edge(message.data)
                msg = Message(type="bid", data=edge)
                self.send(msg, edge)
                self.have_bid = True


            elif message.type == "bid":
                self.bids.append(message.data)

            elif message.type == "winner":
                if self.state in ["helping out", "solving task, awaiting help"]:
                    return

                #print(self, "won!", message.data[1])
                self._set_task(message.data[1])
                self.state = "helping out"

        elif message.type is "broadcast":
            if self.method == "quorum":
                if self.threshold == 0:
                    pass
                elif self.quorum < np.random.uniform(self.threshold):
                    # pass
                # if np.random.uniform(10) < self.threshold:
                    return
                # if self.quorum < self.threshold and self.quorum >= 0:
                    # print(self.quorum)
                    # print(self.id, self.task, self.quorum)
                    # return
            elif self.method == "task behavior":
                if not self._task_behavior():
                    return

            if self.state is "search":
                self._set_task(message.data)
                self.state = "helping out"


class Edge:
    # __com_units = 0

    def __init__(self,  r1: "Robot", r2: "Robot", d: float, counter, noise=0) -> None:
        self.r1 = r1
        self.r2 = r2
        self.length = d
        self.counter = counter
        self.noise = noise

    def send(self, sender: "Robot", message: "Message") -> None:
        if np.random.uniform() < self.noise:
            return

        receiver = self.r1 if sender is not self.r1 else self.r2
        receiver.receive(message)
        self.counter.i += 1
        # Edge.__com_units += 1

    # @classmethod
    # def com_units(cls) -> int:
        # return cls.__com_units


class Task:
    __last_id = 0

    def __init__(self, x: int, y: int, r: int, cap: int) -> None:
        self.id = Task.__last_id
        Task.__last_id += 1

        self.x = x
        self.y = y
        self.r = r
        self.cap = cap

    def __eq__(self, other):
        if not hasattr(other, 'id'):
            return False
        # print(self.id, other.id, self.id == other.id)
        return self.id == other.id
    
    def __repr__(self):
        return f"t{self.id}"

class Counter:
    def __init__(self):
        self.i = 0

