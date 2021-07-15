from typing import List


class Statistics:
    def __init__(self):
        self._log_tasks = [[0,0]]
        self._tasks = [[0,0]]
        self._com_units = [[0,0]]
        self._log_com_units = [[0,0]]

    def log_tasks(self, iteration: int, tasks: int) -> None:

        if iteration == 0:
            return

        score = (tasks - self._log_tasks[-1][1]) / (iteration - self._tasks[-1][0])
        self._log_tasks.append([iteration, tasks])
        self._tasks.append([iteration, score])
        # self._tasks.append([iteration, len(tasks) / (iteration+1)])

    def log_com_units(self, iteration: int, com_units: int) -> None:

        if iteration == 0:
            return

        score = (com_units - self._log_com_units[-1][1]) / (iteration -
                self._com_units[-1][0])
        self._log_com_units.append([iteration, com_units])
        self._com_units.append([iteration, score])


