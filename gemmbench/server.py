import collections
import itertools
import random
import zmq

import gbm

from tqdm import tqdm

WORKERS_PORT = 7178
RESULTS_PORT = 7179


def ready_to_send(work, work_sender):
    return (len(work) > 0) and (work_sender.poll(0, zmq.POLLOUT) == zmq.POLLOUT)


def ready_to_recv(results_receiver):
    return results_receiver.poll(0, zmq.POLLIN) == zmq.POLLIN


def run_problems(problems, solutions, configurations, shuffle=True, repeat=10):
    """Run GEMM problems!"""

    # create a work deque where the configuration is grouped (and slow
    # moving).  within each group, gemms are shuffled.
    work = collections.deque()
    for configuration in configurations:
        w = []
        for problem in problems:
            w.extend([(problem, solution, configuration) for solution in solutions] * repeat)
        if shuffle:
            random.shuffle(w)
        work.extend(w)

    num_work_items = len(work)
    num_completed_work_items = 0

    pbar = tqdm(desc="GEMM runs", total=num_work_items)

    with zmq.Context() as context:
        work_sender = context.socket(zmq.PUSH)
        work_sender.bind(f"tcp://*:{WORKERS_PORT}")

        results_receiver = context.socket(zmq.PULL)
        results_receiver.bind(f"tcp://*:{RESULTS_PORT}")

        poller = zmq.Poller()
        poller.register(work_sender, zmq.POLLOUT)
        poller.register(results_receiver, zmq.POLLIN)

        work_serial_number = 0
        submitted_work = {}

        while poller.poll():
            # send work
            while ready_to_send(work, work_sender):
                problem, solution, configuration = work.popleft()
                work_sender.send_multipart(
                    [
                        work_serial_number.to_bytes(4, "little"),
                        problem.serialize(),
                        solution.serialize(),
                        configuration.serialize(),
                    ]
                )

                submitted_work[work_serial_number] = (problem, solution, configuration)
                work_serial_number += 1

            # read results
            while ready_to_recv(results_receiver):
                message = results_receiver.recv_multipart(copy=True)
                serial = int.from_bytes(message[0], "little")
                result = gbm.deserialize_result(message[1])
                result["sclk"] = gbm.float1d(message[2])
                result["mclk"] = gbm.float1d(message[3])
                result["temperature"] = gbm.float1d(message[4])
                result["power"] = gbm.float1d(message[5])

                problem, solution, configuration = submitted_work[serial]
                yield problem.to_dict(), solution.to_dict(), configuration.to_dict(), result

                num_completed_work_items += 1

                pbar.update(1)

                if num_completed_work_items >= num_work_items:
                    break

            if num_completed_work_items >= num_work_items:
                break

        results_receiver.close()
        work_sender.close()

    pbar.close()
