"""Connect to a responsive Meshcat instance, or start a local one."""

import zmq
from config import MESHCAT_URL
from tools import setupwithmeshcat as _setup


def setupwithmeshcat():
    # Meshcat normally waits forever on an absent ZMQ server. A short probe
    # allows a fresh notebook to recover without leaving its cell stuck.
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    try:
        socket.connect(MESHCAT_URL)
        socket.send(b'url')
        ready = bool(socket.poll(500))
        if ready:
            socket.recv()
    finally:
        socket.close()
        context.term()
    return _setup(url=MESHCAT_URL if ready else None)
