"""Connect to a responsive Meshcat instance, or start a local one."""

import sys

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


def open_viewer(viz):
    """Open the live Meshcat page for a direct Terminal run."""
    if sys.stdin.isatty():
        viz.viewer.open()


def keep_viewer_open():
    """Keep an auto-started Meshcat server alive until the user is done."""
    if sys.stdin.isatty():
        try:
            input("Meshcat is open. Press Enter here when you are done viewing it. ")
        except (EOFError, KeyboardInterrupt):
            print()
