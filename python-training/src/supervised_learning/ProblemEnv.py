import grpc
import queue
from typing import Optional, Iterator

import rl_pb2 as pb
import rl_pb2_grpc as pb_grpc


def request_generator(q: "queue.Queue[Optional[pb.StepRequest]]") -> Iterator[pb.StepRequest]:
    while True:
        item = q.get()
        if item is None:
            return
        yield item


class RemoteProblemsEnv:
    def __init__(self, num_envs: int, host: str):
        self.num_envs = num_envs
        self.channel = grpc.insecure_channel(host)
        self.stub = pb_grpc.ProblemsEnvStub(self.channel)

        self.actions_queue: "queue.Queue[Optional[pb.StepRequest]]" = queue.Queue()
        self.stream = None  # will hold the Step(...) stream iterator
        self.current: Optional[pb.StepResponse] = None
        self._active = False

    def init(self, fen: str) -> pb.StepResponse:
        """
        Initializes a new supervised puzzle episode on the server (unary RPC).
        After this, we open a Step stream for rollout.
        """
        # If a stream is already active, end it cleanly before re-init
        if self._active:
            self.end_episode()

        init_resp = self.stub.Init(pb.InitRequest(num_envs=self.num_envs, fen=fen))
        self.current = init_resp

        # Start the bidirectional Step stream for this episode
        self.actions_queue = queue.Queue()
        self.stream = self.stub.Step(request_generator(self.actions_queue))
        self._active = True

        return init_resp

    def step(self, actions: list[pb.ProtoMove]) -> pb.StepResponse:
        """
        Sends one step to the server and blocks until we receive the next StepResponse.
        actions: length should be num_envs (one move per env).
        """
        if not self._active or self.stream is None:
            raise RuntimeError("Call init(fen) before step().")

        req = pb.StepRequest(num_envs=self.num_envs, action=actions)
        self.actions_queue.put(req)

        # Receive next server response
        try:
            self.current = next(self.stream)
            return self.current
        except StopIteration:
            self._active = False
            raise RuntimeError("Server closed the Step stream.")

    def end_episode(self):
        """
        Ends ONLY the Step stream for the current episode.
        Does not close the channel; you can call init() again afterward.
        """
        if self._active:
            self.actions_queue.put(None)  # ends request_generator
            # Drain/close server side by letting it observe stream end
            self._active = False
            self.stream = None

    def close(self):
        """
        Closes the whole client (optional server Close RPC + close channel).
        """
        # End step stream if running
        self.end_episode()

        # Tell server to close resources if it uses this flag
        try:
            self.stub.Close(pb.CloseRequest(close=True))
        except grpc.RpcError:
            pass

        self.channel.close()

    def reset_with_new_fen(self, fen: str) -> pb.StepResponse:
        """
        Convenience: end current episode and init again with a new puzzle.
        """
        return self.init(fen)
