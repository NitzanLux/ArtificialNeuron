import socket
from typing import List
from configuration_factory import config_factory, load_config_file


class TaskSender():
    def __init__(self, resend_number: int,configs:List[str]):
        self.local_address = socket.gethostbyname(socket.gethostname())
        self.resend_number = resend_number


if __name__ == '__main__':
    configurations_to_sand = [
        config_factory(),
        config_factory()
    ]
