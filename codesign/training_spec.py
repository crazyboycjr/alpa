from dataclasses import dataclass
from typing import Any, Callable, Optional

from serde import serde, field


@serde
@dataclass
class TrainingSpec(object):
    avg_batch_size_per_device: int
    num_iters: int
    global_batch_size: int = field(default_factory=lambda: 0,
                                   skip_if_false=True)
    loss_func: Optional[Callable[...,
                                 Any]] = field(default_factory=lambda: None,
                                               skip=True)
    optim_gen: Optional[Callable[...,
                                 Any]] = field(default_factory=lambda: None,
                                               skip=True)

    def header_csv(self) -> str:
        return ','.join(
            ['global batch size', 'avg batch size per gpu', '# iters'])

    def value_csv(self) -> str:
        return ','.join([
            f'{self.global_batch_size}', f'{self.avg_batch_size_per_device}',
            f'{self.num_iters}'
        ])