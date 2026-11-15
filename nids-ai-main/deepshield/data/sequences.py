"""Per-source-host flow windows for the LSTM (Track B).

The entity is the source IP. Within one chronological block, a host's flows
are ordered by end time (ltime, then original file position); the window
for a flow is that flow plus up to L-1 earlier flows from the same host in
the same block. The model predicts the category of the LAST flow only, so
every flow is a target exactly once and windows never cross block
boundaries (no train/test overlap through windows).

Windows are not materialised: `WindowIndex` stores, for every flow, its
position in host order and the position where its host's run starts, and
`gather` builds a padded batch on demand. This keeps memory at
O(records x features) instead of O(records x L x features).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WindowIndex:
    order: np.ndarray        # record indices sorted by (host, ltime, position)
    run_start: np.ndarray    # for each sorted position, first position of that host
    pos_of: np.ndarray       # record index -> position in `order`

    @classmethod
    def build(cls, host: np.ndarray, ltime: np.ndarray) -> "WindowIndex":
        n = len(host)
        _, host_code = np.unique(host, return_inverse=True)
        order = np.lexsort((np.arange(n), ltime, host_code))
        h = host_code[order]
        new_run = np.r_[True, h[1:] != h[:-1]]
        run_start = np.maximum.accumulate(np.where(new_run, np.arange(n), 0))
        pos_of = np.empty(n, dtype=np.int64)
        pos_of[order] = np.arange(n)
        return cls(order, run_start, pos_of)

    def gather(self, X: np.ndarray, records: np.ndarray, length: int):
        """Left-padded windows (B, L, F) and their true lengths for `records`.

        X is indexed by record index (rows of the block's feature matrix).
        """
        pos = self.pos_of[records]
        offsets = np.arange(-(length - 1), 1)              # L positions ending at pos
        win_pos = pos[:, None] + offsets[None, :]
        valid = win_pos >= self.run_start[pos][:, None]
        win_rec = self.order[np.clip(win_pos, 0, None)]
        out = X[win_rec]
        out[~valid] = 0.0
        return out, valid.sum(axis=1)
