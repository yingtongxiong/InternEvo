import torch


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls._instance


class ProcessGroupSingleton(Singleton):
    def __init__(self):
        self.ULYSSES_PG = None
        self.RING_PG = None
        self.ALLGATHER_PG = None
        self.P2P_PG = None
        


PROCESS_GROUP = ProcessGroupSingleton()


def set_seq_parallel_pg(sp_ulysses_degree, sp_ring_degree, rank, world_size, use_ulysses_low=True, window_size=1):
    """
    sp_ulysses_degree x sp_ring_degree = seq_parallel_degree
    (ulysses_degree, dp_degree)
    """
    sp_degree = sp_ring_degree * sp_ulysses_degree
    dp_degree = world_size // sp_degree

    assert world_size % sp_degree == 0, f"world_size {world_size} % sp_degree {sp_ulysses_degree} == 0"

    num_ulysses_pgs = sp_ring_degree  # world_size // sp_ulysses_degree
    num_ring_pgs = sp_ulysses_degree  #

    if use_ulysses_low:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(
                    range(
                        i * sp_ulysses_degree + offset,
                        (i + 1) * sp_ulysses_degree + offset,
                    )
                )
                # print(f'ulysses_ranks:{ulysses_ranks}')
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

            for i in range(num_ring_pgs):
                
                assert sp_ring_degree % window_size == 0, "the window size should be divided by the sp_ring_degree."
                window_num = sp_ring_degree // window_size
                
                ring_ranks = list(range(i + offset, sp_degree + offset, num_ring_pgs))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group
                
                
                # ring_ranks = [0, 1, 2, 3, 4, 5, 6, 7]
                # window_size = 2
                # window_num = 4
                # all_gather = [[0,1], [2,3], [4,5], [6,7]]
                # p2p = [[0,2,4,6], [1,3,5,7]]
                
                # create the all-gather process group when using sliding window
                for j in range(window_num):
                    window_ranks = ring_ranks[j * window_size : (j + 1) * window_size]
                    # print(f'ring_ranks:{ring_ranks}')
                    group = torch.distributed.new_group(window_ranks)
                    if rank in window_ranks:
                        all_gather_pg = group
                              
                # create the p2p process group when using sliding window
                for j in range(window_size):
                    p2p_ranks = []
                    for t in range(window_num):
                        p2p_ranks.append(ring_ranks[t * window_size + j])
                    group = torch.distributed.new_group(p2p_ranks)
                    if rank in p2p_ranks:
                        p2p_pg = group


    else:
        for dp_rank in range(dp_degree):
            offset = dp_rank * sp_degree
            for i in range(num_ring_pgs):
                ring_ranks = list(range(i * sp_ring_degree + offset, (i + 1) * sp_ring_degree + offset))
                group = torch.distributed.new_group(ring_ranks)
                if rank in ring_ranks:
                    ring_pg = group

            for i in range(num_ulysses_pgs):
                ulysses_ranks = list(range(i + offset, sp_degree + offset, num_ulysses_pgs))
                group = torch.distributed.new_group(ulysses_ranks)
                if rank in ulysses_ranks:
                    ulyssess_pg = group

    PROCESS_GROUP.ULYSSES_PG = ulyssess_pg
    PROCESS_GROUP.RING_PG = ring_pg
    PROCESS_GROUP.ALLGATHER_PG = all_gather_pg
    PROCESS_GROUP.P2P_PG = p2p_pg
