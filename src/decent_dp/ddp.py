import time
import os
import copy
import math

# from loguru import logger
from functools import partial
from typing import Callable, Iterator, List, Optional, Tuple, cast
import torch
from torch import Tensor
from torch.nn import Module
import torch.distributed as dist
from torch.optim import Optimizer
from torch.distributed import Work
from torch import GradScaler
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from torch.optim.lr_scheduler import LRScheduler
from .topo import TopologyReg, Topology
from .jw_utils import *

"""Data type for the optimizer function"""
OPTIM_FN_TYPE = Callable[[List[Tuple[str, Tensor]]], Optimizer]


"""Data type for the learning rate scheduler function"""
LR_SCHEDULER_FN_TYPE = Callable[[Optimizer], LRScheduler]


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1)) ** 0.5
    return update


class DecentralizedDataParallel(Module):
    """Decentralized data parallel wrapper for PyTorch module

    1. The wrapper places hooks during the backward pass to trace the order of used parameters in the first iteration, and \
    2. Split the parameters into buckets and create optimizers and LR schedulers for each bucket, \
        Add hooks on the last parameter of each bucket to perform the bucket-wise update and communication, \
    3. During the backward passes in the training loop, the hooks are triggered to perform the bucket-wise update and communication

    :Warning: The wrapper currently does not support "channels_last" memory format

    :Warning: The wrapper assumes that the parameter will only be used once in the backward pass

    Args:
        model (Module): PyTorch module to be wrapped
        optim_fn (OPTIM_FN_TYPE): Function to create the optimizer, which takes a list of tuples of parameters and their names
        lr_scheduler_fn (Optional[LR_SCHEDULER_FN_TYPE], optional): Function to create the learning rate scheduler, \
            which takes the optimizer as input. Defaults to None.
        topology (str, optional): Topology of the decentralized communication graph. Defaults to 'complete'.
        scaler (Optional[GradScaler], optional): Gradient scaler for mixed precision training. Defaults to None.
        grad_clip_norm (float, optional): Gradient clipping norm, set to 0.0 if no gradient clipping is applied. Defaults to 0.0.
        param_as_bucket_view (bool, optional): Whether to use the parameter as a view of part of the contiguous buffer. Defaults to True.
        sync_buffer_in_global_avg (bool, optional): Whether to synchronize the float buffers in the global average. Defaults to False.
        bucket_size_in_mb (int, optional): Size of the bucket in MB. Defaults to 25 MB.
        local_world_size (Optional[int], optional): Provide the local world size if not using the environment variable. Defaults to None.
    """

    """Buffer data types that need to be synchronized in global average"""
    FLOAT_DTYPES = [torch.float16, torch.float32, torch.float64]

    def __init__(
        self,
        model: Module,
        optim_fn: OPTIM_FN_TYPE,
        lr_scheduler_fn: Optional[LR_SCHEDULER_FN_TYPE] = None,
        topology: str = "complete",
        scaler: Optional[GradScaler] = None,
        grad_clip_norm: float = 0.0,
        param_as_bucket_view: bool = True,
        sync_buffer_in_global_avg: bool = False,
        bucket_size_in_mb: int = 25,
        _local_world_size: Optional[int] = None,
        toy_test=False,
    ):

        super(DecentralizedDataParallel, self).__init__()
        assert (
            dist.is_available() and dist.is_initialized()
        ), "Distributed environment is not initialized"

        self._model = model
        self._model = model.cuda() if torch.cuda.is_available() else model
        self._optim_fn = optim_fn
        self._lr_schd_fn = lr_scheduler_fn
        self._scaler = scaler
        self._grad_clip_norm = grad_clip_norm
        self._param_as_bucket_view = param_as_bucket_view
        self._sync_buffer_in_global_avg = sync_buffer_in_global_avg
        self._bucket_size = bucket_size_in_mb * 1024 * 1024
        self._local_world_size = (
            _local_world_size
            if _local_world_size is not None
            else int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        )

        # get the rank and world size
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()

        # check if the model is with "channels_last" memory format
        if self._check_channels_last():
            if self._rank == 0:
                logger.debug(f'The model is with "channels_last" memory format')

        if self._rank == 0:
            logger.debug(f"Initializing Decentralized Data Parallel")
            logger.debug(
                f"Rank: {self._rank}, Local World Size: {self._local_world_size}, World Size: {self._world_size}, Topology: {topology}"
            )

        # model parameters
        # self._params: List[Tensor] = list([x for _, x in self._model.named_parameters() if x.requires_grad])
        # self._param_names: List[str] = list([n for n, x in self._model.named_parameters() if x.requires_grad])

        logger.debug("start")
        (
            self._params_muon,
            self._params_adamw,
            self._params_muon_names,
            self._params_adamw_names,
        ) = split_params_by_module_type(model)
        self._params_muon_objId = [id(ele) for ele in self._params_muon]
        self._params_adamw_objId = [id(ele) for ele in self._params_adamw]
        logger.debug(f"{len(self._params_muon)}, {len(self._params_adamw)}")
        logger.debug("end")

        self._params = self._params_muon + self._params_adamw
        self._params_names = self._params_muon_names + self._params_adamw_names

        # trace hooks and traced parameter ids
        self._trace_hooks: List[RemovableHandle] = []
        # self._traced_param_ids: List[int] = []
        self._traced_param_ids_muon: List[int] = []
        self._traced_param_ids_adamw: List[int] = []

        self._step: int = 0
        self._comm_ops: List[Optional[Work]] = []

        self._ddp_hooks: List[RemovableHandle] = []
        self._param_buckets: List[List[Tensor]] = []
        self._param_blocks: List[Tensor] = []
        self._comm_buffers: List[List[Tensor]] = []
        self._comm_blocks: List[Tensor] = []

        # Optimizer and LR scheduler
        self._optims: List[Optimizer] = []
        self._lr_schedulers: List[Optional[LRScheduler]] = []

        # initialize the topology
        self._topo: Topology = TopologyReg.registry[topology](self._local_world_size)

        # create hooks to trace the used parameters in backward
        self._create_trace_hooks()

        # sync the parameters at the start
        self._sync_at_start()

        # flag for gradient accumulation
        self._is_grad_accum_enable: bool = False

        # flag for initializing the parameters
        self._initialized: bool = False
        self.theta = 0.2
        self.toy_test = toy_test
        self.inf_grad = False

    def zeropower_via_newtonschulz5(self, G, steps: int):
        """
        Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
        quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
        of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
        zero even beyond the point where the iteration no longer converges all the way to one everywhere
        on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
        where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
        performance at all relative to UV^T, where USV^T = G is the SVD.
        """
        assert (
            G.ndim >= 2
        )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
        a, b, c = (3.4445, -4.7750, 2.0315)
        #        logger.debug(f'{G.dtype}')
        X = G.bfloat16()
        #        logger.debug(f'{X.dtype, X}')
        if G.size(-2) > G.size(-1):
            X = X.mT

        #        logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}')

        # Ensure spectral norm is at most 1
        #        logger.debug(f'{X.norm(dim=(-2, -1), keepdim=True)}')
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        #        logger.debug(f'{X}')
        # Perform the NS iterations
        for _ in range(steps):
            A = X @ X.mT
            #            logger.debug(f'{A}')
            B = (
                b * A + c * A @ A
            )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            #            logger.debug(f'{B}')
            X = a * X + B @ X
        #            logger.debug(f'{X}')

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    def _check_channels_last(self) -> bool:
        """Check if the model is with "channels_last" memory format

        Returns:
            bool: True if the model is with "channels_last" memory format
        """
        if any(
            [
                x.is_contiguous(memory_format=torch.channels_last)
                and (not x.is_contiguous())
                for x in self._model.parameters()
                if len(x.shape) == 4
            ]
        ):
            return True
        return False

    def _create_trace_hooks(self):
        """Create hooks to trace the order of used parameters in backward pass"""
        # [pid for pid, param in enumerate(self._params)] is the same at different nodes, but this list is in order of parameter registration. The parameter order of doing backward is better, because it's faster. We group parameters who get their gradients fast and then synchronize them among nodes. If we use registration order, then first few parameters may not get gradients first, which is not efficient.
        for pid, param in enumerate(self._params):
            #            param.register_post_accumulate_grad_hook(
            #                    self.check_grad_hook)

            self._trace_hooks.append(
                param.register_post_accumulate_grad_hook(
                    partial(lambda data, pid: self._trace_fn(data, pid), pid=pid)
                )
            )  # after gradients accumulate and before optimizer.step()

    @torch.no_grad()
    def _sync_at_start(self):
        """Broadcast the parameters of worker 0 to all other workers at the start"""
        for param in self._params:
            dist.broadcast(param, 0)

    def set_accumulate_grad(self, enable: bool = True):
        """Set the gradient accumulation mode

        Args:
            enable (bool, optional): Whether to accumulate the gradients. Defaults to True.
        """
        self._is_grad_accum_enable = enable

    """Hook functions"""

    @torch.no_grad()
    def _trace_fn(self, _: Tensor, pid: int):
        """Hook function to trace the order of used parameters in backward pass

        Args:
            _ (Tensor): corresponding tensor (not used)
            pid (int): parameter id

        Raises:
            AssertionError: The parameter is used more than once in the backward pass
        """
        if self._is_grad_accum_enable:
            return
        assert not (
            pid in self._traced_param_ids_muon + self._traced_param_ids_adamw
        ), "The parameter is used more than once in the backward pass"
        #        logger.debug(f'{pid, type(self._params[pid]), self._params[pid].shape, type(self._params_muon)}')
        if id(self._params[pid]) in self._params_muon_objId:
            self._traced_param_ids_muon.append(pid)
        elif id(self._params[pid]) in self._params_adamw_objId:
            self._traced_param_ids_adamw.append(pid)
        else:
            logger.debug("error")
            1 / 0

    @torch.no_grad()
    def _ddp_fn(self, param: Tensor, bucket_id: int):
        1 / 0
        """Hook function to perform the bucket-wise update and communication

        Args:
            _ (Tensor): corresponding tensor (not used)
            bucket_id (int): bucket id
        """

        # skip the update and communication if the model is accumulating gradients
        if self._is_grad_accum_enable:
            return
        if param is self._param_buckets_adamw[bucket_id][-1]:
            _param_buckets = self._param_buckets_adamw
        elif param is self._param_buckets_muon[bucket_id][-1]:
            _param_buckets = self._param_buckets_muon
            _param_blocks -= msgn(self.V_blocks[bucket_id])
            _comm_blocks_param.copy_(_param_blocks)
            _comm_blocks_param.mul_((1 - weight) / (len(edge.ranks) - 1))
            dist.all_reduce(
                _comm_blocks_param,
                op=dist.ReduceOp.SUM,
                group=edge.group,
                async_op=True,
            )
            _param_blocks.mul_(weight - (1 - weight) / (len(edge.ranks) - 1))
            _param_blocks.add_(_comm_blocks_param[bucket_id])

        else:
            logger.debug("error")
            1 / 0

            # TODO: update V

        # perform the bucket-wise update and communication when all gradients in the bucket are accumulated
        comm_op = self._comm_ops[bucket_id]
        if comm_op is not None:
            # wait for the communication from the last iteration
            comm_op.wait()
            self._comm_ops[bucket_id] = None

            # get the peers to communicate with in this iteration
            edge = self._topo.get_edge(self._step)
            weight = edge.weight
            # logger.info(f'{weight}') # When fixed i, if w_{ij} are the same value for all j, then weight can be a scaler.

            # optionally call the pre_average_hook for optimizers using the communication information
            if hasattr(self._optims[bucket_id], "pre_average_hook"):
                self._optims[bucket_id].pre_average_hook(edge, weight)  # type: ignore

            # replace the local model with the mixed model
            # the following should be the consensus step
            if self._param_as_bucket_view:  # true
                self._param_blocks[bucket_id].mul_(
                    weight - (1 - weight) / (len(edge.ranks) - 1)
                )
                self._param_blocks[bucket_id].add_(
                    self._comm_blocks[bucket_id]
                )  # this step change model weights
                # note that the above mixing steps are triggered after the gradient is calculated, that means gradient is calculated with respect to param before mixing. treat this as the mixing step of first iteration and then do optim.step, then it aligns with the algorithm in paper.
            else:
                torch._foreach_mul_(
                    _param_buckets[bucket_id],
                    weight - (1 - weight) / (len(edge.ranks) - 1),
                )
                torch._foreach_add_(
                    _param_buckets[bucket_id], self._comm_buffers[bucket_id]
                )

        # perform local update
        if self._scaler:
            if self._grad_clip_norm > 0:
                self._scaler.unscale_(self._optims[bucket_id])
                torch.nn.utils.clip_grad_norm_(
                    _param_buckets[bucket_id], self._grad_clip_norm
                )
            self._scaler.step(self._optims[bucket_id])
            if bucket_id == len(_param_buckets) - 1:
                logger.debug(f"{self._scaler.get_scale()}")
                self._scaler.update()
                logger.debug(f"{self._scaler.get_scale()}")
        else:
            if self._grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    _param_buckets[bucket_id], self._grad_clip_norm
                )
            self._optims[bucket_id].step()
        self._optims[bucket_id].zero_grad()

        if self._lr_schedulers[bucket_id] is not None:
            scheduler = cast(LRScheduler, self._lr_schedulers[bucket_id])
            scheduler.step()

        # launch the next communication after updating the weights. Last iteration ends here and next iteration starts here.
        if self._param_as_bucket_view:  # true
            self._comm_blocks[bucket_id].copy_(self._param_blocks[bucket_id])
        else:
            torch._foreach_copy_(
                self._comm_buffers[bucket_id], _param_buckets[bucket_id]
            )

        edge = self._topo.get_edge(self._step + 1)
        weight = edge.weight
        self._comm_blocks[bucket_id].mul_((1 - weight) / (len(edge.ranks) - 1))

        self._comm_ops[bucket_id] = dist.all_reduce(
            self._comm_blocks[bucket_id],
            op=dist.ReduceOp.SUM,
            group=edge.group,
            async_op=True,
        )  # this step doesn't change model weights

        # for consensus error, to be done
        # g = dist.group.WORLD
        # N = dist.get_world_size(g)

        # sum_x = self._param_blocks[bucket_id].clone()
        # dist.all_reduce(sum_x, op=dist.ReduceOp.SUM, group=g)
        # xbar = sum_x / N
        # (self._param_blocks[bucket_id]-xbar)**2

    @torch.no_grad()
    def _initialize_params(self):
        """Initialize the parameter buckets and communication buffers

        Raises:
            RuntimeError: Number/Order of elements in used parameters is different on different nodes
        """

        # verify the number of elements and the order of the parameters on different nodes are the same
        verify = [[(i, self._params[i].numel()) for i in self._traced_param_ids_muon]]
        result = [[(0, 0)]] if self._rank != 0 else verify
        dist.broadcast_object_list(result, src=0)
        if not all([x == y for x, y in zip(verify[0], result[0])]):
            raise RuntimeError(
                "Number/Order of elements in used parameters is different on different nodes"
            )

        verify = [[(i, self._params[i].numel()) for i in self._traced_param_ids_adamw]]
        result = [[(0, 0)]] if self._rank != 0 else verify
        dist.broadcast_object_list(result, src=0)
        if not all([x == y for x, y in zip(verify[0], result[0])]):
            raise RuntimeError(
                "Number/Order of elements in used parameters is different on different nodes"
            )

        # remove the trace hooks because they only need to execute once and already execute once when the first time the gradient is calculated
        for hook in self._trace_hooks:
            hook.remove()
        del self._trace_hooks

        # split the parameters into roughly equal-size buckets, and register hooks on the last parameter of each bucket
        self._param_buckets_muon, self._grad_buckets_muon = self.create_buckets(
            self._traced_param_ids_muon,
        )

        self._param_buckets_adamw, self._grad_buckets_adamw = self.create_buckets(
            self._traced_param_ids_adamw,
        )

        self._param_buckets = self._param_buckets_muon + self._param_buckets_adamw
        self._param_blocks_comm, self._param_blocks = self.create_block(
            self._param_buckets
        )
        self._M_blocks, self._grad_blocks = self.create_block(
            self._grad_buckets_muon + self._grad_buckets_adamw
        )
        if self.toy_test:
            logger.debug(
                f"{self._param_buckets_muon, self._grad_buckets_muon, self._param_blocks_comm, self._param_blocks, self._M_blocks, self._grad_blocks}"
            )
        (
            self._V_blocks,
            self._V_blocks_old,
            self._V_blocks_comm,
            self._comm_op_V,
            self._comm_op,
        ) = [list() for _ in range(5)]
        for ele in self._M_blocks:
            self._V_blocks.append(ele.clone())
            self._V_blocks_old.append(ele.clone())
            self._V_blocks_comm.append(ele.clone())
            self._comm_op_V.append(None)
            self._comm_op.append(None)
        # self._comm_blocks_adamw, self._blocks_adamw = self.create_block(self._param_buckets_adamw)

        self._comm_ops = [None] * len(self._param_buckets)

    def _align(self, size: int):
        """Align the size to 128-byte boundary"""
        return math.ceil(size / 32) * 32

    #    def check_grad_hook(self, inp):
    #        if torch.isinf(inp.grad).any():
    #            self.inf_grad = True

    def create_buckets(self, inp_traced_param_ids):
        start = 0
        size = 0
        _param_buckets, _grad_buckets = list(), list()
        for i in range(len(inp_traced_param_ids)):
            size += (
                self._align(self._params[inp_traced_param_ids[i]].numel())
                * self._params[inp_traced_param_ids[i]].element_size()
            )
            if (size >= self._bucket_size) or (i == len(inp_traced_param_ids) - 1):
                # register hooks on the last parameter of each bucket, passing the bucket id
                # self._ddp_hooks.append(
                #     self._params[inp_traced_param_ids[i]].register_post_accumulate_grad_hook(
                #         partial(
                #             lambda data, bucket_id: self._ddp_fn(data, bucket_id),
                #             bucket_id=len(self._ddp_hooks)
                #         )
                #     )
                # )
                tmp1, tmp2 = list(), list()
                for j in inp_traced_param_ids[start : i + 1]:
                    tmp1.append(self._params[j])
                    X = self._params[j].grad
                    if torch.isnan(X).any():
                        logger.debug("error")
                        1 / 0
                    if torch.isinf(X).any():
                        logger.debug(f"{self._params_names[j]} grad inf")
                        1 / 0
                    tmp2.append(X)
                _param_buckets.append(tmp1)

                _grad_buckets.append(tmp2)

                # the element of _param_buckets is a list whose element is one element from self._params
                # param_names = [self._param_names[j] for j in inp_traced_param_ids[start:i+1]]

                # create optimizer and learning rate scheduler for parameters in each bucket
                #                self._optims.append(self._optim_fn(list(zip(param_names, _param_buckets[-1]))))
                self._optims.append(self._optim_fn(_param_buckets[-1]))
                self._lr_schedulers.append(
                    self._lr_schd_fn(self._optims[-1])
                    if self._lr_schd_fn is not None
                    else None
                )
                size = 0
                start = i + 1

        return _param_buckets, _grad_buckets

    """Delegation functions"""

    def train(self, mode: bool = True):
        """Set the module in training mode

        Args:
            mode (bool, optional): Whether to set the module in training mode. Defaults to True.
        """
        self._model.train(mode)
        return self

    def eval(self):
        """Set the module in evaluation mode"""
        self._model.eval()
        return self

    def forward(self, *args, **kwargs):
        """Forward pass of the model"""
        # lazy initialization at the second iteration
        if self._step >= 1:
            if self._step == 1:
                # initialize the parameters and communication buffers
                start = time.time()
                self._initialize_params()
                logger.debug(f"initialize time cost {time.time() - start}")

            self.eta = self._lr_schedulers[0].get_last_lr()[0]
            if self._step <= 10:
                logger.debug(f"step: {self._step}, lr: {self.eta}")
            # manually trigger the communications for the first iteration only
            with torch.no_grad():
                t = torch.tensor([0], dtype=torch.int32).cuda()  # int16 doesn't work
                for i in range(len(self._param_buckets)):
                    # optionally call the pre_average_hook for optimizers using the communication information
                    # print(hasattr(self._optims[i], 'pre_average_hook'))
                    # if hasattr(self._optims[i], 'pre_average_hook'):
                    #     self._optims[i].pre_average_hook(edge, weight) # type: ignore

                    # # update parameters and launch the first communication
                    self._scaler.step(self._optims[i])  # do unscale internally

                scale1 = self._scaler.get_scale()
                self._scaler.update()
                scale2 = self._scaler.get_scale()
                if scale2 < scale1:
                    t.fill_(1)
                    logger.debug(f"scale changed from {scale1} to {scale2}")

                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                if t.item() == 0:
                    edge = self._topo.get_edge(self._step)
                    weight = edge.weight

                    for i in range(len(self._param_buckets)):
                        # optionally call the pre_average_hook for optimizers using the communication information
                        # print(hasattr(self._optims[i], 'pre_average_hook'))
                        # if hasattr(self._optims[i], 'pre_average_hook'):
                        #     self._optims[i].pre_average_hook(edge, weight) # type: ignore

                        # # update parameters and launch the first communication
                        if self._scaler:
                            if self._grad_clip_norm > 0:
                                # self._scaler.unscale_(self._optims[i])
                                torch.nn.utils.clip_grad_norm_(
                                    self._param_buckets[i], self._grad_clip_norm
                                )
                            # self._scaler.step(self._optims[i])
                            # if i == len(self._param_buckets) - 1:
                            #     self._scaler.update()
                        #                                logger.debug(f"current scale:{self._scaler.get_scale()}")
                        # TODO: synchronize the scaler state across all workers?
                        else:
                            if self._grad_clip_norm > 0:
                                torch.nn.utils.clip_grad_norm_(
                                    self._param_buckets[i], self._grad_clip_norm
                                )
                        #                        self._optims[i].step()
                        #                    if self._lr_schedulers[i] is not None:
                        #                        scheduler = cast(LRScheduler, self._lr_schedulers[i])
                        #                        scheduler.step()
                        #                    logger.debug(f'{self._M_blocks[i], self._grad_blocks[i]}')
                        new_M = (1 - self.theta) * self._M_blocks[
                            i
                        ] + self.theta * self._grad_blocks[i]
                        #                    X = self._grad_blocks[i]
                        #                    logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}', rank=[0,1])
                        #                    X = self._M_blocks[i]
                        #                    logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}', rank=[0,1])
                        #                    X = new_M
                        #                    logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}', rank=[0,1])
                        if self.toy_test:
                            logger.debug(f"new_M {new_M}, {new_M[0]}")
                            tmp = new_M[0].detach().item()
                            if self._step == 1:
                                if self._rank == 0:
                                    assert f"{tmp:.2f}" == "0.02"
                                if self._rank == 1:
                                    assert f"{tmp:.2f}" == "0.04"
                            if self._step == 2:
                                if self._rank == 0:
                                    assert f"{tmp:.3f}" == "0.056"
                                if self._rank == 1:
                                    assert f"{tmp:.3f}" == "0.092"
                                    k

                        tmp = self._V_blocks[i] + new_M - self._M_blocks[i]
                        self._M_blocks[i].copy_(new_M)
                        #                    X = tmp
                        #                    logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}', rank=[0,1])
                        self._V_blocks_comm[i].copy_(tmp)
                        self._V_blocks[i].copy_(tmp)
                        self._V_blocks_comm[i].mul_(
                            (1 - weight) / (len(edge.ranks) - 1)
                        )
                        #                    X = self._V_blocks_comm[i]
                        #                    logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}', rank=[0,1])

                        self._comm_op_V[i] = dist.all_reduce(
                            self._V_blocks_comm[i],
                            op=dist.ReduceOp.SUM,
                            group=edge.group,
                            async_op=True,
                        )
                    for i in range(len(self._param_buckets)):
                        self._comm_op_V[i].wait()
                        self._V_blocks[i].mul_(
                            weight - (1 - weight) / (len(edge.ranks) - 1)
                        )
                        self._V_blocks[i].add_(self._V_blocks_comm[i])

                        #                    X =  self._V_blocks[i]
                        #                    logger.debug(f'{torch.isnan(X).any(), torch.isinf(X).any()}')
                        if self.toy_test:
                            logger.debug(f"{self._V_blocks[i]}")
                            logger.debug(f"{self._grad_blocks[i]}")
                        self._grad_blocks[i].copy_(self._V_blocks[i])
                        if self._step <= 2 and i == 0:
                            tmp1 = self._grad_blocks[i].is_contiguous()
                            tmp2 = self._grad_blocks[i].is_contiguous(
                                memory_format=torch.channels_last
                            )
                            logger.debug(f"{self._grad_blocks[i]}")
                            logger.debug(f"{tmp1, tmp2}")
                        for ind, ele in enumerate(self._param_buckets[i]):
                            update = ele.grad
                            if self._step <= 2 and i == 0 and ind == 0:
                                tmp1 = update.is_contiguous()
                                tmp2 = update.is_contiguous(
                                    memory_format=torch.channels_last
                                )
                                logger.debug(f"{tmp1, tmp2}")
                            #                            logger.debug(f'{update}')
                            if self.toy_test:
                                ele -= self.eta * update
                            else:
                                #                            if i < len(self._param_buckets_muon):
                                if i<len(self._param_buckets_muon):
                                    if update.ndim >= 3:  # for the case of conv filters
                                        update = update.reshape(len(update), -1)
                                    update = self.zeropower_via_newtonschulz5(update, 5)
                                    update *= (
                                        max(1, ele.grad.size(-2) / ele.grad.size(-1))
                                        ** 0.5
                                    )
                                    update = update.reshape(ele.shape)
                                #                                    ele.mul_(1-lr*weigt_decay)
                                #                                ele.add_(update, alpha=-self.eta / math.sqrt(self._step))
                                    ele.add_(update, alpha=-1e-1)
                                else:
                                    ele.add_(update, alpha=-self.eta)
                        self._param_blocks_comm[i].copy_(self._param_blocks[i])
                        self._param_blocks_comm[i].mul_(
                            (1 - weight) / (len(edge.ranks) - 1)
                        )
                        self._comm_op[i] = dist.all_reduce(
                            self._param_blocks_comm[i],
                            op=dist.ReduceOp.SUM,
                            group=edge.group,
                            async_op=True,
                        )

                    self._lr_schedulers[0].step()
                    start = time.time()
                    for i in range(len(self._param_buckets)):
                        self._comm_op[i].wait()
                        self._param_blocks[i].mul_(
                            weight - (1 - weight) / (len(edge.ranks) - 1)
                        )
                        self._param_blocks[i].add_(self._param_blocks_comm[i])
                if self._step <= 10:
                    tmp = self._param_buckets[0][0]
                    logger.debug(f"{tmp.shape,tmp[0,0,0,1], tmp.grad[0,0,0,1]}")

                for i in range(len(self._param_buckets)):
                    self._optims[i].zero_grad(set_to_none=False)

                # here need to wait because it iterates over all buckets. And when executing self.V_comm_op[0].wait(), it will wait until self.V_comm_op[0] is finished. Note that all operations in self.V_comm_op are still running. The reason wait here is that the following lines require it to finish first.
        if self._model.training and (not self._is_grad_accum_enable):
            self._step += 1

        with torch.autograd.profiler.record_function(
            "DecentralizedDataParallel.forward"
        ):
            output = self._model(*args, **kwargs)
            return output

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Get the parameters of the model

        Args:
            recurse (bool, optional): Whether to get the parameters recursively. Defaults to True.

        Yields:
            Iterator[Parameter]: The iterator of the parameters
        """
        yield from self._model.parameters(recurse)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """Get the named parameters of the model"""
        return super().named_parameters(prefix, recurse, remove_duplicate)

    def create_block(self, inp_buckets):
        _comm_blocks = list()
        blocks_param = list()
        for i in range(len(inp_buckets)):
            total_size = sum([self._align(p.numel()) for p in inp_buckets[i]])

            # make sure the total size is unique for each bucket \
            # (not necessary, but make sure the communication operations are unique for each bucket with negligible overhead)

            size_dict = {}
            while total_size in size_dict:
                total_size += 32
            size_dict[total_size] = True

            # create the communication buffer for each bucket
            _comm_blocks.append(
                torch.zeros(
                    total_size,
                    device=inp_buckets[i][0].device,
                    requires_grad=False,
                    dtype=inp_buckets[i][0].dtype,
                )
            )
            if self._param_as_bucket_view:  # true
                # create contiguous blocks for each bucket, and let the parameters be views of the fragments of the block
                blocks_param.append(
                    torch.zeros(
                        total_size,
                        device=inp_buckets[i][0].device,
                        requires_grad=inp_buckets[i][0].requires_grad,
                        dtype=inp_buckets[i][0].dtype,
                    )
                )

                start = 0
                for j in range(len(inp_buckets[i])):
                    size = inp_buckets[i][j].numel()
                    if (
                        (len(inp_buckets[i][j].shape) == 4)
                        and inp_buckets[i][j].is_contiguous(
                            memory_format=torch.channels_last
                        )
                        and (not inp_buckets[i][j].is_contiguous())
                    ):
                        # permute the tensor to the channels_last format
                        blocks_param[-1].narrow(0, start, size).copy_(
                            inp_buckets[i][j].permute(0, 2, 3, 1).view(-1)
                        )
                        tmp1 = (
                            blocks_param[-1]
                            .narrow(0, start, size)
                            .view(
                                (
                                    inp_buckets[i][j].shape[0],
                                    inp_buckets[i][j].shape[2],
                                    inp_buckets[i][j].shape[3],
                                    inp_buckets[i][j].shape[1],
                                )
                            )
                        )
                        tmp2 = tmp1.permute(0, 3, 1, 2)
                        inp_buckets[i][j].data = tmp2
                        #                        logger.debug(f'{tmp1.is_contiguous(), tmp1.is_contiguous(memory_format=torch.channels_last), tmp2.is_contiguous(), tmp2.is_contiguous(memory_format=torch.channels_last)}')

                        assert inp_buckets[i][j].is_contiguous(
                            memory_format=torch.channels_last
                        )
                        assert not inp_buckets[i][j].is_contiguous()
                    else:
                        # otherwise, copy the tensor directly
                        assert inp_buckets[i][j].is_contiguous()
                        blocks_param[-1].narrow(0, start, size).copy_(
                            inp_buckets[i][j].view(-1)
                        )
                        inp_buckets[i][j].data = (
                            blocks_param[-1]
                            .narrow(0, start, size)
                            .view_as(inp_buckets[i][j])
                        )
                        # same storage but different view
                    start += self._align(size)

            # start = 0
            # self._comm_buffers.append([])
            # for j in range(len(inp_buckets[i])):
            #     size = inp_buckets[i][j].numel()
            #     if (len(inp_buckets[i][j].shape) == 4) and inp_buckets[i][j].is_contiguous(memory_format=torch.channels_last) \
            #         and (not inp_buckets[i][j].is_contiguous()):
            #         # permute the tensor to the channels_last format
            #         self._comm_buffers[-1].append(comm_block.narrow(0, start, size).view(
            #             (inp_buckets[i][j].shape[0],
            #              inp_buckets[i][j].shape[2],
            #              inp_buckets[i][j].shape[3],
            #              inp_buckets[i][j].shape[1])
            #         ).permute(0, 3, 1, 2))
            #     else:
            #         self._comm_buffers[-1].append(comm_block.narrow(0, start, size).view_as(inp_buckets[i][j]))
            #     start += self._align(size)

            #     # attach the communication buffer to the parameter for "pre_average_hook" in the optimizer
            #     if hasattr(self._optims[i], 'pre_average_hook'):
            #         setattr(inp_buckets[i][j], 'comm_buffer', self._comm_buffers[-1][-1])
            # initialize the communication buffer with the initial parameters
            # torch._foreach_copy_(self._comm_buffers[-1], inp_buckets[i])

        return _comm_blocks, blocks_param

    """Utility functions"""

    @torch.no_grad()
    def global_avg(self):
        """Perform global average on the parameters (and buffers if sync_buffer_in_global_avg is True)
        The function is called at the end of the training loop to synchronize the parameters across all nodes for evaluation
        """
        for op in self._comm_ops:
            if op is not None:
                op.wait()
        self._comm_ops = [None for _ in range(len(self._param_buckets))]

        if self._param_as_bucket_view:
            torch._foreach_div_(self._param_blocks, self._world_size)
            for i in range(len(self._param_blocks)):
                dist.all_reduce(self._param_blocks[i], op=dist.ReduceOp.SUM)
        else:
            torch._foreach_div_([x.data for x in self._params], self._world_size)
            for x in self._params:
                dist.all_reduce(x.data, op=dist.ReduceOp.SUM)

        if self._sync_buffer_in_global_avg:
            # globally average the float buffers (e.g. running mean and variance in batch normalization)
            for x in self._model.buffers():
                if x.dtype in self.FLOAT_DTYPES:
                    dist.all_reduce(x.data, op=dist.ReduceOp.SUM)
                    x.data.div_(self._world_size)


__all__ = ["DecentralizedDataParallel", "OPTIM_FN_TYPE", "LR_SCHEDULER_FN_TYPE"]

