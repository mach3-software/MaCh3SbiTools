'''
Runs MCMC with the MaCh3 simulator
'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from tqdm.auto import tqdm

from mach3sbitools.simulator import Simulator, Prior
from mach3sbitools.utils import TorchDeviceHandler


@dataclass
class ChainState:
    '''
    All mutable state for a running chain. Constructed once the chain has
    been initialised, so none of these fields need to be Optional elsewhere.
    '''
    current_step:    Tensor  # (n_chains, n_pars)
    adaptive_mean:   Tensor  # (n_chains, n_pars)
    adaptive_matrix: Tensor  # (n_chains, n_pars, n_pars)
    curr_prob:       Tensor  # (n_chains,)
    n_accepted:      Tensor  # (n_chains,)
    adapt_counter: int = 0
    total_steps:   int = 0

    @property
    def n_chains(self) -> int:
        return self.current_step.shape[0]

    @property
    def n_pars(self) -> int:
        return self.current_step.shape[1]


@dataclass
class ParquetSink:
    '''
    Buffers (step, params, logl) tuples in memory and flushes them to a
    parquet file as row groups once the buffer fills, keeping peak RAM
    usage bounded regardless of chain length.
    '''
    par_names:   list[str]
    buffer_size: int
    writer:      pq.ParquetWriter
    schema:      pa.Schema
    buffer: list[tuple[int, Tensor, Tensor]] = field(default_factory=list)

    def append(self, step: int, pars: Tensor, logl: Tensor) -> None:
        self.buffer.append((step, pars.clone(), logl.clone()))
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        n_chains = self.buffer[0][1].shape[0]

        steps: npt.NDArray[np.int64] = np.repeat(
            [s for s, _, _ in self.buffer], n_chains
        )
        chains: npt.NDArray[np.int64] = np.tile(
            np.arange(n_chains), len(self.buffer)
        )

        pars = torch.cat([p for _, p, _ in self.buffer], dim=0).cpu().numpy()
        logl = torch.cat([l for _, _, l in self.buffer], dim=0).cpu().numpy()

        columns: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]] = {
            "step": steps,
            "chain": chains,
        }
        for i, name in enumerate(self.par_names):
            columns[name] = pars[:, i]
        columns["logl"] = logl

        table = pa.Table.from_pydict(columns, schema=self.schema)
        self.writer.write_table(table)
        self.buffer.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


class SimulatorMCMC:
    def __init__(self, simulator: Simulator) -> None:
        self.simulator = simulator
        
        # just to be easier
        self.prior = simulator.prior

        self._data = simulator.simulator_wrapper.get_data_bins()

        self._step_scale: float = 2.38**2 / self.prior.n_params
        self._eps: float = 1e-6

        # Only exists once run() has initialised the chain.
        self._state: ChainState | None = None

    def _require_state(self) -> ChainState:
        '''Narrows self._state to ChainState for the type checker, or raises.'''
        if self._state is None:
            raise ValueError("Chain has not been initialised - call run() first!")
        return self._state

    def create_asimov_data(self, par_values: list[float] | None) -> None:
        if par_values is None:
            par_values = self.prior.prior_data.nominals.tolist()
        
        self._data = self.simulator.simulator_wrapper.simulate(par_values)

    def get_logl(self, par_values: Tensor) -> Tensor:
        '''
        Negative log-likelihood calculation, vectorised over chains.
        par_values: (n_chains, n_pars)
        returns:    (n_chains,)
        '''

        logl = torch.full((len(par_values),), torch.inf)

        # Now we get the simulator likelihood (sadly not vectorisable)
        for i, p in enumerate(par_values):
            logl[i] = self.simulator.simulator_wrapper.get_log_likelihood(p.tolist())

        return -logl

    def _sample_initial_values(self, n_chains: int) -> Tensor:
        '''
        Draws starting points for each chain from N(nominal_values, covariance_matrix).
        '''
        mean = self.prior.prior_data.nominals
        cov = self.prior.prior_data.covariance_matrix

        dist = torch.distributions.MultivariateNormal(
            loc=mean, covariance_matrix=cov
        )
        return dist.sample((n_chains,)).to(torch.double)


    def _init_state(self, n_chains: int, initial_values: Tensor | None) -> ChainState:
        if initial_values is None:
            initial_values = self._sample_initial_values(n_chains)

        n_pars = initial_values.shape[-1]

        current_step = initial_values.clone()
        adaptive_mean = initial_values.clone()

        adaptive_matrix = (2.38**2 / n_pars) * self.prior.prior_data.covariance_matrix.unsqueeze(0).repeat(n_chains, 1, 1)

        curr_prob = self.get_logl(current_step)
        n_accepted = torch.zeros(n_chains)

        return ChainState(
            current_step=current_step,
            adaptive_mean=adaptive_mean,
            adaptive_matrix=adaptive_matrix,
            curr_prob=curr_prob,
            n_accepted=n_accepted,
        )

    def _propose_step(self, state: ChainState) -> Tensor:
        '''Draws a new proposal per chain from N(current_step, adaptive_matrix).'''
        L = torch.linalg.cholesky(state.adaptive_matrix).to(torch.double)
        z = torch.randn_like(state.current_step).to(torch.double)
        prop = state.current_step + torch.einsum('cij,cj->ci', L, z)
        
        # Flip parameteres
        prop[:, self.prior._flipped_mask] *= (torch.randint(0, 2, prop[:, self.prior._flipped_mask].shape,
                                                            device=prop.device) * 2 - 1)
        
        # Cyclical parameters
        sub = prop[:, self.prior.cyclical_mask]
        prop[:, self.prior.cyclical_mask] = (sub + torch.pi) % (2 * torch.pi) - torch.pi

        return prop

    def _accept_step(self, state: ChainState, par_values: Tensor) -> Tensor:
        '''Metropolis hastings accepted/reject condition. Mutates state in place.'''

        acc_prob = self.get_logl(par_values)
        n_chains = len(acc_prob)

        rand = torch.rand((n_chains,))

        alpha = state.curr_prob - acc_prob
        accepted = alpha > torch.log(rand)

        state.current_step[accepted] = par_values[accepted]
        state.curr_prob[accepted] = acc_prob[accepted]
        state.n_accepted[accepted] += 1

        return accepted

    def _adaptive_step(self, state: ChainState, step_number: int) -> None:
        '''
        Haario adaptive covariance update, vectorised over chains.
        Mutates state.adaptive_mean and state.adaptive_matrix in place.
        '''

        n_pars = state.n_pars
        I_d = torch.eye(
            n_pars, device=state.adaptive_mean.device, dtype=state.adaptive_mean.dtype
        )

        # t * Xbar_{t-1} Xbar_{t-1}^T
        factor = step_number * torch.einsum(
            'ci,cj->cij', state.adaptive_mean, state.adaptive_mean
        )

        # Xbar_t = Xbar_{t-1} + (X_t - Xbar_{t-1}) / t
        state.adaptive_mean = state.adaptive_mean + (
            state.current_step - state.adaptive_mean
        ) / step_number

        # subtract (t+1) * Xbar_t Xbar_t^T
        factor -= (step_number + 1) * torch.einsum(
            'ci,cj->cij', state.adaptive_mean, state.adaptive_mean
        )

        # add X_t X_t^T
        factor += torch.einsum('ci,cj->cij', state.current_step, state.current_step)

        # regularisation term
        factor += self._eps * I_d

        state.adaptive_matrix = (
            (step_number - 1) / step_number * state.adaptive_matrix
            + (self._step_scale / step_number) * factor
        )

    def _open_parquet(
        self, outfile: str, par_names: list[str], buffer_size: int
    ) -> ParquetSink:
        schema = pa.schema(
            [("step", pa.int64()), ("chain", pa.int64())]
            + [(name, pa.float64()) for name in par_names]
            + [("logl", pa.float64())]
        )
        writer = pq.ParquetWriter(outfile, schema)
        return ParquetSink(
            par_names=par_names, buffer_size=buffer_size, writer=writer, schema=schema
        )

    def run(
        self,
        n_steps: int,
        n_chains: int,
        adapt_start: int,
        outfile: Path,
        buffer_size: int = 1000,
        adapt_every: int = 1,
        initial_values: Tensor | None = None,
        par_names: list[str] | None = None,
    ) -> ChainState:
        '''
        Initialises, runs, and streams the full chain to a parquet file in one call.

        Returns the final ChainState (e.g. for inspecting acceptance rates).
        '''

        state = self._init_state(n_chains, initial_values)
        self._state = state

        if par_names is None:
            par_names = [f"par_{i}" for i in range(state.n_pars)]
        elif len(par_names) != state.n_pars:
            raise ValueError(
                f"par_names has length {len(par_names)}, expected {state.n_pars}"
            )

        sink = self._open_parquet(outfile, par_names, buffer_size)
        
        state.curr_prob = self.get_logl(state.current_step)

        try:
            for step in tqdm(range(1, n_steps + 1), desc="Running MCMC"):
                proposal = self._propose_step(state)
                self._accept_step(state, proposal)

                if step >= adapt_start and step % adapt_every == 0:
                    state.adapt_counter += 1
                    self._adaptive_step(state, state.adapt_counter)

                state.total_steps += 1
                sink.append(state.total_steps, state.current_step, state.curr_prob)
        finally:
            sink.close()

        return state

    @property
    def acceptance_rate(self) -> Tensor:
        '''Per-chain acceptance rate over the whole run so far.'''
        state = self._require_state()
        if state.total_steps == 0:
            raise ValueError("No steps have been run yet!")
        return state.n_accepted / state.total_steps