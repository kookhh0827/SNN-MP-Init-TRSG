from typing import Optional
from spikingjelly.activation_based.auto_cuda.neuron_kernel import *
from spikingjelly.activation_based.neuron import *
import spikingjelly.activation_based.auto_cuda.base as base
from spikingjelly.activation_based.auto_cuda.cfunction import *

class NeuronFPTTKernel(base.CKernel2D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}',
            reverse=False)
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='x_seq')
        self.add_param(ctype=f'{dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'{dtype} *', cname='h_seq')
        self.add_param(ctype=f'{dtype} *', cname='spike_seq')
        self.add_param(ctype=f'{dtype} *', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    def neuronal_charge(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`H[t] = f(X[t], V[t-1], ...)`.

        This function should define how ``h_seq[t]`` is calculated by ``x_seq[t], v_v_seq[t]`` and other params if
        the neuron needs.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def neuronal_charge(self) -> str:
                # note that v_v_seq[t] is v_seq[t - dt]
                return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)
        """
        return '// neuronal_charge should be defined here!'

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(neuronal_fire(spike='spike_seq[t]', v='h_seq[t]', v_th='v_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_reset='v_reset',
                                    dtype=self.dtype))
        else:
            core_codes.append(
                neuronal_soft_reset(v_next='v_v_seq[t + dt]', h='h_seq[t]', spike='spike_seq[t]', v_th='v_th',
                                    dtype=self.dtype))

        self._core = core_codes.codes
        return self._core


class NeuronBPTTKernel(base.CKernel2D):
    def __init__(self, surrogate_function: Callable, hard_reset: bool, detach_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f'{self.__class__.__name__}_{dtype}_{"hard_reset" if hard_reset else "soft_reset"}_{"detach_reset" if detach_reset else "nodetach_reset"}',
            reverse=True)
        self.surrogate_function = surrogate_function
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.dtype = dtype
        self.add_param(ctype=f'const {dtype} *', cname='grad_spike_seq')
        self.add_param(ctype=f'const {dtype} *', cname='grad_v_seq')
        self.add_param(ctype=f'const {dtype} *', cname='h_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_x_seq')
        self.add_param(ctype=f'{dtype} *', cname='grad_v_init')
        self.add_param(ctype=f'{dtype} *', cname='v_th')
        if hard_reset:
            self.add_param(ctype=f'{dtype} &', cname='v_reset')

    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        if self.dtype == 'float':
            codes.append('float grad_h = 0.0f;')
        elif self.dtype == 'half2':
            codes.append(cfunction.float2half2(y='half2 grad_h', x='0.0f'))
        else:
            raise NotImplementedError(self.dtype)

        self._pre_core = codes.codes
        return self._pre_core

    @property
    def post_core(self):

        codes = base.CodeTyper(16)
        codes.append(self.grad_h_next_to_v())
        codes.append(cfunction.mul(z='grad_v_init[index]', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        self._post_core = codes.codes
        return self._post_core

    def grad_h_next_to_v(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t+1]}{\\mathrm{d} V[t]}`.

        This function should define how ``grad_h_next_to_v`` is calculated. Note that ``grad_h_next_to_v`` has not been
        declared. Thus, this function should also declare ``grad_h_next_to_v``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_next_to_v(self) -> str:
                return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)
        """
        return '// grad_h_next_to_v should be defined here!'


    def grad_h_to_x(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t]}{\\mathrm{d} X[t]}`.

        This function should define how ``grad_h_to_x`` is calculated. Note that ``grad_h_to_x`` has not been
        declared. Thus, this function should also declare ``grad_h_to_x``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_to_x(self) -> str:
                return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
        """
        return '// grad_h_to_x should be defined here!'

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(cfunction.sub(z=f'const {self.dtype} over_th', x='h_seq[t]', y='v_th', dtype=self.dtype))
        core_codes.append(cfunction.heaviside(y=f'const {self.dtype} spike_seq_t', x='over_th', dtype=self.dtype))
        core_codes.append(self.surrogate_function(y=f'const {self.dtype} grad_s_to_h', x='over_th', dtype=self.dtype))

        if self.hard_reset:
            core_codes.append(
                cfunction.sub(z=f'{self.dtype} grad_v_to_h', x=cfunction.constant(y=None, x=1., dtype=self.dtype),
                              y='spike_seq_t', dtype=self.dtype))

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(z=f'{self.dtype} temp_var', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                    core_codes.append(cfunction.mul(z=f'temp_var', x='temp_var', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.add(z=f'grad_v_to_h', x='temp_var', y='grad_v_to_h', dtype=self.dtype))


        else:
            core_codes.append(f'{self.dtype} grad_v_to_h = {cfunction.constant(None, 1., dtype=self.dtype)}')

            if not self.detach_reset:
                with base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(z=f'{self.dtype} temp_var', x='v_th', y='grad_s_to_h', dtype=self.dtype))
                    core_codes.append(cfunction.sub(z=f'grad_v_to_h', x='grad_v_to_h', y='temp_var', dtype=self.dtype))

        core_codes.append(self.grad_h_next_to_v())
        core_codes.append(cfunction.mul(z='grad_h', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        core_codes.append(cfunction.add(z='grad_h', x='grad_v_seq[t]', y='grad_h', dtype=self.dtype))
        core_codes.append(cfunction.mul(z='grad_h', x='grad_h', y='grad_v_to_h', dtype=self.dtype))
        with base.CodeBlock(core_codes):
            core_codes.append(
                cfunction.mul(z=f'{self.dtype} temp_var', x='grad_spike_seq[t]', y='grad_s_to_h', dtype=self.dtype))
            core_codes.append(cfunction.add(z='grad_h', x='grad_h', y='temp_var', dtype=self.dtype))

        core_codes.append(self.grad_h_to_x())
        core_codes.append(cfunction.mul(z='grad_x_seq[t]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))

        self._core = core_codes.codes
        return self._core

# ----------------------------------------------------------------------------
# Forward pass (FPTT) – output is raw spike
# ----------------------------------------------------------------------------
class TrLIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.kernel_name = (
            f'{self.__class__.__name__}_{dtype}_'
            f'{"hard_reset" if hard_reset else "soft_reset"}_'
            f'{"decay_input" if decay_input else "no_decay_input"}_'
            f'output_spike_only_norm'
        )
        self.add_param(ctype=f'{dtype} *', cname='output_seq')
        self.add_param(ctype=f'const {dtype} *', cname='decay')

    # integrate
    def neuronal_charge(self) -> str:
        v_reset_eff = 'v_reset' if self.hard_reset else constant(None, 0.0, self.dtype)
        code = ''
        code += sub(z=f'{self.dtype} dv', x='v_v_seq[t]', y=v_reset_eff, dtype=self.dtype)

        if self.decay_input:  # H = V_prev + (X - dv)*decay
            code += sub(z=f'{self.dtype} tmp', x='x_seq[t]', y='dv', dtype=self.dtype)
            code += mul(z='tmp', x='tmp', y='decay[0]', dtype=self.dtype)
            code += add(z='h_seq[t]', x='v_v_seq[t]', y='tmp', dtype=self.dtype)
        else:                 # H = V_prev - dv*decay + X
            code += mul(z=f'{self.dtype} tmp', x='dv', y='decay[0]', dtype=self.dtype)
            code += sub(z='tmp', x='v_v_seq[t]', y='tmp', dtype=self.dtype)
            code += add(z='h_seq[t]', x='tmp', y='x_seq[t]', dtype=self.dtype)
        return code

    # full CUDA body
    @property
    def core(self):
        c = base.CodeTyper(18)

        # (1) integrate
        c.append(self.neuronal_charge())

        # (2) over_th' = H / v_th – 1, spike = Θ(over_th')
        c.append(div(z=f'{self.dtype} h_div_vth', x='h_seq[t]', y='v_th[0]', dtype=self.dtype))
        c.append(sub(z=f'{self.dtype} over_th', x='h_div_vth', y=constant(None, 1.0, self.dtype), dtype=self.dtype))
        c.append(heaviside(y='spike_seq[t]', x='over_th', dtype=self.dtype))

        # (3) output is spike
        c.append('output_seq[t] = spike_seq[t];')

        # (4) reset
        if self.hard_reset:
            c.append(
                neuronal_hard_reset(
                    v_next='v_v_seq[t + dt]',
                    h='h_seq[t]',
                    spike='spike_seq[t]',
                    v_reset='v_reset',
                    dtype=self.dtype
                )
            )
        else:
            c.append(f'v_v_seq[t + dt] = h_seq[t] - v_th[0] * spike_seq[t];')

        self._core = c.codes
        return self._core

# ----------------------------------------------------------------------------
# Back‑propagation (BPTT)
# ----------------------------------------------------------------------------
class TrLIFNodeBPTTKernel(NeuronBPTTKernel):
    def __init__(self,
                 decay_input: bool,
                 surrogate_function: Callable,
                 hard_reset: bool,
                 detach_reset: bool,
                 dtype: str):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.kernel_name = (
            f'{self.__class__.__name__}_{dtype}_'
            f'{"hard_reset" if hard_reset else "soft_reset"}_'
            f'{"detach_reset" if detach_reset else "nodetach_reset"}_'
            f'{"decay_input" if decay_input else "no_decay_input"}_'
            f'output_spike_only_norm'
        )
        self.add_param(ctype=f'const {dtype} *', cname='grad_output_seq')
        self.add_param(ctype=f'const {dtype} *', cname='decay')
        self.add_param(ctype=f'float *', cname='grad_decay')
        self.add_param(ctype=f'const {dtype} *', cname='v_v_seq')
        self.add_param(ctype=f'float *', cname='grad_v_th')

    # kernel scaffolding ------------------------------------------------------
    @property
    def head(self):
        code = f'''
        {{
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ float sdata_decay[{configure.cuda_threads}];
            __shared__ float sdata_vth[{configure.cuda_threads}];
            if (index < N)
            {{
                const int dt = N;
        ''' + self.pre_core
        if self.reverse:
            code += '''
                for (int t = numel - N + index; t >= 0; t -= dt)
                {'''
        else:
            code += '''
                for (int t = index; t < numel; t += dt)
                {'''
        return code

    @property
    def pre_core(self):
        codes = base.CodeTyper(16)
        codes.append('sdata_decay[threadIdx.x] = 0.0f;')
        codes.append('sdata_vth[threadIdx.x] = 0.0f;')
        return super().pre_core + '\n' + codes.codes

    @property
    def tail(self):
        return '''
                }   // end time loop
                ''' + self.post_core + '''
            }
            else
            {
                sdata_decay[threadIdx.x] = 0.0f;
                sdata_vth[threadIdx.x] = 0.0f;
            }
            int stride_total = blockDim.x;
            #pragma unroll
            for (int stride = stride_total >> 1; stride > 0; stride >>= 1)
            {
                __syncthreads();
                if (threadIdx.x < stride)
                {
                    sdata_decay[threadIdx.x] += sdata_decay[threadIdx.x + stride];
                    sdata_vth[threadIdx.x]   += sdata_vth[threadIdx.x + stride];
                }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
                atomicAdd(grad_decay, sdata_decay[0]);
                atomicAdd(grad_v_th,  sdata_vth[0]);
            }
        }
        '''

    # analytic helpers --------------------------------------------------------
    def grad_h_next_to_v(self) -> str:
        return sub(z=f'const {self.dtype} grad_h_next_to_v',
                   x=constant(None, 1.0, self.dtype),
                   y='decay[0]', dtype=self.dtype)

    def grad_h_to_x(self) -> str:
        if self.decay_input:
            return f'const {self.dtype} grad_h_to_x = decay[0];'
        return constant(y=f'const {self.dtype} grad_h_to_x', x=1.0, dtype=self.dtype)

    # main CUDA body ----------------------------------------------------------
    @property
    def core(self):
        c = base.CodeTyper(18)

        # (a) propagate grad from next step
        c.append(self.grad_h_next_to_v())
        c.append(mul(z='grad_h', x='grad_h', y='grad_h_next_to_v', dtype=self.dtype))
        c.append(add(z='grad_h', x='grad_v_seq[t]', y='grad_h', dtype=self.dtype))
        c.append(f'const {self.dtype} g_V_after = grad_h;')

        # (b) local spike math -------------------------------------------------
        c.append(div(z=f'{self.dtype} h_div_vth', x='h_seq[t]', y='v_th[0]', dtype=self.dtype))
        c.append(sub(z=f'{self.dtype} over_th', x='h_div_vth', y=constant(None,1.0,self.dtype), dtype=self.dtype))
        c.append(heaviside(y=f'const {self.dtype} S_t', x='over_th', dtype=self.dtype))
        c.append(self.surrogate_function(y=f'const {self.dtype} dphi', x='over_th', dtype=self.dtype))

        # d(over')/dH = 1 / v_th
        c.append(div(z=f'const {self.dtype} d_over_dH', x=constant(None,1.0,self.dtype), y='v_th[0]', dtype=self.dtype))
        # d(over')/dv_th = -H / v_th^2
        c.append(mul(z=f'{self.dtype} vth_sq', x='v_th[0]', y='v_th[0]', dtype=self.dtype))
        c.append(neg(y=f'{self.dtype} neg_H', x='h_seq[t]', dtype=self.dtype))
        c.append(div(z=f'const {self.dtype} d_over_dVth', x='neg_H', y='vth_sq', dtype=self.dtype))

        # dS/dH , dS/dV_th
        c.append(mul(z=f'const {self.dtype} dS_dH',   x='dphi', y='d_over_dH',   dtype=self.dtype))
        c.append(mul(z=f'const {self.dtype} dS_dVth', x='dphi', y='d_over_dVth', dtype=self.dtype))

        # (c) dV_after/dH (reset path) ----------------------------------------
        if self.hard_reset:
            c.append(f'{self.dtype} dV_after_dH;')
            c.append(sub(z='dV_after_dH', x=constant(None,1.0,self.dtype), y='S_t', dtype=self.dtype))
            if not self.detach_reset:
                c.append(sub(z=f'{self.dtype} tmp', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                c.append(mul(z='tmp', x='tmp', y='dS_dH', dtype=self.dtype))
                c.append(add(z='dV_after_dH', x='dV_after_dH', y='tmp', dtype=self.dtype))
        else:
            if self.detach_reset:
                c.append(f'{self.dtype} dV_after_dH = {constant(None,1.0,self.dtype)};')
            else:
                c.append(f'{self.dtype} dV_after_dH;')
                c.append(mul(z=f'{self.dtype} vth_dS', x='v_th[0]', y='dS_dH', dtype=self.dtype))
                c.append(sub(z='dV_after_dH', x=constant(None,1.0,self.dtype), y='vth_dS', dtype=self.dtype))

        # (d) grad_h at current step ------------------------------------------
        c.append(mul(z='grad_h', x='g_V_after', y='dV_after_dH', dtype=self.dtype))
        c.append(mul(z=f'{self.dtype} tmp2', x='grad_output_seq[t]', y='dS_dH', dtype=self.dtype))
        c.append(add(z='grad_h', x='grad_h', y='tmp2', dtype=self.dtype))

        # (e) v_th gradient ----------------------------------------------------
        c.append(mul(z=f'{self.dtype} g_vth_out', x='grad_output_seq[t]', y='dS_dVth', dtype=self.dtype))

        c.append(f'{self.dtype} g_vth_reset = {constant(None,0.0,self.dtype)};')
        if not self.detach_reset:
            c.append(f'{self.dtype} dV_after_dVth;')
            if self.hard_reset:
                c.append(sub(z=f'{self.dtype} vr_minus_h', x='v_reset', y='h_seq[t]', dtype=self.dtype))
                c.append(mul(z='dV_after_dVth', x='vr_minus_h', y='dS_dVth', dtype=self.dtype))
            else:
                c.append(mul(z=f'{self.dtype} vth_dSdV', x='v_th[0]', y='dS_dVth', dtype=self.dtype))
                c.append(sub(z='dV_after_dVth', x=neg(y=None,x='S_t',dtype=self.dtype), y='vth_dSdV', dtype=self.dtype))
            c.append(mul(z='g_vth_reset', x='g_V_after', y='dV_after_dVth', dtype=self.dtype))

        c.append(add(z=f'{self.dtype} g_vth_total', x='g_vth_out', y='g_vth_reset', dtype=self.dtype))
        if self.dtype == 'float':
            c.append('sdata_vth[threadIdx.x] += g_vth_total;')
        elif self.dtype == 'half2':
            c.append('sdata_vth[threadIdx.x] += __half2float('
                     '__hadd(__low2half(g_vth_total), __high2half(g_vth_total)));')

        # (f) decay gradient ---------------------------------------------------
        c.append(f'{self.dtype} dH_dDecay;')
        if self.decay_input:
            c.append(sub(z=f'{self.dtype} diff', x='h_seq[t]', y='v_v_seq[t]', dtype=self.dtype))
            c.append(div(z='dH_dDecay', x='diff', y=f'(decay[0]+{constant(None,1e-8,self.dtype)})', dtype=self.dtype))
        else:
            v_reset_eff = 'v_reset' if self.hard_reset else constant(None,0.0,self.dtype)
            c.append(sub(z='dH_dDecay', x='v_v_seq[t]', y=v_reset_eff, dtype=self.dtype))
            c.append(neg(y='dH_dDecay', x='dH_dDecay', dtype=self.dtype))
        c.append(mul(z=f'{self.dtype} g_decay_local', x='grad_h', y='dH_dDecay', dtype=self.dtype))
        if self.dtype == 'float':
            c.append('sdata_decay[threadIdx.x] += g_decay_local;')
        elif self.dtype == 'half2':
            c.append('sdata_decay[threadIdx.x] += __half2float('
                     '__hadd(__low2half(g_decay_local), __high2half(g_decay_local)));')

        # (g) grad_x
        c.append(self.grad_h_to_x())
        c.append(mul(z='grad_x_seq[t]', x='grad_h', y='grad_h_to_x', dtype=self.dtype))

        self._core = c.codes
        return self._core

# ----------------------------------------------------------------------------
# Autograd bridge (ATGF) – identical call signature
# ----------------------------------------------------------------------------
class TrLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x_seq: torch.Tensor,
                v_init: torch.Tensor,
                v_th_param: torch.Tensor,
                v_reset: Optional[float],
                decay_param: torch.Tensor,
                forward_kernel: TrLIFNodeFPTTKernel,
                backward_kernel: TrLIFNodeBPTTKernel):

        py_dict = {
            'x_seq':   x_seq,
            'v_init':  v_init,
            'v_th':    v_th_param,
            'v_reset': v_reset,
            'decay':   decay_param,
        }

        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
        new_tensors(('output_seq',), py_dict, ref='x_seq')

        if py_dict.get('v_reset') is None:
            py_dict.pop('v_reset', None)
        forward_kernel((blocks,), (threads,), py_dict)

        NeuronATGFBase.ctx_save(
            ctx, requires_grad,
            py_dict['h_seq'],         # 0
            py_dict['v_v_seq'],       # 1
            v_th_param,               # 2
            decay_param,              # 3
            blocks=blocks, threads=threads,
            numel=py_dict['numel'], N=py_dict['N'],
            v_th=py_dict['v_th'],
            v_reset=py_dict.get('v_reset'),
            decay=py_dict['decay'],
            backward_kernel=backward_kernel
        )

        return py_dict['output_seq'], py_dict['v_v_seq'][1:,]

    @staticmethod
    def backward(ctx, grad_output_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel = ctx.backward_kernel
        blocks, threads = ctx.blocks, ctx.threads

        h_seq, v_v_seq, v_th_param, decay_param = ctx.saved_tensors
        _, _, _, py_dict = NeuronATGFBase.pre_backward(ctx, grad_output_seq, grad_v_seq)

        py_dict.update({
            'h_seq':            h_seq,
            'v_v_seq':          v_v_seq,
            'v_th':             ctx.v_th,
            'decay':            ctx.decay,
            'grad_output_seq':  grad_output_seq,
            'grad_decay':       torch.zeros_like(decay_param, dtype=torch.float, device=decay_param.device),
            'grad_v_th':        torch.zeros_like(v_th_param,   dtype=torch.float, device=v_th_param.device),
        })
        if ctx.v_reset is not None:
            py_dict['v_reset'] = ctx.v_reset

        backward_kernel((blocks,), (threads,), py_dict)

        return (py_dict['grad_x_seq'],      # x_seq grad
                py_dict['grad_v_init'],     # v_init grad
                py_dict['grad_v_th'],       # v_th grad
                None,                       # v_reset (not learnable)
                py_dict['grad_decay'],      # decay grad
                None, None, None, None, None)

class TrLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                x_seq: torch.Tensor,
                v_init: torch.Tensor,
                v_th_param: torch.Tensor, # Learnable v_th (scalar or per-neuron tensor)
                v_reset: Optional[float],
                decay_param: torch.Tensor, # Learnable decay (scalar or per-neuron tensor)
                forward_kernel: TrLIFNodeFPTTKernel,
                backward_kernel: TrLIFNodeBPTTKernel,
                ):
        """
        Autograd Function for TrLIFNode.
        Output is spike * v_th.
        v_th and decay are learnable torch.Tensor parameters.
        """
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            # This check might be specific to certain paddings for half2.
            # If v_th or decay are per-neuron and float16, similar checks might be needed.
            logging.warning('Half2 with odd neuron count might have padding issues for per-neuron params.')

        # Ensure v_th_param and decay_param are correctly shaped (e.g., broadcast if scalar)
        # For CuPy kernels, they often expect pointers to memory that matches device and dtype.
        # scalar_to_cupy handles float->cupy conversion. For tensors, they should be on the right device.

        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th_param, # Pass tensor directly
            'v_reset': v_reset,
            'decay': decay_param, # Pass tensor directly
            # output_seq will be created by new_tensors
        }

        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
        # pre_forward converts scalar v_th, v_reset, decay in py_dict to cupy arrays
        # For tensor v_th, decay, they are already tensors. scalar_to_cupy skips them.
        # It also adds 'h_seq', 'spike_seq', 'v_seq', 'v_v_seq'
        # and CUDA specific 'N', 'numel'.

        # Add output_seq to py_dict (same shape as spike_seq or x_seq)
        new_tensors(('output_seq',), py_dict, ref='x_seq')


        if py_dict.get('v_reset') is None: # Use .get for safety if pre_forward might remove it
            if 'v_reset' in py_dict: py_dict.pop('v_reset')


        # --- Ensure kernels are correctly initialized (optional, if not handled by module) ---
        # This is usually done in the nn.Module layer before calling ATGF.apply
        # For example:
        # if forward_kernel is None or not forward_kernel.check_attributes(decay_input=decay_input, hard_reset=hard_reset, dtype=forward_kernel.dtype):
        #     forward_kernel = TrLIFNodeFPTTKernel(decay_input, hard_reset, x_seq.dtype_str)
        # if backward_kernel is None or not backward_kernel.check_attributes(decay_input=decay_input, hard_reset=hard_reset, detach_reset=detach_reset, surrogate_function=..., dtype=...):
        #     backward_kernel = TrLIFNodeBPTTKernel(decay_input, surrogate_function, hard_reset, detach_reset, x_seq.dtype_str)

        forward_kernel((blocks,), (threads,), py_dict)

        if 'v_reset' not in py_dict and v_reset is not None : # If popped and was not None initially
             py_dict['v_reset'] = v_reset # keep original for saving if needed by backward

        # Save tensors and non-tensor context for backward
        # Saved tensors: h_seq, v_v_seq (for dH/dDecay), v_th_param, decay_param
        # v_th and decay are saved as their original torch tensor forms for autograd.
        NeuronATGFBase.ctx_save(ctx, requires_grad,
                                py_dict['h_seq'],         # tensor 0
                                py_dict['v_v_seq'],       # tensor 1
                                v_th_param,             # tensor 2 (original torch tensor)
                                decay_param,            # tensor 3 (original torch tensor)
                                blocks=blocks, threads=threads,
                                numel=py_dict['numel'], N=py_dict['N'],
                                # Pass cupy versions of v_th, v_reset, decay for BPTT kernel if they were modified
                                # by scalar_to_cupy. If they were already tensors, pre_forward might not change them.
                                # The BPTT kernel expects pointers, so py_dict versions are fine.
                                v_th=py_dict['v_th'], # cupy version from pre_forward
                                v_reset=py_dict.get('v_reset'), # cupy version or None
                                decay=py_dict['decay'],# cupy version from pre_forward
                                backward_kernel=backward_kernel)

        # Return the calculated output (spike * v_th) and the final voltage sequence (excluding init)
        return py_dict['output_seq'], py_dict['v_v_seq'][1:, ]

    @staticmethod
    def backward(ctx,
                 grad_output_seq: torch.Tensor, # Gradient of Loss wrt (spike * v_th)
                 grad_v_seq: torch.Tensor      # Gradient of Loss wrt v_v_seq[1:]
                 ):

        backward_kernel = ctx.backward_kernel
        blocks = ctx.blocks
        threads = ctx.threads

        h_seq = ctx.saved_tensors[0]
        v_v_seq = ctx.saved_tensors[1]
        v_th_param_torch = ctx.saved_tensors[2] # Original torch tensor for v_th
        decay_param_torch = ctx.saved_tensors[3]# Original torch tensor for decay

        # py_dict for BPTT kernel
        # NeuronATGFBase.pre_backward prepares grad_x_seq, grad_v_init based on grad_output_seq shape,
        # and also handles numel, N, and scalar_to_cupy for passed items.
        # We pass grad_output_seq as if it were grad_spike_seq for shape purposes.
        # The kernel itself knows it's grad_output_seq.
        _ , _ , _, py_dict = NeuronATGFBase.pre_backward(ctx, grad_output_seq, grad_v_seq)
        # pre_backward might have used ctx.v_th, ctx.v_reset, etc.
        # We need to ensure the correct (potentially cupy converted) versions are in py_dict
        # for the BPTT kernel call.

        py_dict['h_seq'] = h_seq
        py_dict['v_v_seq'] = v_v_seq # Needed for dH/dDecay

        # Use cupy versions saved from forward's pre_forward for kernel call
        py_dict['v_th'] = ctx.v_th
        py_dict['decay'] = ctx.decay
        if ctx.v_reset is not None:
            py_dict['v_reset'] = ctx.v_reset
        elif 'v_reset' in py_dict: # if pre_backward added a default v_reset (e.g. 0.0) but forward didn't have one
             py_dict.pop('v_reset')


        # Gradients to be computed by the kernel (must be float for atomicAdd)
        # Ensure they are created with the correct shape (scalar or per-neuron)
        # Matching the shape of the original torch parameters v_th_param, decay_param
        py_dict['grad_decay'] = torch.zeros_like(decay_param_torch, dtype=torch.float, device=decay_param_torch.device)
        py_dict['grad_v_th'] = torch.zeros_like(v_th_param_torch, dtype=torch.float, device=v_th_param_torch.device)

        # Pass the gradient input (dL/dO) to the kernel
        py_dict['grad_output_seq'] = grad_output_seq


        # Call the BPTT CUDA kernel
        backward_kernel((blocks,), (threads,), py_dict)

        # Gradients for x_seq, v_init are already in py_dict['grad_x_seq'], py_dict['grad_v_init']
        # Gradients for v_th_param, decay_param are in py_dict['grad_v_th'], py_dict['grad_decay']

        # Return gradients in the order of inputs to forward function
        # x_seq, v_init, v_th_param, v_reset, decay_param, forward_kernel, backward_kernel, kernel_params...
        # None for non-tensor inputs or inputs not requiring grad
        grad_for_v_reset = None # v_reset is not learnable here

        return (py_dict['grad_x_seq'],
                py_dict['grad_v_init'],
                py_dict['grad_v_th'],    # Grad for v_th_param
                grad_for_v_reset,        # Grad for v_reset
                py_dict['grad_decay'],   # Grad for decay_param
                None,                    # Grad for forward_kernel
                None,                    # Grad for backward_kernel
                None, None, None)        # Grads for decay_input, hard_reset, detach_reset

def lsq_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha_half = constant(None, alpha/2, dtype)
    alpha_inv = constant(None, 1/alpha, dtype)
    
    # |x| <= alpha/2 인지 확인
    codes = greater_equal(z=f'const {dtype} lsq_backward__mask', 
                         x=alpha_half, 
                         y=abs(y=None, x=x, dtype=dtype), 
                         dtype=dtype)
    
    # mask가 True면 1/alpha, False면 0
    codes += if_else(z=y, 
                    x=alpha_inv, 
                    y=constant(None, 0., dtype), 
                    mask=f'lsq_backward__mask', 
                    dtype=dtype)
    
    return codes

def lsq_cuda_codes(y: str, x: str, dtype: str):
    return lsq_backward(y=y, x=x, alpha=1.0, dtype=dtype)
    
def tri_backward(y: str, x: str, alpha: float, dtype: str):
    assert y is not None
    alpha_const = constant(None, alpha, dtype)
    alpha_inv = constant(None, 1/alpha, dtype)
    
    # alpha - |x| 계산
    codes = sub(z=f'const {dtype} tri_backward__diff',
               x=alpha_const,
               y=abs(y=None, x=x, dtype=dtype),
               dtype=dtype)
    
    # (1/alpha) * (1/alpha) * (alpha - |x|) 계산
    codes += mul(z=f'const {dtype} tri_backward__grad',
                x=mul(z=None, x=alpha_inv, y=alpha_inv, dtype=dtype),
                y=f'tri_backward__diff',
                dtype=dtype)
    
    # 0보다 작으면 0으로 clamp
    codes += greater_equal(z=f'const {dtype} tri_backward__mask',
                          x=f'tri_backward__grad',
                          y=constant(None, 0., dtype),
                          dtype=dtype)
    
    codes += if_else(z=y,
                    x=f'tri_backward__grad',
                    y=constant(None, 0., dtype),
                    mask=f'tri_backward__mask',
                    dtype=dtype)
    
    return codes

def tri_cuda_codes(y: str, x: str, dtype: str):
    return tri_backward(y=y, x=x, alpha=1.0, dtype=dtype)

class TrLIFNode(BaseNode):
    """
    Multi‑step Leaky‑Integrate‑and‑Fire neuron with learnable decay (1/τ)
    and threshold v_th.  The kernel selection now emits raw spikes (0/1).
    """
    def __init__(self,
                 init_tau: float = 2.0,
                 decay_input: bool = True,
                 init_threshold: float = 1.,
                 v_reset: Optional[float] = None,
                 surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False,
                 step_mode='m',
                 backend='cupy',
                 store_v_seq: bool = False,
                 use_mpinit: bool = True):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(init_threshold, v_reset, surrogate_function,
                         detach_reset, step_mode, backend, store_v_seq)

        self.decay_input = decay_input
        self.use_mpinit  = use_mpinit

        # Parameterization: tau via sigmoid weight, threshold via softplus.
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.as_tensor(init_w))
        init_z = math.log(math.exp(init_threshold) - 1.0)
        self.z = nn.Parameter(torch.as_tensor(init_z))

        self.w.requires_grad = False   # tau fixed during training
        self.z.requires_grad = True

        self.running_mean = torch.tensor([0.0])
        self.count = 0

    # --------------------------------------------------------------
    # Public helpers
    # --------------------------------------------------------------
    @property
    def supported_backends(self):
        if self.step_mode == 'm':
            return ('cupy',)
        raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
            thr = F.softplus(self.z)
        return (super().extra_repr() +
                f', decay_input={self.decay_input}, tau={tau}, thr={thr}, '
                f'running_mean={self.running_mean}, use_mpinit={self.use_mpinit}')

    # --------------------------------------------------------------
    # Multi‑step forward (uses kernels above)
    # --------------------------------------------------------------
    def multi_step_forward(self, x_seq: torch.Tensor):
        self.running_mean = self.running_mean.to(x_seq)
        if self.backend != 'cupy':
            raise ValueError(self.backend)

        hard_reset = self.v_reset is not None
        dtype = 'float' if x_seq.dtype == torch.float else 'half2'
        if x_seq.dtype not in (torch.float, torch.half):
            raise NotImplementedError(x_seq.dtype)

        # Build / reuse kernels
        if (self.forward_kernel is None or
            not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                     dtype=dtype,
                                                     decay_input=self.decay_input)):
            self.forward_kernel = TrLIFNodeFPTTKernel(decay_input=self.decay_input,
                                                      hard_reset=hard_reset,
                                                      dtype=dtype)
        if (self.backward_kernel is None or
            not self.backward_kernel.check_attributes(
                surrogate_function=self.surrogate_function.cuda_codes,
                hard_reset=hard_reset,
                detach_reset=self.detach_reset,
                dtype=dtype,
                decay_input=self.decay_input)):
            self.backward_kernel = TrLIFNodeBPTTKernel(
                decay_input=self.decay_input,
                surrogate_function=self.surrogate_function.cuda_codes,
                hard_reset=hard_reset,
                detach_reset=self.detach_reset,
                dtype=dtype)

        # Membrane potential init
        self.v_float_to_tensor(x_seq[0])
        self.v = torch.ones_like(x_seq[0], device=x_seq.device) * self.running_mean.to(x_seq.device)

        thr  = F.softplus(self.z)
        tau  = 1. / self.w.sigmoid()

        # Call custom ATGF
        spike_seq, v_seq = TrLIFNodeATGF.apply(
            x_seq.flatten(1),
            self.v.flatten(0),
            thr.to(x_seq),
            self.v_reset,
            (1 / tau).to(x_seq),
            self.forward_kernel,
            self.backward_kernel
        )
        spike_seq = spike_seq.reshape(x_seq.shape)
        v_seq     = v_seq.reshape(x_seq.shape)

        if self.store_v_seq:
            self.v_seq = v_seq
        if self.use_mpinit:
            self.mask = torch.sum(spike_seq, dim=0) > 0.
            self.count += 1

        self.v = v_seq[-1].clone()
        return spike_seq

    # --------------------------------------------------------------
    # Running‑stat update helpers (unchanged)
    # --------------------------------------------------------------
    def update_running_stats(self):
        with torch.no_grad():
            mask = self.mask
            masked_v = self.v * mask.float()
            count_non_zero = torch.clamp(mask.float().sum(), min=1.0)
            mean = masked_v.sum() / count_non_zero
            self.running_mean = self.running_mean * 0.9 + mean * 0.1
            self.running_mean = torch.clamp(self.running_mean, min=0.0)

    def reset(self):
        if self.training and self.count > 0 and self.use_mpinit:
            self.update_running_stats()
        super().reset()
        self.count = 0
