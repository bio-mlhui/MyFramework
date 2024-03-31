class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        kwargs.update(
            d_model=d_model, d_state=d_state, ssm_ratio=ssm_ratio, dt_rank=dt_rank,
            act_layer=act_layer, d_conv=d_conv, conv_bias=conv_bias, dropout=dropout, bias=bias,
            dt_min=dt_min, dt_max=dt_max, dt_init=dt_init, dt_scale=dt_scale, dt_init_floor=dt_init_floor,
            initialize=initialize, forward_type=forward_type, channel_first=channel_first,
        )
        # only used to run previous version
        if forward_type in ["v0", "v0seq"]:
            self.__initv0__(seq=("seq" in forward_type), **kwargs)
            return
        elif forward_type.startswith("xv"):
            self.__initxv__(**kwargs)
            return
        else:
            self.__initv2__(**kwargs)
            return

    # only used to run previous version
    def __initv0__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        # ======================
        dropout=0.0,
        # ======================
        seq=False,
        force_fp32=True,
        **kwargs,
    ):
        if "channel_first" in kwargs:
            assert not kwargs["channel_first"]
        act_layer = nn.SiLU
        dt_min = 0.001
        dt_max = 0.1
        dt_init = "random"
        dt_scale = 1.0
        dt_init_floor = 1e-4
        bias = False
        conv_bias = True
        d_conv = 3
        k_group = 4
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.forward = self.forwardv0 
        if seq:
            self.forward = partial(self.forwardv0, seq=True)
        if not force_fp32:
            self.forward = partial(self.forwardv0, force_fp32=False)

        # in proj ============================
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            groups=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj

        # dt proj ============================
        self.dt_projs = [
            self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
        del self.dt_projs
            
        # A, D =======================================
        self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)     

        # out proj =======================================
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def __initv2__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,    
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardv2

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.oact, forward_type = checkpostfix("oact", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        elif channel_first:
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
            v02=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v03=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v04=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            v05=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
            # ===============================
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            ),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            ),
            # ===============================
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex),
            v4=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, no_einsum=True, CrossScan=CrossScanTriton, CrossMerge=CrossMergeTriton),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        k_group = 4

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_act = nn.GELU() if self.oact else nn.Identity()
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

    def __initxv__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        channel_first=False,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv
        self.channel_first = channel_first
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_inner = d_inner
        Linear = Linear2d if channel_first else nn.Linear
        self.forward = self.forwardxv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        self.out_norm_shape = "v1"
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            class SoftmaxSpatial(nn.Softmax):
                def forward(self, x: torch.Tensor):
                    B, C, H, W = x.shape
                    return super().forward(x.view(B, C, -1)).view(B, C, H, W)
            self.out_norm = SoftmaxSpatial(dim=-1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        elif channel_first:
            self.out_norm = LayerNorm2d(d_inner)
        else:
            self.out_norm_shape = "v0"
            self.out_norm = nn.LayerNorm(d_inner)

        k_group = 4
        # in proj =======================================
        self.out_act: nn.Module = nn.Identity()
        # 0309 -> 0319 needs to be rerun...
        if False:
            # change Conv2d to Linear2d Next
            if forward_type.startswith("xv1"):
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv2"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight

            if forward_type.startswith("xv3"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv4"):
                self.forward = partial(self.forwardxv, mode="xv3")
                self.in_proj = nn.Conv2d(d_model, d_inner + 4 * dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            if forward_type.startswith("xv5"):
                self.in_proj = nn.Conv2d(d_model, d_inner + d_inner + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.forward = partial(self.forwardxv, mode="xv2")
                del self.dt_projs_weight
                self.out_act = nn.GELU()

            if forward_type.startswith("xv6"):
                self.forward = partial(self.forwardxv, mode="xv1")
                self.in_proj = nn.Conv2d(d_model, d_inner + dt_rank + 8 * d_state, 1, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()

            # to see if Linear2d and nn.Conv2d differ, as they will be inited differ
            if forward_type.startswith("xv61"):
                self.forward = partial(self.forwardxv, mode="xv1")
                self.in_proj = Linear2d(d_model, d_inner + dt_rank + 8 * d_state, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()
            
            if forward_type.startswith("xv7"):
                self.forward = partial(self.forwardxv, mode="xv1", omul=True)
                self.in_proj = Linear2d(d_model, d_inner + dt_rank + 8 * d_state, bias=bias, **factory_kwargs)
                self.out_act = nn.GELU()
            
        if True:
            omul, forward_type = checkpostfix("mul", forward_type)
            if omul:
                self.omul = nn.Identity()
            oact, forward_type = checkpostfix("act", forward_type)
            self.out_act = nn.GELU() if oact else nn.Identity()

            if forward_type.startswith("xv1a"):
                self.forward = partial(self.forwardxv, mode="xv1a", omul=omul)
                self.in_proj = Linear2d(d_model, d_inner + dt_rank + 8 * d_state, bias=bias, **factory_kwargs)

            if forward_type.startswith("xv2a"):
                self.forward = partial(self.forwardxv, mode="xv2a", omul=omul)
                self.in_proj = Linear2d(d_model, d_inner + d_inner + 8 * d_state,bias=bias, **factory_kwargs)

            if forward_type.startswith("xv3a"):
                self.forward = partial(self.forwardxv, mode="xv3a", omul=omul)
                self.in_proj = Linear2d(d_model, d_inner + 4 * dt_rank + 8 * d_state,bias=bias, **factory_kwargs)

        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_model,
                out_channels=d_model,
                groups=d_model,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act: nn.Module = act_layer()

        # out proj =======================================
        self.out_proj = Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(0.1 * torch.rand((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(0.1 * torch.rand((k_group, d_inner)))

        if forward_type.startswith("xv2"):
            del self.dt_projs_weight

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D
    
    # only used to run previous version
    def forwardv0(self, x: torch.Tensor, SelectiveScan = SelectiveScanMamba, seq=False, force_fp32=True, **kwargs):
        x = self.in_proj(x)
        x, z = x.chunk(2, dim=-1) # (b, h, w, d)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.contiguous() # (b, k, d_state, l)
        Cs = Cs.contiguous() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1
        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)
        
        if force_fp32:
            xs, dts, Bs, Cs = to_fp32(xs, dts, Bs, Cs)

        if seq:
            out_y = []
            for i in range(4):
                yi = selective_scan(
                    xs.view(B, K, -1, L)[:, i], dts.view(B, K, -1, L)[:, i], 
                    As.view(K, -1, N)[i], Bs[:, i].unsqueeze(1), Cs[:, i].unsqueeze(1), Ds.view(K, -1)[i],
                    delta_bias=dt_projs_bias.view(K, -1)[i],
                    delta_softplus=True,
                ).view(B, -1, L)
                out_y.append(yi)
            out_y = torch.stack(out_y, dim=1)
        else:
            out_y = selective_scan(
                xs, dts, 
                As, Bs, Cs, Ds,
                delta_bias=dt_projs_bias,
                delta_softplus=True,
            ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    # Note: we did not use csm_triton in and before vssm1_0230, we used pytorch version !
    # Note: we did not use no_einsum in and before vssm1_0230, we used einsum version !    
    def forward_corev2(
        self,
        x: torch.Tensor=None, 
        x_proj_weight: torch.Tensor=None,
        x_proj_bias: torch.Tensor=None,
        dt_projs_weight: torch.Tensor=None,
        dt_projs_bias: torch.Tensor=None,
        A_logs: torch.Tensor=None,
        Ds: torch.Tensor=None,
        delta_softplus = True,
        out_norm: torch.nn.Module=None,
        out_norm_shape="v0",
        channel_first=False,
        # ==============================
        to_dtype=True, # True: final out to dtype
        force_fp32=False, # True: input fp32
        # ==============================
        nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
        ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
        # ==============================
        SelectiveScan=None,
        CrossScan=CrossScan,
        CrossMerge=CrossMerge,
        no_einsum=False, # replace einsum with linear or conv1d to raise throughput
        **kwargs,
    ):
        x_proj_weight = self.x_proj_weight
        dt_projs_weight = self.dt_projs_weight
        dt_projs_bias = self.dt_projs_bias
        A_logs = self.A_logs
        Ds = self.Ds
        out_norm = getattr(self, "out_norm", None)
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        channel_first = self.channel_first

        # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W

        if nrows == 0:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1
            
        if backnrows == 0:
            if D % 4 == 0:
                backnrows = 4
            elif D % 3 == 0:
                backnrows = 3
            elif D % 2 == 0:
                backnrows = 2
            else:
                backnrows = 1

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
        
        if no_einsum:
            xs = CrossScan.apply(x)
            x_dbl = F.conv1d(xs.view(B, -1, L), x_proj_weight.view(-1, D, 1), bias=(x_proj_bias.view(-1) if x_proj_bias is not None else None), groups=K)
            dts, Bs, Cs = torch.split(x_dbl.view(B, K, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * D, -1, 1), groups=K)
        else:
            xs = CrossScan.apply(x)
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Bs = Bs.contiguous().view(B, K, N, L)
        Cs = Cs.contiguous().view(B, K, N, L)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        if force_fp32:
            xs = xs.to(torch.float)
            dts = dts.to(torch.float)
            Bs = Bs.to(torch.float)
            Cs = Cs.to(torch.float)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
        
        y: torch.Tensor = CrossMerge.apply(ys)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=xs, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        if channel_first:
            y = y.view(B, -1, H, W)
            if out_norm_shape in ["v1"]:
                y = out_norm(y)
            else:
                y = out_norm(y.permute(0, 2, 3, 1))
                y = y.permute(0, 3, 1, 2)
            return (y.to(x.dtype) if to_dtype else y)

        if out_norm_shape in ["v1"]: # (B, C, H, W)
            y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
        else: # (B, L, C)
            y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
            y = out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)

    def forwardv2(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=(1 if self.channel_first else -1)) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        
        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if with_dconv:
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        
        y = self.forward_core(x)
        y = self.out_act(y)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out

    def forwardxv(self, x: torch.Tensor, mode="xv1a", omul=False, **kwargs):
        B, C, H, W = x.shape
        if not self.channel_first:
            B, H, W, C = x.shape
        L = H * W
        K = 4
        dt_projs_weight = getattr(self, "dt_projs_weight", None)
        A_logs = self.A_logs
        dt_projs_bias = self.dt_projs_bias
        force_fp32 = False
        delta_softplus = True
        out_norm_shape = getattr(self, "out_norm_shape", "v0")
        out_norm = self.out_norm
        to_dtype = True
        Ds = self.Ds

        to_fp32 = lambda *args: (_a.to(torch.float32) for _a in args)

        def selective_scan(u, delta, A, B, C, D, delta_bias, delta_softplus):
            return SelectiveScanOflex.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, 1, True)

        if not self.channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()

        if self.d_conv > 1:
            x = self.conv2d(x) # (b, d, h, w)
            x = self.act(x)
        x = self.in_proj(x)

        if mode in ["xv1", "xv2", "xv3", "xv7"]:
            print(f"ERROR: MODE {mode} will be deleted in the future, use {mode}a instead.")

        if mode in ["xv1"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, -1, L)
            dts = F.conv1d(dts, dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        elif mode in ["xv2"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton.apply(dts).contiguous().view(B, -1, L)
        elif mode in ["xv3"]:
            _us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            us = CrossScanTriton.apply(_us.contiguous()).view(B, -1, L)
            dts = CrossScanTriton1b1.apply(dts.contiguous().view(B, K, -1, H, W))
            dts = F.conv1d(dts.view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K).contiguous().view(B, -1, L)
        else:
            ...

        if mode in ["xv1a"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            us, dts = us.contiguous().view(B, -1, L), dts
            _us = us.view(B, K, -1, H, W)[:, 0, :, :, :]
        elif mode in ["xv2a"]:
            us, dts, Bs, Cs = x.split([self.d_inner, self.d_inner, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton.apply(dts.contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            us, dts = us.contiguous().view(B, -1, L), dts.contiguous().view(B, -1, L)
        elif mode in ["xv3a"]:
            # us, dtBCs = x.split([self.d_inner, 4 * self.dt_rank + 4 * self.d_state + 4 * self.d_state], dim=1)
            # _us = us
            # us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            # dtBCs = CrossScanTriton1b1.apply(dtBCs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            # dts, Bs, Cs = dtBCs.split([self.dt_rank, self.d_state, self.d_state], dim=2)
            # dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            # us, dts = us.contiguous().view(B, -1, L), dts
            
            us, dts, Bs, Cs = x.split([self.d_inner, 4 * self.dt_rank, 4 * self.d_state, 4 * self.d_state], dim=1)
            _us = us
            us = CrossScanTriton.apply(us.contiguous()).view(B, 4, -1, L)
            dts = CrossScanTriton1b1.apply(dts.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Bs = CrossScanTriton1b1.apply(Bs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            Cs = CrossScanTriton1b1.apply(Cs.view(B, 4, -1, H, W).contiguous()).view(B, 4, -1, L)
            dts = F.conv1d(dts.contiguous().view(B, -1, L), dt_projs_weight.view(K * self.d_inner, self.dt_rank, 1), None, groups=K)
            us, dts = us.contiguous().view(B, -1, L), dts
        else: 
            ...

        Bs, Cs = Bs.view(B, K, -1, L).contiguous(), Cs.view(B, K, -1, L).contiguous()
    
        As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
        Ds = Ds.to(torch.float) # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float) # (K * c)

        if force_fp32:
            us, dts, Bs, Cs = to_fp32(us, dts, Bs, Cs)

        ys: torch.Tensor = selective_scan(
            us, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
        ).view(B, K, -1, H, W)
            
        y: torch.Tensor = CrossMergeTriton.apply(ys)
        y = y.view(B, -1, H, W)

        if getattr(self, "__DEBUG__", False):
            setattr(self, "__data__", dict(
                A_logs=A_logs, Bs=Bs, Cs=Cs, Ds=Ds,
                us=us, dts=dts, delta_bias=delta_bias,
                ys=ys, y=y,
            ))

        # originally:
        # y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        # y = out_norm(y).view(B, H, W, -1)

        if (not self.channel_first) or (out_norm_shape in ["v0"]):
            y = out_norm(y.permute(0, 2, 3, 1))
            if self.channel_first:
                y = y.permute(0, 3, 1, 2)
        else:
            y = out_norm(y)

        y = (y.to(x.dtype) if to_dtype else y)
        y = self.out_act(y)
        if omul:
            y = y * (_us.permute(0, 2, 3, 1) if not self.channel_first else _us)
        out = self.dropout(self.out_proj(y))
        return out
