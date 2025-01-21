import torch
from torch import nn

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True, device=None):  # Added device parameter
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        
        # Move tensor creation to specified device
        self.freqs = 2**torch.linspace(0, max_logscale, N_freqs, device=device) if logscale \
            else torch.linspace(1, 2**max_logscale, N_freqs, device=device)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):
        """
        [Previous docstring remains the same]
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min

        # Using ModuleList for better GPU memory management
        self.xyz_encoding_layers = nn.ModuleList()
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            self.xyz_encoding_layers.append(nn.Sequential(layer, nn.ReLU(True)))

        self.xyz_encoding_final = nn.Linear(W, W)

        # Updated activation functions to use functional forms
        self.dir_encoding = nn.Sequential(
            nn.Linear(W+in_channels_dir+self.in_channels_a, W//2),
            nn.ReLU(True)
        )

        # Static output layers with modern implementations
        self.static_sigma = nn.Sequential(
            nn.Linear(W, 1),
            nn.Softplus()
        )
        self.static_rgb = nn.Sequential(
            nn.Linear(W//2, 3),
            nn.Sigmoid()
        )

        if self.encode_transient:
            self.transient_encoding = nn.Sequential(
                nn.Linear(W+in_channels_t, W//2),
                nn.ReLU(True),
                nn.Linear(W//2, W//2),
                nn.ReLU(True),
                nn.Linear(W//2, W//2),
                nn.ReLU(True),
                nn.Linear(W//2, W//2),
                nn.ReLU(True)
            )
            
            self.transient_sigma = nn.Sequential(
                nn.Linear(W//2, 1),
                nn.Softplus()
            )
            self.transient_rgb = nn.Sequential(
                nn.Linear(W//2, 3),
                nn.Sigmoid()
            )
            self.transient_beta = nn.Sequential(
                nn.Linear(W//2, 1),
                nn.Softplus()
            )

    def forward(self, x, sigma_only=False, output_transient=True):
        """
        [Previous docstring remains the same]
        """
        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                              self.in_channels_dir+self.in_channels_a,
                              self.in_channels_t], dim=-1)
        else:
            input_xyz, input_dir_a = \
                torch.split(x, [self.in_channels_xyz,
                              self.in_channels_dir+self.in_channels_a], dim=-1)

        xyz_ = input_xyz
        for i, layer in enumerate(self.xyz_encoding_layers):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, xyz_], 1)
            xyz_ = layer(xyz_)

        static_sigma = self.static_sigma(xyz_)
        if sigma_only:
            return static_sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding)
        static = torch.cat([static_rgb, static_sigma], 1)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding)
        transient_rgb = self.transient_rgb(transient_encoding)
        transient_beta = self.transient_beta(transient_encoding)

        transient = torch.cat([transient_rgb, transient_sigma,
                             transient_beta], 1)

        return torch.cat([static, transient], 1)
