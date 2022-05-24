import torch
from torch import nn
import torch.fft
import numpy as np
import math
import functools
import torchvision.models as models


def init_weights_requ(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        # if hasattr(m, 'bias'):
        #     nn.init.uniform_(m.bias, -.5,.5)


def init_weights_normal(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')
        if hasattr(m, 'bias'):
            nn.init.uniform_(m.bias, -1, 1)
            # m.bias.data.fill_(0.)


def init_weights_selu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_elu(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))
        # if hasattr(m, 'bias'):
        #     m.bias.data.fill_(0.)


def init_weights_xavier(m):
    if type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)
        if hasattr(m, 'bias'):
            m.bias.data.fill_(0.)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class FirstSine(nn.Module):
    def __init__(self, w0=20):
        """
        Initialization of the first sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class Sine(nn.Module):
    def __init__(self, w0=20.0):
        """
        Initialization of sine nonlinearity.

        Parameters
        ----------
        w0: float
        """
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0 * input)


class RandSine(nn.Module):
    def __init__(self, mu_w0=50, std_w0=40, num_features=256):  # 30, 29
        super().__init__()
        self.w0 = mu_w0 + 2. * std_w0 * (torch.rand(num_features, dtype=torch.float32) - .5).cuda()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.w0 * input)


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5 * self.relu(input) ** 2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class ReQLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_sq = 1 ** 2

    def forward(self, input):
        r_input = torch.relu(input)
        return self.p_sq * (torch.sqrt(1. + r_input ** 2 / self.p_sq) - 1.)


def layer_factory(layer_type):
    layer_dict = \
        {'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'requ': (ReQU(inplace=False), init_weights_requ),
         'reqlu': (ReQLU, init_weights_normal),
         'sigmoid': (nn.Sigmoid(), init_weights_xavier),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'randsine': (RandSine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
         }
    return layer_dict[layer_type]


class FCBlock(nn.Module):
    def __init__(self, in_features, features, out_features,
                 nonlinearity='relu', last_nonlinearity=None,
                 batch_norm=False):
        """
        Initialization of a fully connected network.

        Parameters
        ----------
        in_features: int
        features: list
        out_features: int
        nonlinearity: str
        last_nonlinearity: str
        batch_norm: bool
        """
        super().__init__()

        # Create hidden features list
        self.hidden_features = [int(in_features)]
        if features != []:
            self.hidden_features.extend(features)
        self.hidden_features.append(int(out_features))

        self.net = []
        for i in range(len(self.hidden_features) - 1):
            hidden = False
            if i < len(self.hidden_features) - 2:
                if nonlinearity is not None:
                    nl = layer_factory(nonlinearity)[0]
                    init = layer_factory(nonlinearity)[1]
                hidden = True
            else:
                if last_nonlinearity is not None:
                    nl = layer_factory(last_nonlinearity)[0]
                    init = layer_factory(last_nonlinearity)[1]

            layer = nn.Linear(self.hidden_features[i], self.hidden_features[i + 1])

            if (hidden and (nonlinearity is not None)) or ((not hidden) and (last_nonlinearity is not None)):
                init(layer)
                self.net.append(layer)
                self.net.append(nl)
            else:
                # init_weights_normal(layer)
                self.net.append(layer)
            if hidden:
                if batch_norm:
                    self.net.append(nn.BatchNorm1d(num_features=self.hidden_features[i + 1]))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=False, gaussian_variance=38.0):
        """
        Initilization of a positional encoder.

        Parameters
        ----------
        num_encoding_functions: int
        include_input: bool
        log_sampling: bool
        normalize: bool
        input_dim: int
        gaussian_pe: bool
        gaussian_variance: float
        """
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1 / self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx] * func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)


class SIREN(nn.Module):
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, w0=30.0):
        """
        Initialization of a SIREN.

        Parameters
        ----------
        in_features: int
        out_features: int
        num_hidden_layers: int
        hidden_features: int
        outermost_linear: bool
        w0: float
        """
        super(SIREN, self).__init__()

        nl = Sine(w0)
        first_nl = FirstSine(w0)
        self.weight_init = functools.partial(sine_init, w0=w0)
        self.first_layer_init = first_layer_sine_init

        self.net = []
        self.net.append(nn.Sequential(
            nn.Linear(in_features, hidden_features),
            first_nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features),
            ))
        else:
            self.net.append(nn.Sequential(
                nn.Linear(hidden_features, out_features),
                nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if self.first_layer_init is not None:
            self.net[0].apply(self.first_layer_init)

    def forward(self, coords):
        output = self.net(coords)
        return output


class FourierNet(nn.Module):
    def __init__(self, channels=1, layers=[3, 2], params=[256, 256], nl=['sin', 'sin'], w0=[40, 30],
                 force_symmetry=False):
        """
        Initialization of a FourierNet.

        Parameters
        ----------
        channels: int
        layers: list
            [number of layers in the modulant, number of layers in the envelope]
        params: list
            [number of hidden dimensions in the modulant, number of hidden dimensions in the envelope]
        nl: list
            [nonlinearities in the modulant, nonlinearities in the envelope]
        w0: list
            [w0 for the modulant, w0 for the envelope]
        force_symmetry: bool
        """
        super(FourierNet, self).__init__()

        in_features = 3

        self.force_symmetry = force_symmetry
        if force_symmetry:
            self.symmetrizer = Symmetrizer()

        self.net_modulant = build_model_fouriernet(nl[0], in_features, channels, layers[0], params[0], w0[0])
        self.net_enveloppe = build_model_fouriernet(nl[1], in_features, channels, layers[1], params[1], w0[1])

    def forward(self, coords):
        coords_clone = torch.clone(coords).to(coords.device)

        # Add a dummy dimension when the number of dimensions is only 2
        if coords_clone.dim() == 2:
            coords_clone = coords_clone.unsqueeze(0)

        if self.force_symmetry:
            self.symmetrizer.initialize(coords_clone)
            coords_clone = self.symmetrizer.symmetrize_input(coords_clone)

        output = torch.exp(self.net_enveloppe(coords_clone)) * self.net_modulant(coords_clone)

        if self.force_symmetry:
            output = self.symmetrizer.antisymmetrize_output(output)

        return output


def build_model_fouriernet(nl, in_features, channels, layers, params, w0):
    if nl == 'sin':
        return SIREN(in_features, 2 * channels, layers, params, outermost_linear=True, w0=w0)
    elif nl == 'ReLU':
        return FCBlock(in_features, [params] * layers, 2 * channels, nonlinearity='relu')
    else:
        raise NotImplementedError


class Symmetrizer():
    def __init__(self):
        """
        Initialization of a Symmetrizer, to enforce symmetry in Fourier space.
        """
        self.half_space_indicator = None
        self.DC_indicator = None

    def initialize(self, coords):
        self.half_space_indicator = which_half_space(coords)
        self.DC_indicator = where_DC(coords)

    def symmetrize_input(self, coords):
        # Place the "negative" coords in the "positive" half space
        coords[self.half_space_indicator] = -coords[self.half_space_indicator]
        return coords

    def antisymmetrize_output(self, output):
        # Flip the imaginary part on the "negative" half space and force DC component to be zero
        batch_sz = output.shape[0]
        N = output.shape[1]
        output = output.reshape(batch_sz, N, -1, 2)
        # output.shape = Batch, N, channels, 2
        channels = output.shape[2]
        half_space = self.half_space_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        DC = self.DC_indicator.reshape(batch_sz, N, 1, 1).repeat(1, 1, channels, 2)
        output_sym = torch.where(half_space, torch.cat((output[..., 0].unsqueeze(-1),
                                                        -output[..., 1].unsqueeze(-1)), dim=-1), output)
        output_sym_DC = torch.where(DC, torch.cat((output_sym[..., 0].unsqueeze(-1),
                                                torch.zeros_like(output_sym[..., 0].unsqueeze(-1))), dim=-1),
                                 output_sym)
        output_sym_DC = output_sym_DC.reshape(batch_sz, N, -1)
        return output_sym_DC


def which_half_space(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_xyz = (x < -eps)
    slab_yz = torch.logical_and(torch.logical_and(x > -eps, x < eps), y < -eps)
    slab_z = torch.logical_and(torch.logical_and(torch.logical_and(x > -eps, x < eps),
                                                 torch.logical_and(y > -eps, y < eps)), z < -eps)

    return torch.logical_or(slab_xyz, torch.logical_or(slab_yz, slab_z))


def where_DC(coords, eps=1e-6):
    x = coords[..., 0]
    y = coords[..., 1]
    z = coords[..., 2]

    slab_x = torch.logical_and(x > -eps, x < eps)
    slab_y = torch.logical_and(y > -eps, y < eps)
    slab_z = torch.logical_and(z > -eps, z < eps)

    return torch.logical_and(slab_x, torch.logical_and(slab_y, slab_z))


def conv3x3(in_planes, out_planes, stride=1, bias=True):
    """
    3x3 convolution with padding.

    Parameters
    ----------
    in_planes: int
    out_planes: int
    stride: int
    bias: bool

    Returns
    -------
    out: torch.nn.Module
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=bias)


class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, batch_norm, triple=False):
        """
        Initialization of a double convolutional block.

        Parameters
        ----------
        in_size: int
        out_size: int
        batch_norm: bool
        triple: bool
        """
        super(DoubleConvBlock, self).__init__()
        self.batch_norm = batch_norm
        self.triple = triple

        self.conv1 = conv3x3(in_size, out_size)
        self.conv2 = conv3x3(out_size, out_size)
        if triple:
            self.conv3 = conv3x3(out_size, out_size)

        self.relu = nn.ReLU(inplace=True)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(out_size)
            self.bn2 = nn.BatchNorm2d(out_size)
            if triple:
                self.bn3 = nn.BatchNorm2d(out_size)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        if self.triple:
            out = self.relu(out)

            out = self.conv3(out)
            if self.batch_norm:
                out = self.bn3(out)

        out = self.relu(out)

        return out


class CNNEncoderVGG16(nn.Module):
    def __init__(self, in_channels=3, batch_norm=False, pretrained=False, flip_images=False, high_res=False):
        """
        Initialization of a VGG16-like encoder.

        Parameters
        ----------
        in_channels: int
        batch_norm: bool
        pretrained: bool
        flip_images: bool
        high_res: bool
        """
        super(CNNEncoderVGG16, self).__init__()
        self.pretrained = pretrained

        if pretrained:
            self.in_channels = 3
        else:
            self.in_channels = in_channels
        if high_res:
            self.feature_channels = [64, 128, 256, 256, 1024, 2048]
        else:
            self.feature_channels = [64, 128, 256, 1024, 2048]
        self.flip_images = flip_images

        self.net = []

        # VGG16 first 3 layers
        prev_channels = self.in_channels
        next_channels = self.feature_channels[0]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[1]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[2]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm, triple=True)
        )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )

        if pretrained:
            vgg16 = models.vgg16(pretrained=True)

        # Rest of encoder
        prev_channels = next_channels
        next_channels = self.feature_channels[3]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool2d(kernel_size=2)
        )
        prev_channels = next_channels
        next_channels = self.feature_channels[4]
        self.net.append(
            DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
        )
        self.net.append(
            nn.AvgPool2d(kernel_size=2)
        )
        if high_res:
            prev_channels = next_channels
            next_channels = self.feature_channels[5]
            self.net.append(
                DoubleConvBlock(prev_channels, next_channels, batch_norm=batch_norm)
            )
            self.net.append(
                nn.AvgPool2d(kernel_size=2)
            )
        self.net.append(
            nn.MaxPool2d(kernel_size=2)
        )

        self.net = nn.Sequential(*self.net)

        if pretrained:
            self.state_dict_vgg16 = {'cnn_encoder.net.0.conv1.weight': vgg16.state_dict()['features.0.weight'],
                          'cnn_encoder.net.0.conv1.bias': vgg16.state_dict()['features.0.bias'],
                          'cnn_encoder.net.0.conv2.weight': vgg16.state_dict()['features.2.weight'],
                          'cnn_encoder.net.0.conv2.bias': vgg16.state_dict()['features.2.bias'],
                          'cnn_encoder.net.2.conv1.weight': vgg16.state_dict()['features.5.weight'],
                          'cnn_encoder.net.2.conv1.bias': vgg16.state_dict()['features.5.bias'],
                          'cnn_encoder.net.2.conv2.weight': vgg16.state_dict()['features.7.weight'],
                          'cnn_encoder.net.2.conv2.bias': vgg16.state_dict()['features.7.bias'],
                          'cnn_encoder.net.4.conv1.weight': vgg16.state_dict()['features.10.weight'],
                          'cnn_encoder.net.4.conv1.bias': vgg16.state_dict()['features.10.bias'],
                          'cnn_encoder.net.4.conv2.weight': vgg16.state_dict()['features.12.weight'],
                          'cnn_encoder.net.4.conv2.bias': vgg16.state_dict()['features.12.bias'],
                          'cnn_encoder.net.4.conv3.weight': vgg16.state_dict()['features.14.weight'],
                          'cnn_encoder.net.4.conv3.bias': vgg16.state_dict()['features.14.bias']}

            # Renormalize and be consistent with what is expected from the ImageNet dataset
            self.register_buffer('means', torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3))
            self.register_buffer('stds', torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3))

        else:
            self.register_buffer('means', torch.tensor([0.45] * self.in_channels).reshape(1, self.in_channels))
            self.register_buffer('stds', torch.tensor([0.226] * self.in_channels).reshape(1, self.in_channels))

    def get_out_shape(self, h, w):
        """
        Returns the expected number of dimensions at the output og the CNN.

        Parameters
        ----------
        h: int
        w: int

        Returns
        -------
        out: int
        """
        if self.pretrained:
            return self.forward(torch.rand(1, 1, h, w)).shape[1:]
        else:
            return self.forward(torch.rand(1, self.in_channels, h, w)).shape[1:]

    def normalize_repeat(self, input):
        """
        Normalize input.

        Parameters
        ----------
        input: torch.Tensor

        Returns
        -------
        out: torch.Tensor
        """
        N = input.shape[0]
        if self.pretrained:
            C_in = 1
            C_out = 3
        else:
            C_in = self.in_channels
            C_out = self.in_channels
        # input: N, C_in, H, W
        # self.means/std: N, C_out
        means = torch.mean(input, (2, 3))  # N, C_in
        stds = torch.std(input, (2, 3))  # N, C_in
        alphas = (self.stds / stds).reshape(N, C_out, 1, 1)  # N, C_out, 1, 1
        c = (self.means.reshape(1, C_out, 1, 1) / alphas -
             means.reshape(N, C_in, 1, 1)).reshape(N, C_out, 1, 1)
        return alphas * (input.repeat(1, int(C_out/C_in), 1, 1) + c)

    def augment_batch(self, input):
        """
        Augment the dataset (batch-zise) with flipped images.

        Parameters
        ----------
        input: torch.Tensor (B, 1, S, S)

        Returns
        -------
        out: torch.Tensor (2*B, 1, S, S)
        """
        batch_size = input.shape[0]
        self.flip_status = torch.zeros((2 * batch_size))
        self.flip_status[batch_size:] = 1

        return torch.cat((input, torch.flip(input, [2, 3])), 0)

    def forward(self, input):
        if self.flip_images:
            input_augmented = self.augment_batch(input)
        else:
            input_augmented = input
        out = self.net(self.normalize_repeat(input_augmented))
        return out


class OrientationPredictor(nn.Module):
    def __init__(self, cryoai):
        """
        Initialization of an OrientationPredictor.

        Parameters
        ----------
        cryoai: CryoAI
        """
        super(OrientationPredictor, self).__init__()

        # only valid for a specific configuration of cryoai for now
        self.gaussian_filters = cryoai.gaussian_filters
        self.encoder = cryoai.cnn_encoder
        self.orientation_encoder = cryoai.orientation_encoder
        self.orientation_regressor = cryoai.orientation_regressor
        self.latent_to_rot3d_fn = cryoai.latent_to_rot3d_fn
        self.shift_encoder = cryoai.shift_encoder


    def forward(self, in_dict):
        proj = in_dict['proj_input']
        proj = self.gaussian_filters(proj)
        latent_code = torch.flatten(self.encoder(proj), start_dim=1)

        latent_code_pose = self.orientation_encoder(latent_code)
        latent_code_prerot = self.orientation_regressor(latent_code_pose)
        pred_rotmat = self.latent_to_rot3d_fn(latent_code_prerot)

        shift_params = self.shift_encoder(latent_code)
        pred_shift_params = {'shiftX': shift_params[..., 0].reshape(-1),
                             'shiftY': shift_params[..., 1].reshape(-1)}

        return {'rotmat': pred_rotmat,
                'pred_shift_params': pred_shift_params}
