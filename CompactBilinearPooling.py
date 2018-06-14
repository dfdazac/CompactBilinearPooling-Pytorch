import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import pytorch_fft.fft.autograd as afft


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if rand_h_1 is None:
            torch.random.manual_seed(1)
            rand_h_1 = torch.randint(output_dim, size=(self.input_dim1,))
        if rand_s_1 is None:
            torch.random.manual_seed(3)
            rand_s_1 = 2 * torch.randint(2, size=(self.input_dim1,)) - 1

        self.sparse_sketch_matrix1 = self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim))

        if rand_h_2 is None:
            torch.random.manual_seed(5)
            rand_h_2 = torch.randint(output_dim, size=(self.input_dim2,))
        if rand_s_2 is None:
            torch.random.manual_seed(7)
            rand_s_2 = 2 * torch.randint(2, size=(self.input_dim2,)) - 1

        self.sparse_sketch_matrix2 = self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim))

        self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.to(self.device)
        self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.to(self.device)

    def forward(self, bottom1, bottom2):
        """
        bottom1: 1st input, 4D Tensor of shape [batch_size, input_dim1, height, width].
        bottom2: 2nd input, 4D Tensor of shape [batch_size, input_dim2, height, width].
        """
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1_real, fft1_imag = afft.Fft()(sketch_1, Variable(torch.zeros(sketch_1.size())).cuda())
        fft2_real, fft2_imag = afft.Fft()(sketch_2, Variable(torch.zeros(sketch_2.size())).cuda())

        temp_rr, temp_ii = fft1_real.mul(fft2_real), fft1_imag.mul(fft2_imag)
        fft_product_real = temp_rr - temp_ii
        fft_product_imag = temp_rr + temp_ii

        cbp_flat = afft.Ifft()(fft_product_real, fft_product_imag)[0]

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim == 1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bottom1 = torch.randn(128, 512, 14, 14).to(device)
    bottom2 = torch.randn(128, 512, 14, 14).to(device)

    layer = CompactBilinearPooling(512, 512, 8000).to(device)

    layer.train()

    out = layer(bottom1, bottom2)
