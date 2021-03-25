import torch
import torch.fft as fft

nfft = 512
fcutoff = 257

fourier_basis = torch.rfft(torch.eye(nfft), signal_ndim=1, onesided=False)
forward_basis = fourier_basis[:fcutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis.shape[1])

fourier_basis_new = torch.view_as_real(fft.fft(torch.eye(nfft), dim=1))
forward_basis_new = fourier_basis_new[:fcutoff].permute(2, 0, 1).reshape(-1, 1, fourier_basis_new.shape[1])

diff = forward_basis-forward_basis_new

print('basis diff', diff.mean())

assert diff.mean() == 0.0

print('success!')
