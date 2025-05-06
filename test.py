import torch

d, d_k, s, r = 4, 2, 2, 1          # toy GQA group: 2 heads share one key
WQ = torch.tensor([[2,0,1,0],
                   [0,2,0,1],
                   [1,0,2,0],
                   [0,1,0,2.]],dtype=torch.float)
WK = torch.tensor([[1,0],
                   [0,1],
                   [1,0],
                   [0,1.]],dtype=torch.float)

Q, R = torch.linalg.qr(WQ)         # wide → tall QR
Kbar = Q.T @ WK
Cbar = R[:,:d_k].T @ Kbar
Λ, V  = torch.linalg.eigh(Cbar)    # both 3√5
U_r   = V[:,[-1]] * Λ[-1].sqrt()   # keep top eigvec

WK_new = WK @ U_r
WQ_new = WQ @ torch.linalg.inv(R[:,:d_k]) @ U_r

print('diag product =', (WQ_new.T @ WK_new))   # → [9.]
