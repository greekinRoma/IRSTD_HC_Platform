import torch
from torch import nn
class IPI(nn.Module):
    def __init__(self,dw=50,dh=50,x_step=10,y_step=10):
        super().__init__()
        self.dw = dw
        self.dh = dh
        self.x_step = x_step
        self.y_step = y_step
        self.sample = nn.Unfold(kernel_size=(dh,dw),stride=(x_step,y_step))
        self.tol = 1e-6
        self.maxIter = 1000
        self.continuationFlag = -1
        self.mu = 1e-3

    def APG_IR_torch(self,
                     D,
                    lam,                   # 对应原代码中的 lambda
                    maxIter=1000,
                    tol=1e-6,
                    lineSearchFlag=0,      # 原代码中有相关参数，但本实现中默认不启用
                    continuationFlag=1,
                    eta=0.9,              # line search 参数，若 lineSearchFlag=1 才会用到
                    mu=1e-3,
                    outputFileName=None    # 若需要把每次迭代信息输出到文件，可在代码中实现
                    ):
        """
        基于 PyTorch 的 APG_IR 算法实现，用于低秩矩阵 + 稀疏误差分解。
        
        参数解释（与原 Matlab 代码对应）:
            D (torch.Tensor): 观测数据矩阵 (m x n)
            lam (float):      对应稀疏项的正则化系数 (lambda)
            maxIter (int):    最大迭代次数
            tol (float):      收敛判据阈值
            lineSearchFlag (int):  原代码中的线搜索标识 (这里不做实现，保留参数以防后续扩展)
            continuationFlag (int): 是否使用 continuation 技巧 (1: 使用; 0: 不使用)
            eta (float):      line search 的步长衰减参数 (若 lineSearchFlag=1 才会用到)
            mu (float):       松弛参数 (若 continuationFlag=0，则使用该固定值)
            outputFileName (str): 若需要将迭代信息输出到文件，可传文件名，本文不做详细实现

        返回:
            A_hat (torch.Tensor): 低秩矩阵
            E_hat (torch.Tensor): 稀疏误差矩阵
        """

        # ------------- 一些辅助函数和初始处理 -------------
        
        # “正部”函数，相当于 Matlab 中的 pos(x) = max(x, 0)
        # 也可用 torch.nn.functional.relu(x)
        def pos(x):
            return torch.clamp(x, min=0)
        
        # 如果用户没指定 tol 或 maxIter，可自行设置默认值，这里已在函数参数处完成
        
        m, n = D.shape
        
        # 初始化变量
        t_k = 1.0
        t_km1 = 1.0
        
        # APG 里的 Lipschitz 常数（平方）
        # 原代码 tau_0=2，仅作为初始常数
        tau_0 = 2.0
        tau_k = tau_0

        # X^{k-1} = (A^{k-1}, E^{k-1})
        X_km1_A = torch.zeros_like(D)
        X_km1_E = torch.zeros_like(D)

        # X^{k} = (A^{k}, E^{k})
        X_k_A = torch.zeros_like(D)
        X_k_E = torch.zeros_like(D)

        # 处理 continuationFlag
        # 若使用 continuation，则每次迭代会动态减小 mu_k
        # 若不使用，则 mu_k = mu
        if continuationFlag == 1:
            # 参考原文：mu_0 = norm(D), mu_k = 0.99 * mu_0, mu_bar = 1e-9 * mu_0
            # 但原代码在后面又改为: mu_k = s(2), mu_bar = 0.005 * s(4)
            # 这里我们先做一次 SVD，看能不能取到第2、4大奇异值
            U, S, V = torch.svd(D, some=True)
            # S 是奇异值向量，S[0]为最大奇异值，S[1]为第二大...
            # 注意数据维度不够时需要判断 S 的长度
            if len(S) >= 4:
                mu_k = S[1].item()      # 对应 Matlab s(2)
                mu_bar = 0.005 * S[3].item()   # 对应 Matlab s(4)
            else:
                # 如果数据维度较小，兜底
                mu_k = 0.99 * torch.norm(D).item()
                mu_bar = 1e-9 * torch.norm(D).item()
        else:
            mu_k = mu
            mu_bar = mu  # 仅做个兜底，不会被真正使用

        # 额外的收敛/停止控制
        NOChange_counter = 0
        pre_rank = 0
        pre_cardE = 0

        converged = False
        numIter = 0

        # ------------- 主循环开始 -------------
        while not converged:
            # 计算 Y_k = X_k + ((t_{k-1} - 1) / t_k) * (X_k - X_{k-1})
            Y_k_A = X_k_A + ((t_km1 - 1.0) / t_k) * (X_k_A - X_km1_A)
            Y_k_E = X_k_E + ((t_km1 - 1.0) / t_k) * (X_k_E - X_km1_E)

            # 梯度 G_k = Y_k - (1/tau_k) * (Y_k_A + Y_k_E - D)
            T = (Y_k_A + Y_k_E - D) / tau_k  # 先算这个避免多次重复
            G_k_A = Y_k_A - T
            G_k_E = Y_k_E - T

            # 对 A 部分进行奇异值阈值化
            U, Svals, V = torch.svd(G_k_A, some=True)
            # 计算 Svals - mu_k / tau_k 并取正部
            s_threshold = pos(Svals - mu_k / tau_k)
            # 重新组装成矩阵
            # Torch 的 svd 返回 U, S, V，其中矩阵 G_k_A ~ U * diag(Svals) * V^T
            # 所以更新 A_hat = U * diag(s_threshold) * V^T
            S_diag = torch.diag(s_threshold)
            X_kp1_A = U.mm(S_diag).mm(V.t())

            # 对 E 部分进行软阈值化
            # X_kp1_E = sign(G_k_E) * pos(|G_k_E| - lam * mu_k / tau_k)
            X_kp1_E = torch.sign(G_k_E) * pos(G_k_E.abs() - lam * mu_k / tau_k)

            # 计算 A 的秩
            rankA = (Svals > (mu_k / tau_k)).sum().item()

            # 计算 E 的非零元素个数
            cardE = (X_kp1_E.abs() > 0).sum().item()

            # 更新 t_k
            t_kp1 = 0.5 * (1.0 + torch.sqrt(torch.tensor(1.0 + 4.0 * t_k * t_k)))

            # 计算更新后的 S_k+1 用于判断收敛
            temp_ = X_kp1_A + X_kp1_E - Y_k_A - Y_k_E
            S_kp1_A = tau_k * (Y_k_A - X_kp1_A) + temp_
            S_kp1_E = tau_k * (Y_k_E - X_kp1_E) + temp_

            # Fro 范数
            numerator = torch.cat((S_kp1_A.view(-1), S_kp1_E.view(-1))).norm(p=2)
            denominator = tau_k * max(1.0, 
                            torch.cat((X_kp1_A.view(-1), X_kp1_E.view(-1))).norm(p=2))
            stoppingCriterion = numerator / denominator

            # 判断是否收敛
            if stoppingCriterion <= tol:
                converged = True

            # 若使用 continuation，就要让 mu_k 往下收敛
            if continuationFlag == 1:
                mu_k = max(0.9 * mu_k, mu_bar)

            # 更新历史量，进入下一迭代
            t_km1 = t_k
            t_k = t_kp1.item()  # t_kp1 是一个 0-dim 张量，用 .item() 取值
            X_km1_A = X_k_A
            X_km1_E = X_k_E
            X_k_A = X_kp1_A
            X_k_E = X_kp1_E

            numIter += 1

            # ------- 额外终止条件：若 rank(A) 连续多次不变且 E 的变化很小，就退出 -------
            if pre_rank == rankA:
                NOChange_counter += 1
                if NOChange_counter > 10 and abs(cardE - pre_cardE) < 20:
                    converged = True
            else:
                NOChange_counter = 0
                pre_cardE = cardE

            pre_rank = rankA

            # ------- 若 rank(A) 大于 0.3 * min(m, n)，强行停止 -------
            if rankA > 0.3 * min(m, n):
                converged = True

            # 若需要输出到文件或屏幕，可自行写
            # if outputFileName is not None:
            #     with open(outputFileName, 'a') as f:
            #         f.write(f"Iter {numIter}, rank(A)={rankA}, card(E)={cardE}, stoppingCriterion={stoppingCriterion}\n")

            # 最大迭代次数限制
            if not converged and numIter >= maxIter:
                print("Warning: Reached maximum iterations without full convergence.")
                converged = True

        # 迭代结束，返回结果
        A_hat = X_k_A
        E_hat = X_k_E
        return A_hat, E_hat


    def forward(self,inp):
        inp = inp * 255.
        b,c,h,w = inp.shape
        D     = self.sample(inp).view(b,c,self.dh*self.dw,-1)
        D_min = D.min(dim=1, keepdim=True)[0].min(dim=2,keepdim=True)[0] 
        D_max = D.max(dim=1, keepdim=True)[0].max(dim=2,keepdim=True)[0]
        D_normalized = (D - D_min) / (D_max - D_min + 1e-8)
        _,_,m1,n1 = D.shape
        lamda = 1/torch.sqrt(torch.tensor(max(m1,n1)/1.))
        outputs = []
        for d in D_normalized:
            d = torch.mean(d,dim=0)
            m,n = h,w
            _, E1 = self.APG_IR_torch(d,lamda)
            EE = torch.zeros(m, n, 100)
            C = torch.zeros(m, n).long()
            temp1 = torch.zeros(self.dh,self.dw)
            index = 0
            for i in range(0, m - self.dh + 1, self.y_step):
                for j in range(0, n - self.dw + 1, self.x_step):
                    index += 1
                    C[i:i+self.dh, j:j+self.dw] += 1
                    temp1 = E1[:, index - 1].reshape(self.dh, self.dw)
                    for ii in range(self.dh):
                        for jj in range(self.dw):
                            EE[i+ii, j+jj, C[i+ii, j+jj] - 1] = temp1[ii, jj]
            # Process AA and EE to calculate A_hat and E_hat
            E_hat = torch.zeros((m, n))

            for i in range(m):
                for j in range(n):
                    if C[i, j] > 0:
                        E_hat[i, j] = torch.median(EE[i, j, :C[i, j]])
            outputs.append(E_hat)
        outputs =torch.stack(outputs,dim=0)
        outputs = outputs.unsqueeze(1)/(outputs.max()+1e-10)
        print(outputs.max())
        return outputs