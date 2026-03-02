import torch
import torch.nn as nn


EPS = 1e-8


def _ema_series(x: torch.Tensor, span: int) -> torch.Tensor:
    alpha = 2.0 / (float(span) + 1.0)
    out = [x[:, 0]]
    for i in range(1, x.shape[1]):
        out.append(alpha * x[:, i] + (1.0 - alpha) * out[-1])
    return torch.stack(out, dim=1)


def _tail_k_mean(series: torch.Tensor, k: int, name: str) -> torch.Tensor:
    if series.shape[1] < k:
        raise ValueError(f"Not enough history for rolling({k}) mean in feature {name}: got {series.shape[1]}")
    return series[:, -k:].mean(dim=1)


class KPIWindowEncoder(nn.Module):
    def __init__(self, feature_names: list[str], kpi_type: str = "full_short", k: int = 5):
        super().__init__()
        self.feature_names = list(feature_names)
        self.kpi_type = str(kpi_type)
        self.k = int(k)
        self.periods = [21, 63, 126] if "short" in self.kpi_type.lower() else [21, 63, 126, 189]

    def _check_window(self, prices: torch.Tensor):
        min_len = max(self.periods) + self.k
        if prices.shape[1] < min_len:
            raise ValueError(f"Window length must be >= {min_len}, got {prices.shape[1]}")

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        if prices.dim() == 1:
            prices = prices.unsqueeze(0)
        if prices.dim() != 2:
            raise ValueError("Expected prices shape [batch, window]")

        prices = prices.float()
        self._check_window(prices)

        ret1 = (prices[:, 1:] - prices[:, :-1]) / torch.clamp(prices[:, :-1], min=EPS)
        feats = {}

        n = prices.shape[1]
        for p in self.periods:
            roi_series = (prices[:, p:] - prices[:, :-p]) / torch.clamp(prices[:, :-p], min=EPS)
            feats[f"past_profitability_{p}d"] = _tail_k_mean(roi_series, self.k, f"past_profitability_{p}d")

            vol_series = ret1.unfold(1, p, 1).std(dim=2, unbiased=True) * torch.sqrt(
                torch.tensor(252.0, device=prices.device)
            )
            feats[f"volatility_{p}d"] = _tail_k_mean(vol_series, self.k, f"volatility_{p}d")

            avg_series = prices.unfold(1, p, 1).mean(dim=2)
            feats[f"avg_price_{p}d"] = _tail_k_mean(avg_series, self.k, f"avg_price_{p}d")

            m_series = prices[:, p:] - prices[:, :-p]
            feats[f"m_{p}d"] = _tail_k_mean(m_series, self.k, f"m_{p}d")

            roc_series = m_series / torch.clamp(prices[:, :-p], min=EPS)
            feats[f"roc_{p}d"] = _tail_k_mean(roc_series, self.k, f"roc_{p}d")

            sharpe_series = roi_series / torch.clamp(vol_series, min=EPS)
            sharpe_series = torch.nan_to_num(sharpe_series, nan=0.0, posinf=0.0, neginf=0.0)
            feats[f"sharpe_{p}d"] = _tail_k_mean(sharpe_series, self.k, f"sharpe_{p}d")

            min_series = prices.unfold(1, p, 1).min(dim=2).values
            max_series = prices.unfold(1, p, 1).max(dim=2).values
            feats[f"min_{p}d"] = _tail_k_mean(min_series, self.k, f"min_{p}d")
            feats[f"max_{p}d"] = _tail_k_mean(max_series, self.k, f"max_{p}d")

            exp_mean_series = _ema_series(prices, p)
            feats[f"exp_mean_{p}d"] = _tail_k_mean(exp_mean_series, self.k, f"exp_mean_{p}d")

        ema12_series = _ema_series(prices, 12)
        ema26_series = _ema_series(prices, 26)
        macd_series = ema12_series - ema26_series
        feats["MACD"] = _tail_k_mean(macd_series, self.k, "MACD")

        diff = torch.cat([torch.zeros(prices.shape[0], 1, device=prices.device), prices[:, 1:] - prices[:, :-1]], dim=1)
        up = torch.clamp(diff, min=0.0)
        down = torch.clamp(-diff, min=0.0)
        up_ema = _ema_series(up, 14)
        down_ema = _ema_series(down, 14)
        rs = up_ema / torch.clamp(down_ema, min=EPS)
        rsi_series = 100.0 - 100.0 / (1.0 + rs)
        rsi_series = torch.nan_to_num(rsi_series, nan=0.0, posinf=0.0, neginf=0.0)
        feats["rsi_14"] = _tail_k_mean(rsi_series, self.k, "rsi_14")

        dco_n = 22
        mid_index = int(dco_n / 2 + 1)
        rolling_mean = prices.unfold(1, dco_n, 1).mean(dim=2)
        shifted = prices[:, dco_n - 1 - mid_index : n - mid_index]
        dco_series = shifted - rolling_mean
        feats["dco_22"] = _tail_k_mean(dco_series, self.k, "dco_22")

        ordered = []
        missing = []
        for name in self.feature_names:
            if name in feats:
                ordered.append(feats[name].unsqueeze(1))
            else:
                missing.append(name)

        if missing:
            raise ValueError(f"Unsupported feature(s) for torch KPI encoder: {missing}")

        return torch.cat(ordered, dim=1)


class WindowToFeatureHeadModel(nn.Module):
    def __init__(self, encoder: KPIWindowEncoder, head_model: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head_model = head_model

    def forward(self, prices: torch.Tensor) -> torch.Tensor:
        x = self.encoder(prices)
        out = self.head_model(x)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, list):
            out = out[0]
        return out
