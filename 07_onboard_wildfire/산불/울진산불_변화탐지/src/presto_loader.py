"""Presto(nasaharvest) 모델을 데이터-export 의존(ee·openmapflow·gcloud) 없이 로드.
멀티템포럴 임베딩 레버용 — 봄 개엽 intra-year 신호를 시계열로 포착 시도.

핵심: presto 패키지 __init__ 이 데이터파이프라인(ee 등)을 강제 import 하므로,
meta-path finder 로 그 외부 패키지들을 stub 처리한 뒤 모델·상수만 로드한다.
가중치=presto_data/default_model.pt (repo 에서 받음).
"""
import sys, types, importlib.abc, importlib.machinery, os
import torch


class _Any:
    def __getattr__(s, k): return s
    def __call__(s, *a, **k): return s
    def __getitem__(s, k): return s


_any = _Any()


class _Stub(types.ModuleType):
    __path__ = []  # 패키지처럼 → 서브모듈 import 허용

    def __getattr__(s, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return _any


_STUB_PREFIX = ("ee", "openmapflow", "xarray", "geopandas",
                "webdataset", "hurry", "shapely", "google")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(s, name, path, target=None):
        if name.split(".")[0] in _STUB_PREFIX:
            return importlib.machinery.ModuleSpec(name, s)
        return None

    def create_module(s, spec): return _Stub(spec.name)
    def exec_module(s, m): pass


def _install_stubs():
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _StubFinder())


_install_stubs()

from presto.presto import Presto  # noqa: E402
from presto.dataops.utils import construct_single_presto_input  # noqa: E402
from presto.dataops.pipelines.s1_s2_era5_srtm import NORMED_BANDS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "..", "presto_data", "default_model.pt")
# 우리 ko_refl 밴드 = [B02,B03,B04,B8A,B11,B12] → Presto S2 밴드명
OUR_S2_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]


def load_presto(weights=WEIGHTS, device="cpu"):
    m = Presto.construct()
    m.load_state_dict(torch.load(weights, map_location=device))
    m.eval().to(device)
    return m


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_presto(device=dev)
    print("✅ Presto 로드 OK — 파라미터:", sum(p.numel() for p in m.parameters()), "| dev", dev)
    # 합성 s2 시계열 (T=12개월, 6밴드) 1픽셀 forward 검증
    T = 12
    s2 = torch.rand(T, len(OUR_S2_BANDS))
    x, mask, dw = construct_single_presto_input(s2=s2, s2_bands=OUR_S2_BANDS)
    print("construct 출력 x:", tuple(x.shape), "mask:", tuple(mask.shape), "dw:", tuple(dw.shape))
    latlons = torch.tensor([[36.5, 129.0]]).float().to(dev)
    months = torch.tensor([3]).long().to(dev)  # 시작월(봄)
    with torch.no_grad():
        emb = m.encoder(
            x.unsqueeze(0).to(dev), dynamic_world=dw.unsqueeze(0).long().to(dev),
            latlons=latlons, mask=mask.unsqueeze(0).to(dev), month=months, eval_task=True)
    print("✅ 임베딩 shape:", tuple(emb.shape), "→ Presto 임베딩 차원 =", emb.shape[-1])
