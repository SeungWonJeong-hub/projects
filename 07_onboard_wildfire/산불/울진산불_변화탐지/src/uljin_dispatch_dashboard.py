"""울진 2022 — V3: 대피 정보 송출 대시보드(자체완결 인터랙티브 HTML).

during-fire 파이프라인(열점씨앗→물리확산 V1→실측기상 V2→예측대피)의 마지막 계층=송출.
읍면/현장이 바로 읽을 수 있게: '어디에 언제 불이 닿고(골든타임), 어디로 대피하며(도주로),
어느 대피소가 살아남는가'를 하나의 인터랙티브 지도로. 외부 타일 불필요(오프라인 동작).

입력(전부 파이프라인 산출):
  runs/uljin_dispatch_payload.json  = 소각경계 + 위험건물(예측마진·골든타임·탈출거리) + 도주로좌표 + 대피소/읍시
                                      (uljin_evacuation_forecast.py 산출, V2 시변풍 T_fire 기반)
  runs/uljin_tfire_v2.npz           = 예측 도달시각 필드(배경 히트맵: '발화 후 몇 시간에 불 도달')
출력: runs/uljin_dispatch_dashboard.html  (plotly.js 인라인 → 단일 파일로 열림)
"""
import json, numpy as np
from pyproj import Transformer
import plotly.graph_objects as go

HERE = __file__.rsplit("/", 2)[0]
PAY = HERE + "/runs/uljin_dispatch_payload.json"
TF = HERE + "/runs/uljin_tfire_v2.npz"
OUT = HERE + "/runs/uljin_dispatch_dashboard.html"

d = json.load(open(PAY))
blds = d["buildings"]; dests = d["dests"]; routes = d["routes"]; rings = d["burn_rings"]
bx = np.array([b["x"] for b in blds]); by = np.array([b["y"] for b in blds])
bm = np.array([b["margin"] for b in blds], float)
btf = np.array([(b["t_fire"] if b["t_fire"] is not None else np.nan) for b in blds], float)
besc = np.array([b["esc_km"] for b in blds], float)

# --- 배경: 예측 도달시각 필드(5186 격자 → lon/lat 근사축) ---
z = np.load(TF, allow_pickle=True)
T = z["T"].astype(float); minx = float(z["minx"]); maxy = float(z["maxy"]); cell = float(z["cell"])
H, Wd = int(z["H"]), int(z["W"]); span = float(z["span_h"])
cx = minx + (np.arange(Wd) + 0.5) * cell; cy = maxy - (np.arange(H) + 0.5) * cell
tr = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
lon_1d, _ = tr.transform(cx, np.full_like(cx, cy[H // 2]))      # 중앙행 기준 lon축
_, lat_1d = tr.transform(np.full_like(cy, cx[Wd // 2]), cy)      # 중앙열 기준 lat축
step = 2                                                          # 다운샘플(파일경량)
Tvis = np.where(np.isfinite(T) & (T <= span), T, np.nan)[::step, ::step]
lon_s, lat_s = lon_1d[::step], lat_1d[::step]

# ---------- Plotly figure ----------
fig = go.Figure()
# 1) 도달시각 히트맵(배경) — 색=발화 후 도달시간(h)
fig.add_trace(go.Heatmap(z=Tvis, x=lon_s, y=lat_s, colorscale="Inferno", reversescale=False,
    opacity=0.78, zmin=0, zmax=span, colorbar=dict(title="예측 화재<br>도달 (h)", x=1.005, len=0.8),
    hovertemplate="예측 도달 %{z:.0f}h<extra></extra>", name="도달시각"))
# 2) 소각 경계(흰 실선)
for r in rings:
    if len(r) < 3: continue
    xs = [p[0] for p in r] + [r[0][0]]; ys = [p[1] for p in r] + [r[0][1]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="white", width=1.2),
        opacity=0.7, hoverinfo="skip", showlegend=False))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="white", width=1.5),
    name=f"소각 경계 (~{d['burn_ha']:,} ha)"))
# 3) 예측 도주로(검은 선)
for i, rt in enumerate(routes):
    px = [p[0] for p in rt["path"]]; py = [p[1] for p in rt["path"]]
    fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="#111", width=2),
        opacity=0.85, hoverinfo="skip", showlegend=(i == 0), name="예측 도주로"))
# 4) 최고위험 마을(빨간 별) — 골든타임 카운트다운 hover
rsx = [rt["start"][0] for rt in routes]; rsy = [rt["start"][1] for rt in routes]
rhov = [f"<b>최고위험 마을 #{i+1}</b><br>골든타임(불 도달) {rt['t_fire']}h<br>대피여유(마진) {rt['dest_margin']:+.1f}h<br>대피소까지 {rt['esc_km']:.1f}km"
        for i, rt in enumerate(routes)]
fig.add_trace(go.Scatter(x=rsx, y=rsy, mode="markers", marker=dict(symbol="star", size=15,
    color="#e11", line=dict(color="black", width=1)), name="최고위험 마을", text=rhov, hovertemplate="%{text}<extra></extra>"))
# 5) 위험권 건물(마진색) — 골든타임 hover
bhov = [f"골든타임 {('%.0f' % t) if np.isfinite(t) else '—'}h · 마진 {m:+.1f}h · 대피 {e:.1f}km"
        for t, m, e in zip(btf, bm, besc)]
fig.add_trace(go.Scattergl(x=bx, y=by, mode="markers", marker=dict(size=6, color=bm,
    colorscale="RdYlGn", cmin=-3, cmax=6, line=dict(width=0)), name="위험권 건물",
    text=bhov, hovertemplate="%{text}<extra></extra>", showlegend=True))
# 6) 대피소(녹색▲)·읍시(파란■) — 살아남은 목적지
ds = np.array([[dd["x"], dd["y"]] for dd in dests]); dk = np.array([dd["kind"] for dd in dests])
if len(ds):
    ms = dk == "shelter"; mt = dk == "town"
    if ms.any(): fig.add_trace(go.Scatter(x=ds[ms, 0], y=ds[ms, 1], mode="markers",
        marker=dict(symbol="triangle-up", size=9, color="#0b8a0b", line=dict(color="black", width=0.5)),
        name="대피소(마을회관·학교)", hovertemplate="대피소<extra></extra>"))
    if mt.any(): fig.add_trace(go.Scatter(x=ds[mt, 0], y=ds[mt, 1], mode="markers",
        marker=dict(symbol="square", size=10, color="#1565c0", line=dict(color="black", width=0.5)),
        name="읍·시", hovertemplate="읍·시<extra></extra>"))

lat0 = float(np.nanmean(lat_1d))
fig.update_layout(template="plotly_white", height=760,
    title=None, legend=dict(orientation="h", yanchor="bottom", y=1.005, x=0),
    margin=dict(l=40, r=10, t=10, b=40),
    xaxis=dict(title="경도", constrain="domain"),
    yaxis=dict(title="위도", scaleanchor="x", scaleratio=1.0/np.cos(np.radians(lat0))))

# ---------- 송출 카드 + 도주 순위표(HTML 주입) ----------
n_trap = int((bm < 0).sum()); n_tight = int(((bm >= 0) & (bm < 1)).sum()); n_ok = int((bm >= 1).sum())
med = float(np.nanmedian(bm))
rows_html = "".join(
    f"<tr><td>{i+1}</td><td>{rt['t_fire']}h</td><td class='{'r' if rt['dest_margin']<1 else 'g'}'>{rt['dest_margin']:+.1f}h</td>"
    f"<td>{rt['esc_km']:.1f} km</td><td>{rt['start'][1]:.4f}, {rt['start'][0]:.4f}</td></tr>"
    for i, rt in enumerate(routes))
head = f"""
<div style="font-family:'Noto Sans KR',sans-serif;max-width:1180px;margin:0 auto;padding:14px 18px">
<h2 style="margin:.2em 0">🔥 울진 2022 — 산불 대피 정보 송출 (예측기반)</h2>
<p style="color:#555;margin:.2em 0 1em">발화 즉시 물리 확산예측(실측 시변풍 V2)으로 산정한 <b>읍면·건물별 골든타임과 예측 도주로</b>.
지도 색=발화 후 화재 도달시각(h), 점=위험권 건물(녹색 여유·적색 갇힘), 검은선=대피소로 가는 예측 최단 도주로.</p>
<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px">
  <div style="flex:1;min-width:120px;background:#fdecea;border-radius:10px;padding:10px 14px"><div style="font-size:13px;color:#933">갇힘(마진&lt;0)</div><div style="font-size:26px;font-weight:700;color:#c0392b">{n_trap}<span style="font-size:14px;font-weight:400"> 동</span></div></div>
  <div style="flex:1;min-width:120px;background:#fff6e5;border-radius:10px;padding:10px 14px"><div style="font-size:13px;color:#96690f">초박빙(0~1h)</div><div style="font-size:26px;font-weight:700;color:#e08e0b">{n_tight}<span style="font-size:14px;font-weight:400"> 동</span></div></div>
  <div style="flex:1;min-width:120px;background:#eafaef;border-radius:10px;padding:10px 14px"><div style="font-size:13px;color:#166b2e">대피여유(≥1h)</div><div style="font-size:26px;font-weight:700;color:#1e9e4a">{n_ok}<span style="font-size:14px;font-weight:400"> 동</span></div></div>
  <div style="flex:1;min-width:120px;background:#eef3fb;border-radius:10px;padding:10px 14px"><div style="font-size:13px;color:#2a4d80">소각 · 안전목적지</div><div style="font-size:22px;font-weight:700;color:#1565c0">{d['burn_ha']:,}ha · {len(dests)}곳</div></div>
</div>
<h3 style="margin:.3em 0">📢 우선 송출 — 최고위험 마을 도주 순위 (예측 마진 최소순)</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">
<thead><tr style="background:#f0f0f0;text-align:left"><th style="padding:6px 10px">#</th><th>골든타임(불 도달)</th><th>대피여유(마진)</th><th>대피소까지</th><th>위치(위도,경도)</th></tr></thead>
<tbody>{rows_html}</tbody></table>
<p style="color:#888;font-size:12px;margin-top:6px">※ 예측=발화시점 물리모델 forecast. 한계: FIRMS 375m·오버패스, ERA5 25km 단일지점 바람, 상수 대피속도 {d['v_evac_kmh']:.0f}km/h,
정적 최근접 대피소(도주로가 이미 탄 방향일 수 있음). 절대수치 아닌 <b>상대 위험·우선순위</b> 지도.</p>
</div>
"""
tail = "<div style='font-family:sans-serif;max-width:1180px;margin:0 auto;padding:0 18px 24px;color:#aaa;font-size:11px'>Uljin 2022 wildfire evacuation dispatch — physical spread forecast (V1) + ERA5 wind (V2) + road routing. Self-contained.</div>"
body = fig.to_html(full_html=False, include_plotlyjs="inline", config={"displayModeBar": True, "scrollZoom": True})
html = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>울진 산불 대피 송출</title></head><body style='margin:0;background:#fafafa'>{head}<div style='max-width:1180px;margin:0 auto;padding:0 8px'>{body}</div>{tail}</body></html>"
open(OUT, "w").write(html)
print(f"saved {OUT}  ({len(html)/1e6:.1f} MB, 자체완결)")
print(f"송출 요약: 갇힘 {n_trap} · 초박빙 {n_tight} · 여유 {n_ok} · 마진중앙 {med:+.1f}h · 도주로 {len(routes)}개")
