"""산불 진화·복원 통합 상황판 (자체완결 인터랙티브 HTML) — 화재무관.

한반도 어느 화재든: 예측 도달시각 히트맵 위에 진화(🚒배치·💧헬기투하·◆취수원)+복원(기간·비용) 레이어.
건물/도주로/대피소는 dispatch payload 있으면 표시, 없으면 생략(확산·진화·복원은 항상 표시).

입력: runs/{key}_tfire.npz (또는 레거시), runs/{key}_suppression_payload.json,
      (선택) dispatch payload, 웹앱 fire_dispatch 번들(소각경계).
출력: runs/{key}_ops_dashboard.html

사용: FIRE=sancheong_2025 python src/ops_dashboard.py   (기본 uljin_2022)
"""
import os, sys, json, numpy as np
from pyproj import Transformer
import plotly.graph_objects as go

HERE = __file__.rsplit("/", 2)[0]; R = HERE + "/runs"
sys.path.insert(0, HERE + "/src")
import suppression_restoration as SR       # FIRES 레지스트리 재사용

key = os.environ.get("FIRE", "uljin_2022")
cfg = SR.FIRES[key]
SUPP = R + f"/{key}_suppression_payload.json"
TF = cfg.get("tfire") if os.path.exists(cfg.get("tfire", "")) else cfg.get("tfire_legacy")
DISP = cfg.get("payload")
BUNDLE = HERE + f"/../../웹앱/data/fire_dispatch/{key}.json"
OUT = R + f"/{key}_ops_dashboard.html"

s = json.load(open(SUPP))
disp = json.load(open(DISP)) if (DISP and os.path.exists(DISP)) else None
bundle = json.load(open(BUNDLE)) if os.path.exists(BUNDLE) else None

# --- 배경 도달시각 히트맵 (tfire npz → lon/lat 축) ---
z = np.load(TF, allow_pickle=True)
T = z["T"].astype(float); minx = float(z["minx"]); maxy = float(z["maxy"]); cell = float(z["cell"])
H, Wd = int(z["H"]), int(z["W"]); span = float(z["span_h"])
cx = minx + (np.arange(Wd) + 0.5) * cell; cy = maxy - (np.arange(H) + 0.5) * cell
tr = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
lon_1d, _ = tr.transform(cx, np.full_like(cx, cy[H // 2]))
_, lat_1d = tr.transform(np.full_like(cy, cx[Wd // 2]), cy)
step = 2
Tvis = np.where(np.isfinite(T) & (T <= span), T, np.nan)[::step, ::step]
lon_s, lat_s = lon_1d[::step], lat_1d[::step]

fig = go.Figure()
fig.add_trace(go.Heatmap(z=Tvis, x=lon_s, y=lat_s, colorscale="Inferno", opacity=0.72,
    zmin=0, zmax=span, colorbar=dict(title="예측 화재<br>도달 (h)", x=1.005, len=0.78),
    hovertemplate="예측 도달 %{z:.0f}h<extra></extra>", name="도달시각"))

# 소각 경계: dispatch payload(rings) 또는 webapp 번들
rings = (disp or {}).get("burn_rings") or (bundle or {}).get("burn_rings") or []
for r in rings:
    if len(r) < 3: continue
    xs = [p[0] for p in r] + [r[0][0]]; ys = [p[1] for p in r] + [r[0][1]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="white", width=1.1),
        opacity=0.55, hoverinfo="skip", showlegend=False))
if rings:
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color="white", width=1.5), name="소각 경계"))

# (선택) 대피 레이어 — dispatch payload 있을 때만
if disp:
    routes = disp.get("routes", []); dests = disp.get("dests", []); blds = disp.get("buildings", [])
    for i, rt in enumerate(routes):
        px = [p[0] for p in rt["path"]]; py = [p[1] for p in rt["path"]]
        fig.add_trace(go.Scatter(x=px, y=py, mode="lines", line=dict(color="#111", width=1.5),
            opacity=0.55, hoverinfo="skip", showlegend=(i == 0), name="예측 도주로"))
    if blds:
        bx = [b["x"] for b in blds]; by = [b["y"] for b in blds]; bm = [b["margin"] for b in blds]
        fig.add_trace(go.Scattergl(x=bx, y=by, mode="markers", marker=dict(size=4, color=bm,
            colorscale="RdYlGn", cmin=-3, cmax=6, opacity=0.5, line=dict(width=0)),
            name="위험권 건물", hovertemplate="대피마진 %{marker.color:+.1f}h<extra></extra>"))
    if dests:
        ds = np.array([[d["x"], d["y"]] for d in dests]); dk = np.array([d["kind"] for d in dests])
        ms = dk == "shelter"; mt = dk == "town"
        if ms.any(): fig.add_trace(go.Scatter(x=ds[ms,0], y=ds[ms,1], mode="markers",
            marker=dict(symbol="triangle-up", size=7, color="#0b8a0b", line=dict(color="black", width=0.4)),
            name="대피소", hovertemplate="대피소<extra></extra>"))
        if mt.any(): fig.add_trace(go.Scatter(x=ds[mt,0], y=ds[mt,1], mode="markers",
            marker=dict(symbol="square", size=8, color="#1565c0", line=dict(color="black", width=0.4)),
            name="읍·시", hovertemplate="읍·시<extra></extra>"))

# ② 취수원 + 헬기 투하 + 연결선
water = s["water_sources"]; drops = s["heli_drops"]
if water:
    fig.add_trace(go.Scatter(x=[w["lon"] for w in water], y=[w["lat"] for w in water], mode="markers",
        marker=dict(symbol="diamond", size=7, color="#00b5d8", line=dict(color="#036", width=0.5)),
        name="취수원(저수지·하천)", hovertemplate="취수원<extra></extra>"))
for h in drops:
    fig.add_trace(go.Scatter(x=[h["water"]["lon"], h["lon"]], y=[h["water"]["lat"], h["lat"]], mode="lines",
        line=dict(color="#00b5d8", width=1, dash="dot"), opacity=0.6, hoverinfo="skip", showlegend=False))
if drops:
    dtxt = [f"<b>헬기 투하 #{h['rank']}</b> ({h['front_type']})<br>우선도 {h['priority']}<br>{h['rationale']}" for h in drops]
    fig.add_trace(go.Scatter(x=[h["lon"] for h in drops], y=[h["lat"] for h in drops], mode="markers+text",
        marker=dict(symbol="circle", size=[16+14*h["priority"] for h in drops], color="#2563eb", opacity=0.85,
                    line=dict(color="white", width=1.2)),
        text=[str(h["rank"]) for h in drops], textfont=dict(color="white", size=9),
        name="💧 헬기 물투하점", customdata=dtxt, hovertemplate="%{customdata}<extra></extra>"))

# ① 소방 전진배치
st = s["staging"]["sites"]
if st:
    stxt = [f"<b>전진배치 #{x['rank']}</b><br>방어 {x['covers_new']}(누적 {x['covers']})<br>권장 소방대 {x['crews']}대" for x in st]
    fig.add_trace(go.Scatter(x=[x["lon"] for x in st], y=[x["lat"] for x in st], mode="markers",
        marker=dict(symbol="square", size=[14+2.2*np.sqrt(x["crews"]) for x in st], color="#e11",
                    opacity=0.9, line=dict(color="white", width=1.3)),
        name="🚒 소방 전진배치", customdata=stxt, hovertemplate="%{customdata}<extra></extra>"))

lat0 = float(np.nanmean(lat_1d))
fig.update_layout(template="plotly_white", height=780,
    legend=dict(orientation="h", yanchor="bottom", y=1.005, x=0, font=dict(size=11)),
    margin=dict(l=40, r=10, t=10, b=40), xaxis=dict(title="경도", constrain="domain"),
    yaxis=dict(title="위도", scaleanchor="x", scaleratio=1.0/np.cos(np.radians(lat0))))

# 배너 + 복원표
w = s["meta"]["wind"]; rp = s["restoration"]; cov = s["staging"]; sev = rp["by_severity"]
dk_label = {"위험건물":"위험건물 방어", "화선footprint":"화선 확산차단(건물데이터 없음)"}.get(s["meta"].get("demand_kind",""), "")
banner = f"""
<div style="font-family:'Noto Sans KR',sans-serif;max-width:1200px;margin:0 auto;padding:14px 18px">
<h2 style="margin:.2em 0">🔥 {s['meta']['label']} — 산불 진화·복원 통합 상황판 (지상국)</h2>
<p style="color:#555;margin:.2em 0 .8em">예측 확산(실측 바람) 위에 <b>소방력 전진배치(🚒)·헬기 물투하 좌표(💧)·산림 재조성 기간/비용</b>. 결정시점 t*={s['meta']['t_star_h']}h · 배치기준={dk_label}.</p>
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:6px">
  <div style="flex:1;min-width:150px;background:#eef6ff;border-radius:10px;padding:9px 13px">
    <div style="font-size:12px;color:#2a4d80">현재 바람 ({w['source']})</div>
    <div style="font-size:19px;font-weight:700;color:#1565c0">향함 {w['toward_deg']:.0f}° · {w['speed_ms']:.1f}m/s</div>
    <div style="font-size:12px;color:#555">습도 {w['rh_pct']:.0f}% · {w['obs_time']}</div></div>
  <div style="flex:1;min-width:150px;background:#fdecea;border-radius:10px;padding:9px 13px">
    <div style="font-size:12px;color:#933">🚒 소방 전진배치</div>
    <div style="font-size:22px;font-weight:700;color:#c0392b">{len(st)}곳 · 커버 {cov['coverage_pct']:.0f}%</div>
    <div style="font-size:12px;color:#555">수요 {cov['covered']}/{cov['demand']}</div></div>
  <div style="flex:1;min-width:150px;background:#eef9ff;border-radius:10px;padding:9px 13px">
    <div style="font-size:12px;color:#036">💧 헬기 물투하</div>
    <div style="font-size:22px;font-weight:700;color:#0b74a8">{len(drops)} 지점</div>
    <div style="font-size:12px;color:#555">취수원 {len(water)}곳 · 화두 우선</div></div>
  <div style="flex:1;min-width:150px;background:#eafaef;border-radius:10px;padding:9px 13px">
    <div style="font-size:12px;color:#166b2e">🌲 실질 녹화(수관울폐)</div>
    <div style="font-size:22px;font-weight:700;color:#1e9e4a">약 {rp['canopy_years_weighted']:.0f}년</div>
    <div style="font-size:12px;color:#555">생태성숙 {rp['ecosys_years_weighted']:.0f}년 · 침엽 {rp['conifer_frac']*100:.0f}%</div></div>
</div></div>
"""
sevrow = "".join(
    f"<tr><td>{nm}</td><td>{sev[k]['area_ha']:,.0f} ha</td><td>{sev[k]['frac']*100:.0f}%</td>"
    f"<td>{sev[k]['plant_years']:.0f}년</td><td>{sev[k]['canopy_years']:.0f}년</td><td>{sev[k]['ecosys_years']:.0f}년</td>"
    f"<td>{sev[k]['cost_national_krw']/1e8:,.0f}억원</td></tr>" for k, nm in [("severe","심각"),("moderate","중간"),("light","경미")])
dmg = rp.get("damage_loss")
dmg_line = (f"<p style='color:#888;font-size:12px;margin:.3em 0'>※ 참고 — 산불 <b>피해 손실액</b>(복구투입비와 다름): "
            f"국가 {dmg['national_total_krw']/1e12:.2f}조원 (민간 {dmg['private_total_krw']/1e8:,.0f}억원)</p>") if dmg else ""
restore = f"""
<div style="font-family:'Noto Sans KR',sans-serif;max-width:1200px;margin:0 auto;padding:6px 18px 20px">
<h3 style="margin:.4em 0">🌲 산림 재조성 — 소요기간 · 복원비용 (실소각 {rp['area_real_ha']:,} ha)</h3>
<table style="border-collapse:collapse;width:100%;font-size:14px">
<thead><tr style="background:#eef3ee;text-align:left"><th style="padding:6px 10px">심각도</th><th>면적</th><th>비율</th><th>식재완료</th><th>수관울폐(녹화)</th><th>생태성숙</th><th>국가 복구비</th></tr></thead>
<tbody>{sevrow}
<tr style="font-weight:700;background:#f7f7f7"><td>합계</td><td>{rp['area_real_ha']:,} ha</td><td>100%</td>
<td>~{rp['plant_complete_years']:.0f}년</td><td>~{rp['canopy_years_weighted']:.0f}년</td><td>~{rp['ecosys_years_weighted']:.0f}년</td>
<td>{rp['cost_national_text']} <span style="font-weight:400;color:#777">(민간 자부담 {rp['cost_private_text']})</span></td></tr>
</tbody></table>
{dmg_line}
<p style="color:#888;font-size:12px;margin-top:6px">※ 진화=확산모델 기반 <b>상대 우선순위</b> forecast(절대보장 아님)·소방차 이동=직선×우회 근사. 심각도 면적=비율×총면적(등급별 절대면적 화재간 전이불가 판정 존중). 단가·소요년=한국 소나무 산불 재조림 문헌 근사.</p>
</div>
"""
tail = "<div style='font-family:sans-serif;max-width:1200px;margin:0 auto;padding:0 18px 22px;color:#aaa;font-size:11px'>Integrated wildfire ops board — spread forecast + suppression(MCLP staging, wind-aware heli drops, KMA wind) + restoration. Fire-agnostic, self-contained.</div>"
body = fig.to_html(full_html=False, include_plotlyjs="inline", config={"displayModeBar": True, "scrollZoom": True})
html = (f"<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{s['meta']['label']} 진화·복원 상황판</title></head>"
        f"<body style='margin:0;background:#fafafa'>{banner}"
        f"<div style='max-width:1200px;margin:0 auto;padding:0 8px'>{body}</div>{restore}{tail}</body></html>")
open(OUT, "w").write(html)
print(f"saved {OUT}  ({len(html)/1e6:.1f} MB) · 배치 {len(st)} · 투하 {len(drops)} · 취수 {len(water)} · 복원 {rp['area_real_ha']:,}ha/{rp['cost_national_text']}")
