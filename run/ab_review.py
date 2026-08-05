#!/usr/bin/env python3
"""[평가] 블라인드 페어와이즈 A/B 리뷰 페이지를 만든다.

■ 왜 필요한가 (2026-08-05)
  이 프로젝트에서 지표와 육안이 여섯 번 갈렸고 여섯 번 다 육안이 맞았다.
  그런데 그것은 "지표가 틀렸다"는 증거이지 "육안이 항상 맞다"는 증거가 아니다.
  **어느 쪽이 새 모델인지 알고 보면 판단이 기운다.**

  그래서 세 가지를 강제한다.
    ① 좌우 무작위 + 라벨 은닉 → 편향 제거
    ② 고정 홀드아웃 → 체리피킹 제거
    ③ 질문 분리 → "예쁜가"와 "원본을 살렸나"가 섞이지 않게

  ③ 이 특히 중요하다. 증강 모델을 기각할 때 이유가 "볼에 주근깨가 생겼다"였는데,
  그 점은 원본에 실제로 있는 것을 충실히 그린 결과였다. 두 질문이 섞여 있어서
  "지저분하다(Q1 패배)"와 "충실하다(Q2 승리)"가 하나로 뭉쳤다.

  Emu 와 Diffusion-DPO 가 쓰는 프로토콜이 이것이다 —
  visual appeal 을 물을 때는 원본을 보여주지 않는다.

■ 사용
  python3 run/ab_review.py --a out/eval_A --b out/eval_B \
      --label-a edge3_eq --label-b tgt3k \
      --src out/occ65_crops3k/input --n 40 --out out/ab_review.html
  → 맥으로 받아서 더블클릭. 키보드로 투표하고 마지막에 승률·p값·정답 공개.
"""
import argparse, base64, glob, hashlib, json, os, random
from pathlib import Path

import cv2


def load_b64(path, size, quality):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if max(h, w) > size:
        scale = size / max(h, w)
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def stems_of(directory):
    out = {}
    for path in glob.glob(os.path.join(directory, "*")):
        if Path(path).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            out[Path(path).stem] = path
    return out


HTML = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>A/B 블라인드 리뷰</title>
<style>
:root{--bg:#111418;--fg:#e8eaed;--dim:#9aa0a6;--acc:#8ab4f8}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}
header{padding:14px 20px;border-bottom:1px solid #2a2f36;display:flex;gap:20px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;font-weight:600}
.q{color:var(--acc);font-weight:600}
.bar{flex:1;height:6px;background:#2a2f36;border-radius:3px;overflow:hidden;min-width:120px}
.bar > i{display:block;height:100%;background:var(--acc);width:0}
main{padding:18px 20px 90px;max-width:1500px;margin:0 auto}
.src{text-align:center;margin-bottom:14px}
.src img{max-height:190px;border-radius:8px;border:1px solid #2a2f36}
.src p{color:var(--dim);font-size:13px;margin:6px 0 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:#171b20;border:2px solid #2a2f36;border-radius:10px;padding:10px;text-align:center;cursor:pointer;transition:border-color .12s}
.card:hover{border-color:var(--acc)}
.card img{width:100%;border-radius:6px;display:block}
.card b{display:block;margin-top:8px;color:var(--dim);font-size:13px;font-weight:500}
footer{position:fixed;left:0;right:0;bottom:0;background:#0d1013;border-top:1px solid #2a2f36;padding:12px 20px;display:flex;gap:16px;justify-content:center;flex-wrap:wrap;color:var(--dim);font-size:13px}
kbd{background:#2a2f36;color:var(--fg);border-radius:4px;padding:2px 7px;font-size:12px;font-family:inherit}
#done{display:none;padding:10px 0}
table{border-collapse:collapse;margin:14px 0;font-size:14px}
th,td{border:1px solid #2a2f36;padding:8px 14px;text-align:left}
th{background:#171b20;color:var(--dim);font-weight:500}
.win{color:#81c995;font-weight:600}
.lose{color:#f28b82}
button{background:var(--acc);color:#0d1013;border:0;border-radius:6px;padding:9px 18px;font:inherit;font-weight:600;cursor:pointer}
.note{color:var(--dim);font-size:13px;max-width:760px}
</style></head><body>
<header>
  <h1>A/B 블라인드 리뷰</h1>
  <span class="q" id="qtitle"></span>
  <div class="bar"><i id="prog"></i></div>
  <span id="count" style="color:var(--dim);font-size:13px"></span>
</header>
<main>
  <div id="review">
    <div class="src" id="srcbox"></div>
    <div class="pair">
      <div class="card" id="cardL" onclick="vote('L')"><img id="imgL"><b>← 왼쪽</b></div>
      <div class="card" id="cardR" onclick="vote('R')"><img id="imgR"><b>오른쪽 →</b></div>
    </div>
  </div>
  <div id="done"></div>
</main>
<footer>
  <span><kbd>←</kbd> 왼쪽</span><span><kbd>→</kbd> 오른쪽</span>
  <span><kbd>Space</kbd> 무승부</span><span><kbd>Z</kbd> 되돌리기</span>
</footer>
<script>
const DATA = __DATA__;
const ROUNDS = [
  {key:"appeal", title:"Q1. 어느 쪽이 그림으로서 더 나은가? (원본 숨김)", showSrc:false},
  {key:"faithful", title:"Q2. 어느 쪽이 원본의 표정·인상을 더 잘 살렸나?", showSrc:true}
];
let r = 0, i = 0;
const votes = {appeal:[], faithful:[]};

function render(){
  const round = ROUNDS[r], item = DATA.items[i];
  document.getElementById('qtitle').textContent = round.title;
  document.getElementById('count').textContent = (i+1)+" / "+DATA.items.length;
  document.getElementById('prog').style.width =
    (100*(r*DATA.items.length+i)/(ROUNDS.length*DATA.items.length))+"%";
  const box = document.getElementById('srcbox');
  box.innerHTML = round.showSrc && item.src
    ? '<img src="'+item.src+'"><p>원본</p>' : '';
  document.getElementById('imgL').src = item.left;
  document.getElementById('imgR').src = item.right;
  window.scrollTo(0,0);
}
function vote(side){
  votes[ROUNDS[r].key].push({stem:DATA.items[i].stem, side:side, leftIs:DATA.items[i].leftIs});
  i++;
  if(i >= DATA.items.length){ r++; i = 0; }
  if(r >= ROUNDS.length){ finish(); return; }
  render();
}
function undo(){
  if(i === 0 && r === 0) return;
  if(i === 0){ r--; i = DATA.items.length - 1; } else { i--; }
  votes[ROUNDS[r].key].pop();
  render();
}
function logC(n,k){let s=0;for(let j=1;j<=k;j++)s+=Math.log(n-k+j)-Math.log(j);return s}
function binomP(a,b){ // 양측 정확 이항검정 (p=0.5)
  const n=a+b; if(n===0) return 1;
  const obs=Math.max(a,b); let p=0;
  for(let k=0;k<=n;k++){ if(k>=obs||k<=n-obs) p+=Math.exp(logC(n,k)-n*Math.log(2)); }
  return Math.min(1,p);
}
function tally(list){
  let A=0,B=0,T=0;
  for(const v of list){
    if(v.side==='T'){T++;continue}
    const winner = (v.side==='L') ? v.leftIs : (v.leftIs==='A'?'B':'A');
    if(winner==='A')A++;else B++;
  }
  return {A,B,T};
}
function finish(){
  document.getElementById('review').style.display='none';
  document.getElementById('qtitle').textContent='완료';
  document.getElementById('prog').style.width='100%';
  let html='<h2 style="font-size:17px">결과</h2>';
  for(const round of ROUNDS){
    const t=tally(votes[round.key]), n=t.A+t.B;
    const rate=n?(100*t.A/n):0, p=binomP(t.A,t.B);
    const cls=t.A>t.B?'win':(t.A<t.B?'lose':'');
    html+='<h3 style="font-size:15px;margin-top:18px">'+round.title+'</h3>'
      +'<table><tr><th>'+DATA.labelA+'</th><th>'+DATA.labelB+'</th><th>무승부</th>'
      +'<th>'+DATA.labelA+' 승률</th><th>p (양측 이항)</th></tr>'
      +'<tr><td class="'+cls+'">'+t.A+'</td><td>'+t.B+'</td><td>'+t.T+'</td>'
      +'<td class="'+cls+'">'+rate.toFixed(1)+'%</td><td>'+p.toFixed(4)+'</td></tr></table>';
  }
  html+='<p class="note">읽는 법 — 승률 45~55%는 차이 없음으로 본다. '
     +'p &lt; 0.05 이고 승률이 60%를 넘으면 실질적 개선으로 판정한다. '
     +'표본이 40쌍이면 60%도 p가 0.27 수준이라 유의하지 않을 수 있다. '
     +'애매하면 쌍 수를 늘린다.</p>'
     +'<p class="note"><b>왼쪽/오른쪽 배치는 항목마다 무작위였고 라벨은 지금 처음 공개된다.</b></p>'
     +'<p><button onclick="dl()">결과 JSON 저장</button></p>';
  document.getElementById('done').style.display='block';
  document.getElementById('done').innerHTML=html;
}
function dl(){
  const out={labelA:DATA.labelA,labelB:DATA.labelB,generated:DATA.generated,votes:votes,
             summary:{appeal:tally(votes.appeal),faithful:tally(votes.faithful)}};
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));
  a.download='ab_review_result.json'; a.click();
}
document.addEventListener('keydown',e=>{
  if(document.getElementById('done').style.display==='block') return;
  if(e.key==='ArrowLeft'){e.preventDefault();vote('L')}
  else if(e.key==='ArrowRight'){e.preventDefault();vote('R')}
  else if(e.key===' '){e.preventDefault();vote('T')}
  else if(e.key.toLowerCase()==='z'){e.preventDefault();undo()}
});
render();
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 모델 출력 폴더")
    ap.add_argument("--b", required=True, help="B 모델 출력 폴더")
    ap.add_argument("--src", default=None, help="원본(입력) 폴더. Q2 에서만 보여준다")
    ap.add_argument("--label-a", default="A", dest="label_a")
    ap.add_argument("--label-b", default="B", dest="label_b")
    ap.add_argument("--include-file", default=None, dest="include_file",
                    help="홀드아웃 stem 목록. 지정하면 이 순서로만 쓴다")
    ap.add_argument("--n", type=int, default=40, help="쌍 수. 40쌍이면 한 라운드 5분 남짓")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=512, help="긴 변 최대 픽셀")
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--out", default="out/ab_review.html")
    args = ap.parse_args()

    A, B = stems_of(args.a), stems_of(args.b)
    S = stems_of(args.src) if args.src else {}
    common = sorted(set(A) & set(B))
    if not common:
        raise SystemExit(f"공통 stem 없음: {args.a} / {args.b}")

    if args.include_file:
        wanted = [s.strip() for s in Path(args.include_file).read_text(encoding="utf-8").splitlines() if s.strip()]
        common = [s for s in wanted if s in A and s in B]
        if not common:
            raise SystemExit("include-file 의 stem 이 두 폴더에 없다")

    rng = random.Random(args.seed)
    rng.shuffle(common)
    common = common[:args.n]

    items, skipped = [], 0
    for stem in common:
        left_is = "A" if rng.random() < 0.5 else "B"          # ★ 좌우 무작위
        a_img, b_img = load_b64(A[stem], args.size, args.quality), load_b64(B[stem], args.size, args.quality)
        if a_img is None or b_img is None:
            skipped += 1
            continue
        items.append({
            "stem": stem, "leftIs": left_is,
            "left": a_img if left_is == "A" else b_img,
            "right": b_img if left_is == "A" else a_img,
            "src": load_b64(S[stem], args.size, args.quality) if stem in S else None,
        })

    data = {"labelA": args.label_a, "labelB": args.label_b,
            "generated": hashlib.sha1(f"{args.a}|{args.b}|{args.seed}".encode()).hexdigest()[:12],
            "items": items}
    html = HTML.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(html, encoding="utf-8")

    mb = os.path.getsize(args.out) / 1e6
    print(f"[생성] {len(items)}쌍 (건너뜀 {skipped}) → {args.out}  {mb:.1f} MB")
    print(f"  A={args.label_a}  B={args.label_b}   ※ 라벨은 HTML 안에 있으나 끝나야 표시된다")
    print(f"  좌우 배치 무작위(seed {args.seed}), Q1 은 원본 숨김")
    print("\n맥으로 받아서 더블클릭:")
    print(f"  scp -i <키> ubuntu@<호스트>:{os.path.abspath(args.out)} ~/Desktop/")


if __name__ == "__main__":
    main()
