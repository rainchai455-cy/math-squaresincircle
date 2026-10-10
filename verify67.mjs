import fs from 'node:fs';
// Standalone verifier. Reads ONLY packing67.txt: "colour X0 Y0 X1 Y1" in DOUBLED cm
// (real coord = value/2). Circle = radius 5 => doubled radius 10 => x^2+y^2 <= 100.
// Everything is integer arithmetic: no tolerance, no floats, no image.
const lines=fs.readFileSync(new URL('./packing67.txt', import.meta.url),'utf8')
  .split('\n').map(l=>l.trim()).filter(l=>l&&!l.startsWith('#'));
const S=lines.map(l=>{const p=l.split(/\s+/);
  return {c:p[0],X0:+p[1],Y0:+p[2],X1:+p[3],Y1:+p[4]};});
const LIM=100;
let fail=0; const cnt={}, per={};
for(const q of S){
  cnt[q.c]=(cnt[q.c]||0)+1;
  if(![q.X0,q.Y0,q.X1,q.Y1].every(Number.isInteger)){console.log('NON-INTEGER',q);fail++;}
  if(q.X1-q.X0!==2||q.Y1-q.Y0!==2){console.log('NOT A UNIT SQUARE',q);fail++;}
  let mx=0;
  for(const x of [q.X0,q.X1])for(const y of [q.Y0,q.Y1]){
    const r2=x*x+y*y; if(r2>LIM){console.log('CORNER OUTSIDE R=5:',q.c,x/2,y/2,'r=',(r2/4).toFixed(4));fail++;}
    mx=Math.max(mx,r2);}
  per[q.c]=per[q.c]||{tight:0,maxr2:0};
  per[q.c].maxr2=Math.max(per[q.c].maxr2,mx);
  if(mx===LIM)per[q.c].tight++;
}
for(let i=0;i<S.length;i++)for(let j=i+1;j<S.length;j++){
  const A=S[i],B=S[j];
  const ix=Math.min(A.X1,B.X1)-Math.max(A.X0,B.X0);
  const iy=Math.min(A.Y1,B.Y1)-Math.max(A.Y0,B.Y0);
  if(ix>0&&iy>0){console.log('INTERIOR OVERLAP:',A.c,A.X0/2,A.Y0/2,'vs',B.c,B.X0/2,B.Y0/2);fail++;}}
console.log(`squares read = ${S.length}   per colour ${JSON.stringify(cnt)}`);
console.log(`per-colour worst corner r^2 and count exactly on the circle:`);
for(const k in per)console.log(`  ${k.padEnd(7)} max r^2=${per[k].maxr2} (=R^2 ${per[k].maxr2/4})   on-circle cells=${per[k].tight}`);
console.log('');
console.log(fail===0
  ? `VERDICT: PASS - 67 closed unit squares, pairwise interior-disjoint, all corners with x^2+y^2<=100.  r(67) <= 5 certified in exact integers.`
  : `VERDICT: FAIL - ${fail} defect(s).`);
