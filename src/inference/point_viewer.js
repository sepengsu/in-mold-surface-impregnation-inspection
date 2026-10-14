/* Local image viewer. No network APIs, edits, or resampling of source data. */
(() => {
  'use strict';
  const records = JSON.parse(document.getElementById('viewer-data').textContent);
  const $ = id => document.getElementById(id);
  const viewport = $('viewport'), scene = $('scene'), svg = $('overlay');
  const colors = ['#ffce60', '#41d8e7', '#b58aff'];
  const ns = 'http://www.w3.org/2000/svg';
  let filtered = [], index = 0, current = null, scale = 1, tx = 0, ty = 0;
  let activePoint = null, dragging = null, loadToken = 0;
  const initialTask = Number(new URLSearchParams(window.location.search).get('task'));
  const limit = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const splitNames = {train: 'Train', val: 'Validation', test: 'Test'};
  const sideName = r => r.case.includes('-') ? '이동측' : '고정측';
  const caseNumber = r => Number(r.case.match(/case(\d+)/i)[1]);
  const captureLabel = r => r.capture_kind === 'shot' ? `샷 #${r.shot_number ?? r.specimen}` : `시편 ${r.specimen}`;
  const optionLabel = r => `${sideName(r)} · ${captureLabel(r)} · 촬영 ${r.direction}° · Task ${r.task_id}`;
  const versioned = path => path + (current?.asset_version ? '?v=' + current.asset_version : '');
  const sourceUrl = path => versioned('../' + path);

  function svgElement(name, attrs, text) {
    const node = document.createElementNS(ns, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function renderOverlay() {
    svg.replaceChildren();
    if (!current) return;
    if ($('roi-toggle').checked) {
      const [x, y, x1, y1] = current.roi;
      svg.append(svgElement('rect', {x,y,width:x1-x,height:y1-y,fill:'none',stroke:'#74d99b','stroke-width':1.5/scale,'stroke-dasharray':`${8/scale} ${5/scale}`}));
      if (current.roi_half_counts) {
        svg.append(svgElement('line',{x1:(x+x1)/2,y1:y,x2:(x+x1)/2,y2:y1,stroke:'#ffe17d','stroke-width':1.5/scale,'stroke-dasharray':`${6/scale} ${5/scale}`}));
      }
    }
    current.points.forEach((point, i) => {
      const [x,y,x1,y1] = point.xyxy, c = colors[i], selected = activePoint === point.point;
      const cx=(x+x1)/2,cy=(y+y1)/2, s=1/scale;
      svg.append(svgElement('rect',{x,y,width:x1-x,height:y1-y,fill:'none',stroke:c,'stroke-width':(selected?4:2.5)*s}));
      svg.append(svgElement('path',{d:`M ${cx-6*s},${cy} H ${cx+6*s} M ${cx},${cy-6*s} V ${cy+6*s}`,fill:'none',stroke:c,'stroke-width':1.8*s}));
      const ly=y-30*s, label=`${point.point}  ${point.gt_impregnation_pct.toFixed(2)}%`;
      svg.append(svgElement('rect',{x,y:ly,width:118*s,height:25*s,rx:4*s,fill:'#111d2a',opacity:.96}));
      svg.append(svgElement('text',{x:x+7*s,y:ly+17*s,fill:c,'font-size':13*s,'font-weight':700,'font-family':'Malgun Gothic, Segoe UI, sans-serif'},label));
    });
  }
  function paintTransform() {
    scene.style.transform=`translate(${tx}px,${ty}px) scale(${scale})`;
    $('zoom').textContent=`${Math.round(scale*100)}%`;
    $('raw').style.imageRendering=scale>=1?'pixelated':'auto';
    $('gt-layer').style.imageRendering=scale>=1?'pixelated':'auto';
    renderOverlay();
  }
  function clearPoint() {
    activePoint=null;
    for (const c of document.querySelectorAll('.point-card')) c.classList.remove('active');
  }
  function fit() {
    if (!current) return;
    clearPoint();
    const [w,h]=current.size;
    scale=Math.min((viewport.clientWidth-30)/w,(viewport.clientHeight-30)/h);
    tx=(viewport.clientWidth-w*scale)/2;ty=(viewport.clientHeight-h*scale)/2;
    paintTransform();
  }
  function zoomAt(next, x=viewport.clientWidth/2, y=viewport.clientHeight/2) {
    if (!current) return;
    next=limit(next,.05,8);
    const ratio=next/scale;
    tx=x-(x-tx)*ratio;ty=y-(y-ty)*ratio;scale=next;
    paintTransform();
  }
  function focusPoint(point) {
    activePoint=point.point;
    for (const c of document.querySelectorAll('.point-card')) c.classList.toggle('active',c.dataset.point===activePoint);
    const [x,y,x1,y1]=point.xyxy;
    scale=Math.min(viewport.clientWidth/(x1-x+210),viewport.clientHeight/(y1-y+210),2);
    tx=viewport.clientWidth/2-(x+x1)*scale/2;
    ty=viewport.clientHeight/2-(y+y1)*scale/2;
    paintTransform();
  }
  function createPointCards() {
    $('points').replaceChildren();
    current.points.forEach((p,i) => {
      const card=document.createElement('section');card.className='point-card';card.dataset.point=p.point;
      card.innerHTML=`<div class="point-heading"><strong class="point-name"><i class="point-dot"></i><span></span></strong><span class="gt-value"><small>GT</small><span></span>%</span></div><p class="coords"></p><div class="point-images"><figure><img alt=""><figcaption>원본 크롭</figcaption></figure><figure><img alt=""><figcaption>GT · Exposure</figcaption></figure></div><div class="point-actions"><button type="button">위치 확대</button><a target="_blank" rel="noopener">GT 마스크 열기</a></div>`;
      card.querySelector('.point-dot').style.background=colors[i];
      card.querySelector('.point-name span').textContent=p.point;
      card.querySelector('.gt-value span').textContent=p.gt_impregnation_pct.toFixed(2);
      card.querySelector('.coords').textContent=`좌상단 x ${p.xyxy[0]}, y ${p.xyxy[1]} · 256×256`;
      const images=card.querySelectorAll('img');
      images[0].src=sourceUrl(p.raw);images[0].alt=`${p.point} 원본 크롭`;
      images[1].src=sourceUrl(p.mask);images[1].alt=`${p.point} GT 마스크`;
      card.querySelector('button').onclick=()=>focusPoint(p);
      card.querySelector('a').href=sourceUrl(p.mask);
      $('points').append(card);
    });
  }
  function loadImage(src) {
    return new Promise((resolve,reject)=>{const img=new Image();img.onload=()=>resolve(img);img.onerror=()=>reject(new Error(src));img.src=src;});
  }
  async function show() {
    const token=++loadToken;
    current=filtered[index]||null;
    $('empty').hidden=!!current;$('loading').hidden=!current;scene.hidden=true;
    $('previous').disabled=!current||index===0;$('next').disabled=!current||index===filtered.length-1;
    $('counter').textContent=current?`${index+1} / ${filtered.length}`:'0 / 0';
    if (!current) {
      $('title').textContent='조건에 맞는 촬영본이 없습니다.';
      $('metadata').textContent='Case 또는 데이터 분할을 변경하세요.';
      $('points').replaceChildren();$('rotation-comparison').hidden=true;
      $('split-tag').textContent='';$('rotation-tag').textContent='';
      $('orientation-note').hidden=true;
      $('raw-link').removeAttribute('href');$('mask-link').removeAttribute('href');
      return;
    }
    const r=current;
    $('capture').value=String(r.task_id);
    $('title').textContent=`Case ${String(caseNumber(r)).padStart(2,'0')} · ${sideName(r)} · ${captureLabel(r)}`;
    $('metadata').textContent=`Task ${r.task_id} · 원촬영 방향 ${r.direction}° · ${r.size[0]} × ${r.size[1]} px · 각 측정 영역 256 × 256 px`;
    $('split-tag').textContent=splitNames[r.split];
    $('rotation-tag').textContent=r.rotation_deg?'추가 회전 180°':'원래 방향 유지';
    $('orientation-note').hidden=!(r.roi_half_counts||r.lateral_shape_differs_from_reference);
    if (r.roi_half_counts) {
      const c=r.roi_half_counts, n=v=>v.toLocaleString('ko-KR');
      $('orientation-note').textContent=`ROI 전체 GT 기준 · 회전 전 왼쪽 ${n(c.before_roi_left_exposure_pixels)} / 오른쪽 ${n(c.before_roi_right_exposure_pixels)} px → ${r.rotation_deg}°. 현재 왼쪽 ${n(c.after_roi_left_exposure_pixels)} / 오른쪽 ${n(c.after_roi_right_exposure_pixels)} px. P1·P3 값의 순서는 달라질 수 있습니다.`;
    } else {
      $('orientation-note').textContent='이동측은 기준 사진(고정측)과 좌우 형상 차이가 남습니다. 0°·180° 회전만 적용하며 좌우 반전은 하지 않습니다.';
    }
    $('rotation-comparison').src=versioned(r.position_figure || `rotation/task_${r.task_id}.jpg`);$('rotation-comparison').hidden=false;
    $('raw-link').href=sourceUrl(r.raw);$('mask-link').href=sourceUrl(r.mask);
    scene.style.width=r.size[0]+'px';scene.style.height=r.size[1]+'px';
    svg.setAttribute('viewBox',`0 0 ${r.size[0]} ${r.size[1]}`);
    createPointCards();fit();
    $('loading').textContent='이미지를 불러오는 중입니다.';
    try {
      const [raw,mask]=await Promise.all([loadImage(sourceUrl(r.raw)),loadImage(versioned(r.overlay))]);
      if(token!==loadToken)return;
      $('raw').src=raw.src;$('gt-layer').src=mask.src;
      scene.hidden=false;$('loading').hidden=true;
    } catch(error) {
      if(token!==loadToken)return;
      $('loading').textContent='이미지를 열 수 없습니다. 현재 데이터셋 폴더 전체를 함께 보관해 주세요.';
    }
  }
  function filter() {
    const oldId=current?.task_id || initialTask;
    filtered=records.filter(r=>($('case').value==='all'||caseNumber(r)===Number($('case').value))&&($('split').value==='all'||r.split===$('split').value));
    index=Math.max(0,filtered.findIndex(r=>r.task_id===oldId));
    $('capture').replaceChildren();
    for(const r of filtered){const option=document.createElement('option');option.value=r.task_id;option.textContent=`Case ${String(caseNumber(r)).padStart(2,'0')} · ${optionLabel(r)}`;$('capture').append(option);}
    $('capture').disabled=!filtered.length;
    void show();
  }
  for(const number of [...new Set(records.map(caseNumber))].sort((a,b)=>a-b)) {
    const option=document.createElement('option');option.value=number;option.textContent=`Case ${String(number).padStart(2,'0')}`;$('case').append(option);
  }
  $('case').onchange=$('split').onchange=filter;
  $('capture').onchange=()=>{index=filtered.findIndex(r=>String(r.task_id)===$('capture').value);void show();};
  $('previous').onclick=()=>{if(index>0){index--;void show();}};
  $('next').onclick=()=>{if(index<filtered.length-1){index++;void show();}};
  $('fit').onclick=fit;$('one-to-one').onclick=()=>zoomAt(1);
  $('zoom-in').onclick=()=>zoomAt(scale*1.25);$('zoom-out').onclick=()=>zoomAt(scale/1.25);
  $('roi-toggle').onchange=renderOverlay;
  $('gt-toggle').onchange=()=>{$('gt-layer').hidden=!$('gt-toggle').checked;};
  $('opacity').oninput=()=>{$('gt-layer').style.opacity=Number($('opacity').value)/100;};
  viewport.addEventListener('wheel',e=>{if(!current)return;e.preventDefault();const rect=viewport.getBoundingClientRect();zoomAt(scale*Math.exp(-limit(e.deltaY,-180,180)*.0025),e.clientX-rect.left,e.clientY-rect.top);},{passive:false});
  viewport.addEventListener('pointerdown',e=>{if(!current||e.button!==0)return;dragging={id:e.pointerId,x:e.clientX,y:e.clientY,tx,ty};viewport.setPointerCapture(e.pointerId);viewport.classList.add('dragging');});
  viewport.addEventListener('pointermove',e=>{if(!dragging||dragging.id!==e.pointerId)return;tx=dragging.tx+e.clientX-dragging.x;ty=dragging.ty+e.clientY-dragging.y;paintTransform();});
  const endDrag=()=>{dragging=null;viewport.classList.remove('dragging');};
  viewport.addEventListener('pointerup',endDrag);viewport.addEventListener('pointercancel',endDrag);
  viewport.addEventListener('dblclick',fit);
  document.addEventListener('keydown',e=>{
    if(['INPUT','SELECT','TEXTAREA','BUTTON'].includes(e.target.tagName))return;
    if(e.key==='ArrowLeft'&&index>0){e.preventDefault();index--;void show();}
    if(e.key==='ArrowRight'&&index<filtered.length-1){e.preventDefault();index++;void show();}
    if(e.key==='Escape')fit();
  });
  new ResizeObserver(()=>{if(current)fit();}).observe(viewport);
  filter();
})();
