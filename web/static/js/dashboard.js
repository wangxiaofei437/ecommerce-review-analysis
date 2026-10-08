/* ============================================================
 * dashboard.js
 * 拉取后端 API，使用 ECharts / ECharts-GL 渲染对比图表 + 实时预测
 * 支持上传 / 导出 / 重置，所有图表随数据源切换实时联动
 * ============================================================ */

const COLORS = {
  rf:       '#3b82f6',
  lr:       '#8b5cf6',
  lgb:      '#10b981',
  ensemble: '#f59e0b',
};

const CHART_BG = 'transparent';
const TEXT_DIM = '#7b8fb8';
const TEXT     = '#d6e6ff';
const MODEL_KEYS = ['rf', 'lr', 'lgb', 'ensemble'];
const MODEL_NAMES = { rf: '随机森林', lr: '逻辑回归', lgb: 'LightGBM', ensemble: '融合模型' };

const charts = {};
function getChart(id) {
  if (!charts[id]) charts[id] = echarts.init(document.getElementById(id), null, { renderer: 'canvas' });
  return charts[id];
}
function fmtPercent(v, d = 2) { return (v * 100).toFixed(d) + '%'; }

/* ---------- 顶部时钟 ---------- */
function tickClock() {
  const el = document.getElementById('datetime');
  if (!el) return;
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  el.textContent = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
setInterval(tickClock, 1000);
tickClock();

/* ---------- 工具栏：数据源信息 ---------- */
async function loadSource() {
  const r = await fetch('/api/source').then(r => r.json());
  document.getElementById('tb-source').textContent = r.name || '--';
  const tag = document.getElementById('tb-label-tag');
  tag.textContent = r.has_label ? '已带标签' : '无标签 · 仅预测';
  tag.className = 'tb-tag ' + (r.has_label ? 'ok' : 'warn');
  document.getElementById('tb-time').textContent = r.updated_at ? `更新于 ${r.updated_at}` : '';
}

/* ---------- KPI ---------- */
async function loadOverview() {
  const r = await fetch('/api/overview').then(r => r.json());
  document.getElementById('kpi-total').textContent   = r.total.toLocaleString();
  document.getElementById('kpi-useful').textContent  = r.useful.toLocaleString();
  document.getElementById('kpi-useless').textContent = r.useless.toLocaleString();
  document.getElementById('kpi-ratio').textContent   = fmtPercent(r.useful_ratio, 2);
  document.getElementById('kpi-feat').textContent    = r.n_features;
}

/* ---------- 雷达 ---------- */
async function loadMetrics() {
  const r = await fetch('/api/metrics').then(r => r.json());
  const radar = getChart('chart-radar');
  if (!r.has_label) {
    radar.setOption({
      backgroundColor: CHART_BG,
      title: { text: '无标签数据，仅展示预测结果', left: 'center', top: 'center',
               textStyle: { color: TEXT_DIM, fontSize: 13 } },
      series: [],
    }, true);
    return;
  }
  const indicators = [
    { name: '准确率',  max: 1 },
    { name: '精确率',  max: 1 },
    { name: '召回率',  max: 1 },
    { name: 'F1',      max: 1 },
    { name: 'AUC',     max: 1 },
  ];
  radar.setOption({
    backgroundColor: CHART_BG,
    tooltip: { trigger: 'item' },
    legend: { bottom: 0, textStyle: { color: TEXT_DIM, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    radar: {
      indicator: indicators,
      center: ['50%', '50%'], radius: '62%',
      axisName: { color: TEXT, fontSize: 11 },
      splitArea: { areaStyle: { color: ['rgba(94,173,245,0.04)', 'rgba(94,173,245,0.08)'] } },
      splitLine: { lineStyle: { color: 'rgba(94,173,245,0.25)' } },
      axisLine:  { lineStyle: { color: 'rgba(94,173,245,0.35)' } },
    },
    series: [{
      type: 'radar',
      symbol: 'circle', symbolSize: 5,
      areaStyle: { opacity: 0.2 },
      lineStyle: { width: 2 },
      data: r.models.map(m => ({
        name: m.name,
        value: [m.accuracy, m.precision, m.recall, m.f1, m.auc],
        itemStyle: { color: COLORS[m.key] },
      })),
    }],
  }, true);
}

/* ---------- 多指标分组柱状对比（替换 3D 柱图） ---------- */
async function loadMultiMetric() {
  const r = await fetch('/api/accuracy_bar').then(r => r.json());
  const chart = getChart('chart-multi-metric');
  if (!r.has_label) {
    chart.setOption({
      backgroundColor: CHART_BG,
      title: { text: '上传带 useful_votes 列的数据可显示对比', left: 'center', top: 'center',
               textStyle: { color: TEXT_DIM, fontSize: 13 } },
    }, true);
    return;
  }
  const metricKeys   = ['accuracy', 'precision', 'recall', 'f1', 'auc'];
  const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'];

  const series = r.models.map(m => ({
    name: m.name,
    type: 'bar',
    barGap: '8%',
    barCategoryGap: '38%',
    data: metricKeys.map(k => +(m[k] * 100).toFixed(2)),
    itemStyle: {
      borderRadius: [3, 3, 0, 0],
      color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
        { offset: 0, color: COLORS[m.key] },
        { offset: 1, color: COLORS[m.key] + '55' },
      ]),
    },
    label: {
      show: true, position: 'top',
      fontSize: 10, color: TEXT,
      formatter: (p) => p.value.toFixed(1),
    },
    emphasis: { focus: 'series' },
  }));

  chart.setOption({
    backgroundColor: CHART_BG,
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      valueFormatter: v => v.toFixed(2) + '%',
    },
    legend: { bottom: 0, textStyle: { color: TEXT_DIM, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    grid: { left: 50, right: 16, top: 24, bottom: 36 },
    xAxis: {
      type: 'category', data: metricLabels,
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT, fontSize: 11 },
    },
    yAxis: {
      type: 'value', min: 60, max: 100, name: '分数 (%)',
      nameTextStyle: { color: TEXT_DIM, fontSize: 11 },
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT_DIM, fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(94,173,245,0.1)' } },
    },
    series,
  }, true);
}

/* ---------- 混淆矩阵 ---------- */
let cmCurrentKey = 'rf';
async function loadConfusion(key) {
  const r = await fetch(`/api/confusion/${key}`).then(r => r.json());
  const chart = getChart('chart-cm');
  if (!r.has_label) {
    chart.setOption({
      backgroundColor: CHART_BG,
      title: { text: '上传带 useful_votes 的数据后查看', left: 'center', top: 'center',
               textStyle: { color: TEXT_DIM, fontSize: 13 } },
    }, true);
    return;
  }
  const total = r.matrix.reduce((s, [, , v]) => s + v, 0);
  const max = Math.max(...r.matrix.map(d => d[2]));

  chart.setOption({
    backgroundColor: CHART_BG,
    title: {
      text: `${r.name}  ·  共 ${total.toLocaleString()} 样本`,
      left: 'center', top: 6,
      textStyle: { color: TEXT, fontSize: 14, fontWeight: 500 },
    },
    tooltip: {
      formatter: (p) => {
        const [a, b, v] = p.value;
        const labels = ['真实=无用 / 预测=无用 (TN)',
                        '真实=无用 / 预测=有用 (FP)',
                        '真实=有用 / 预测=无用 (FN)',
                        '真实=有用 / 预测=有用 (TP)'];
        const idx = a * 2 + b;
        return `${labels[idx]}<br/>样本数：<b>${v.toLocaleString()}</b><br/>占比：${(v / total * 100).toFixed(2)}%`;
      },
    },
    grid: { left: 84, right: 60, top: 60, bottom: 60 },
    xAxis: {
      type: 'category', data: [`预测：${r.labels[0]}`, `预测：${r.labels[1]}`],
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT, fontSize: 13 },
      splitArea: { show: true },
    },
    yAxis: {
      type: 'category', data: [`真实：${r.labels[0]}`, `真实：${r.labels[1]}`],
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT, fontSize: 13 },
      splitArea: { show: true },
    },
    visualMap: {
      min: 0, max: max,
      calculable: false, orient: 'vertical', right: 6, top: 'middle',
      inRange: { color: ['#0a1230', '#1e3a8a', '#3b82f6', '#22d3ee', '#fbbf24'] },
      textStyle: { color: TEXT_DIM, fontSize: 11 },
    },
    series: [{
      name: '混淆矩阵', type: 'heatmap', data: r.matrix,
      label: {
        show: true,
        formatter: (p) => {
          const v = p.value[2];
          return `{a|${v.toLocaleString()}}\n{b|${(v / total * 100).toFixed(1)}%}`;
        },
        rich: {
          a: { fontSize: 22, fontWeight: 600, color: '#fff', textShadowColor: '#000', textShadowBlur: 4 },
          b: { fontSize: 12, color: '#cdebff', padding: [4, 0, 0, 0] },
        },
      },
      itemStyle: { borderColor: 'rgba(94,173,245,0.2)', borderWidth: 1 },
    }],
  }, true);
}
document.querySelectorAll('.cm-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.cm-tab').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    cmCurrentKey = btn.dataset.key;
    loadConfusion(cmCurrentKey);
  });
});

/* ---------- 模型预测概率核密度分布（KDE） ---------- */
async function loadProbaDist() {
  const r = await fetch('/api/proba_dist').then(r => r.json());
  const x = r.x.map(v => v.toFixed(2));
  getChart('chart-proba').setOption({
    backgroundColor: CHART_BG,
    tooltip: { trigger: 'axis',
      formatter: (params) => {
        let s = `预测概率 = ${params[0].axisValueLabel}<br/>`;
        params.forEach(p => { s += `${p.marker} ${p.seriesName}: <b>${(+p.value).toFixed(3)}</b><br/>`; });
        return s;
      },
    },
    legend: { bottom: 0, textStyle: { color: TEXT_DIM, fontSize: 11 }, itemWidth: 12, itemHeight: 8 },
    grid: { left: 56, right: 18, top: 14, bottom: 40 },
    xAxis: {
      type: 'category', data: x, name: '预测概率', nameTextStyle: { color: TEXT_DIM },
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT_DIM, fontSize: 10, interval: 9 },
    },
    yAxis: {
      type: 'value', name: '密度', nameTextStyle: { color: TEXT_DIM },
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT_DIM, fontSize: 11 },
      splitLine: { lineStyle: { color: 'rgba(94,173,245,0.1)' } },
    },
    series: r.series.map(s => ({
      name: s.name,
      type: 'line',
      smooth: true, symbol: 'none',
      lineStyle: { width: 2.2, color: COLORS[s.key] },
      areaStyle: {
        opacity: 0.22,
        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: COLORS[s.key] + 'AA' },
          { offset: 1, color: COLORS[s.key] + '00' },
        ]),
      },
      data: s.density,
    })),
  }, true);
}

/* ---------- 特征重要度横向对比 ---------- */
let featCurrentKey = 'lgb';
let featCache = null;
async function loadFeatureImportance() {
  if (!featCache) {
    featCache = await fetch('/api/feature_importance?top_n=12').then(r => r.json());
  }
  const data = featCache[featCurrentKey];
  const chart = getChart('chart-feat');
  if (!data) {
    chart.setOption({ backgroundColor: CHART_BG,
      title: { text: '该模型不提供特征重要度', left: 'center', top: 'center',
               textStyle: { color: TEXT_DIM, fontSize: 13 } } }, true);
    return;
  }
  // 由小到大排序，便于横向条最高在顶部
  const pairs = data.features.map((f, i) => [f, data.importances[i]]).sort((a, b) => a[1] - b[1]);
  const features = pairs.map(p => p[0]);
  const values   = pairs.map(p => p[1]);
  const baseColor = COLORS[featCurrentKey];

  chart.setOption({
    backgroundColor: CHART_BG,
    tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' },
      formatter: (p) => `${p[0].name}<br/>重要度: <b>${(+p[0].value).toFixed(4)}</b>` },
    grid: { left: 8, right: 50, top: 8, bottom: 8, containLabel: true },
    xAxis: {
      type: 'value',
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisLabel: { color: TEXT_DIM, fontSize: 10 },
      splitLine: { lineStyle: { color: 'rgba(94,173,245,0.1)' } },
    },
    yAxis: {
      type: 'category', data: features,
      axisLine: { lineStyle: { color: 'rgba(94,173,245,0.4)' } },
      axisTick: { show: false },
      axisLabel: { color: TEXT, fontSize: 11 },
    },
    series: [{
      type: 'bar',
      data: values,
      barWidth: '55%',
      itemStyle: {
        borderRadius: [0, 4, 4, 0],
        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
          { offset: 0, color: baseColor + '55' },
          { offset: 1, color: baseColor },
        ]),
      },
      label: { show: true, position: 'right', color: TEXT,
               fontSize: 10, formatter: (p) => (+p.value).toFixed(3) },
    }],
  }, true);
}
document.querySelectorAll('#feat-tabs .cm-tab').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#feat-tabs .cm-tab').forEach(x => x.classList.remove('active'));
    btn.classList.add('active');
    featCurrentKey = btn.dataset.key;
    loadFeatureImportance();
  });
});

/* ---------- 实时预测 ---------- */
async function randomPredict() {
  const btn = document.getElementById('btn-random');
  if (!btn) return;
  btn.disabled = true; btn.textContent = '抽取中…';
  try {
    const r = await fetch('/api/random_predict').then(r => r.json());
    document.getElementById('predict-text').textContent = r.text || '(空文本)';
    const truthInfo = r.has_label
      ? `<span class="tag ${r.predictions[3].pred === r.true_label ? 'true' : 'false'}">真实标签：${r.true_label === 1 ? '有用' : '无用'} · 融合预测：${r.predictions[3].pred === 1 ? '有用' : '无用'} ${r.predictions[3].pred === r.true_label ? '✓' : '✗'}</span>`
      : `<span class="tag">融合预测：${r.predictions[3].pred === 1 ? '有用' : '无用'}（无真实标签）</span>`;
    document.getElementById('predict-meta').innerHTML = `
      <span class="tag">平台：${r.platform}</span>
      <span class="tag">类目：${r.category}</span>
      <span class="tag">评分：${r.rating}★</span>
      ${r.has_label ? `<span class="tag">真实有用票数：${r.useful_votes}</span>` : ''}
      ${truthInfo}
      <span class="tag">融合阈值：${r.threshold}</span>
    `;
    const html = r.predictions.map(p => {
      const w = (p.proba * 100).toFixed(1) + '%';
      const okCls = r.has_label ? (p.pred === r.true_label ? 'y' : 'n') : '';
      return `
        <div class="predict-row">
          <span class="name" style="color:${COLORS[p.key]}">${p.name}</span>
          <div class="bar-wrap"><div class="bar" style="width:${w}; background: linear-gradient(90deg, ${COLORS[p.key]}99, ${COLORS[p.key]});"></div></div>
          <span class="prob">${w}</span>
          <span class="mark ${okCls}">${p.pred === 1 ? '有用' : '无用'}</span>
        </div>`;
    }).join('');
    document.getElementById('predict-bars').innerHTML = html;
  } finally {
    btn.disabled = false; btn.textContent = '随机抽一条';
  }
}
document.getElementById('btn-random').addEventListener('click', randomPredict);

/* ---------- 自适应 ---------- */
window.addEventListener('resize', () => {
  Object.values(charts).forEach(c => c.resize());
});

/* ---------- 全图刷新（用于上传 / 重置后联动） ---------- */
async function refreshAll() {
  await Promise.all([loadSource(), loadOverview(), loadMetrics()]);
  await Promise.all([
    loadMultiMetric(),
    loadConfusion(cmCurrentKey),
    loadProbaDist(),
    loadFeatureImportance(),
  ]);
  randomPredict();
}

/* ---------- 上传 / 导出 / 重置 ---------- */
const fileInput = document.getElementById('file-input');
const upStatus  = document.getElementById('up-status');
const loadingMask = document.getElementById('loading-mask');
const loadingText = document.getElementById('loading-text');

function showMask(text) {
  loadingText.textContent = text || '处理中…';
  loadingMask.classList.add('show');
}
function hideMask() { loadingMask.classList.remove('show'); }

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  showMask(`正在上传并预测：${file.name}`);
  upStatus.textContent = '';
  const fd = new FormData();
  fd.append('file', file);
  try {
    const resp = await fetch('/api/upload_csv', { method: 'POST', body: fd });
    const r = await resp.json();
    if (!resp.ok || r.error) throw new Error(r.error || '上传失败');
    upStatus.innerHTML = `<span class="ok">✓ 已加载 ${r.total.toLocaleString()} 条 (${r.has_label ? '带标签' : '无标签'})</span>`;
    await refreshAll();
  } catch (err) {
    upStatus.innerHTML = `<span class="err">✗ ${err.message}</span>`;
    console.error(err);
  } finally {
    hideMask();
    fileInput.value = '';
  }
});

document.getElementById('btn-export').addEventListener('click', () => {
  window.location.href = '/api/download_predictions';
});

document.getElementById('btn-reset').addEventListener('click', async () => {
  if (!confirm('确认切回默认测试集吗？当前上传数据将被替换。')) return;
  showMask('正在重置为默认测试集…');
  upStatus.textContent = '';
  try {
    const resp = await fetch('/api/reset', { method: 'POST' });
    if (!resp.ok) throw new Error('重置失败');
    upStatus.innerHTML = `<span class="ok">✓ 已恢复默认测试集</span>`;
    await refreshAll();
  } catch (err) {
    upStatus.innerHTML = `<span class="err">✗ ${err.message}</span>`;
  } finally {
    hideMask();
  }
});

/* ---------- 初始化 ---------- */
(async function init() {
  try {
    await refreshAll();
    // 特征重要性只需加载一次（与数据无关，依赖训练好的模型）
    // 已移除特征重要性面板，所以不再加载
  } catch (e) {
    console.error('[init failed]', e);
    alert('图表初始化失败，请检查后端是否已启动并完成模型加载。\n' + e.message);
  }
  setInterval(randomPredict, 60000);
})();
