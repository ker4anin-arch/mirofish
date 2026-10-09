<template>
  <div class="bt-container">
    <nav class="navbar">
      <div class="nav-brand" @click="router.push('/')">MIROFISH</div>
      <div class="nav-links">
        <LanguageSwitcher />
      </div>
    </nav>

    <div class="bt-content">
      <!-- ======================= Form ======================= -->
      <section v-if="!testId" class="bt-form">
        <h1 class="bt-title">{{ mode === 'landing' ? $t('banner.landingTitle') : $t('banner.title') }}</h1>
        <div class="bt-modes">
          <button :class="{ active: mode === 'banner' }" @click="mode = 'banner'">{{ $t('banner.modeBanner') }}</button>
          <button :class="{ active: mode === 'landing' }" @click="mode = 'landing'">{{ $t('banner.modeLanding') }}</button>
        </div>
        <p class="bt-lead">{{ mode === 'landing' ? $t('banner.landingLead') : $t('banner.lead') }}</p>

        <div class="bt-field">
          <label class="bt-label">{{ $t('banner.audienceLabel') }}</label>
          <p class="bt-hint">{{ $t('banner.audienceHint') }}</p>
          <input type="file" multiple accept=".md,.txt,.pdf,.markdown" @change="onAudienceFiles" />
          <ul v-if="audienceFiles.length" class="bt-files">
            <li v-for="f in audienceFiles" :key="f.name">{{ f.name }}</li>
          </ul>
          <textarea
            v-model="audienceText"
            rows="3"
            class="bt-textarea"
            :placeholder="$t('banner.audienceTextPlaceholder')"
          ></textarea>
        </div>

        <div class="bt-field">
          <label class="bt-label">{{ mode === 'landing' ? $t('banner.pagesLabel') : $t('banner.bannersLabel') }}</label>
          <p class="bt-hint">{{ mode === 'landing' ? $t('banner.pagesHint') : $t('banner.bannersHint') }}</p>
          <input type="file" multiple accept="image/png,image/jpeg,image/webp" @change="onBannerFiles" />
          <div v-if="bannerPreviews.length" class="bt-previews">
            <figure v-for="(p, i) in bannerPreviews" :key="p.url" class="bt-preview">
              <img :src="p.url" :alt="p.name" />
              <figcaption>{{ String.fromCharCode(65 + i) }} — {{ p.name }}</figcaption>
            </figure>
          </div>
        </div>

        <div class="bt-row">
          <div class="bt-field">
            <label class="bt-label">{{ $t('banner.placementLabel') }}</label>
            <select v-model="placement" class="bt-select">
              <option v-for="p in placements" :key="p" :value="p">{{ $t(`banner.placements.${p}`) }}</option>
            </select>
          </div>
          <div v-if="mode === 'landing'" class="bt-field">
            <label class="bt-label">{{ $t('banner.deviceLabel') }}</label>
            <select v-model="device" class="bt-select">
              <option value="mobile">{{ $t('banner.devices.mobile') }}</option>
              <option value="desktop">{{ $t('banner.devices.desktop') }}</option>
            </select>
          </div>
          <div class="bt-field">
            <label class="bt-label">{{ $t('banner.panelLabel') }}</label>
            <input v-model.number="panelSize" type="number" min="5" max="300" step="10" class="bt-number" />
            <p class="bt-hint">{{ $t('banner.panelHint') }}</p>
          </div>
        </div>

        <div class="bt-field">
          <label class="bt-label">{{ $t('banner.goalLabel') }}</label>
          <textarea v-model="goal" rows="2" class="bt-textarea" :placeholder="$t('banner.goalPlaceholder')"></textarea>
        </div>

        <p v-if="error" class="bt-error">{{ error }}</p>
        <button class="bt-btn" :disabled="!canSubmit || submitting" @click="submit">
          {{ submitting ? $t('banner.starting') : $t('banner.start') }} →
        </button>

        <div v-if="history.length" class="bt-history">
          <h2>{{ $t('banner.historyTitle') }}</h2>
          <ul>
            <li v-for="h in history" :key="h.test_id">
              <a href="#" @click.prevent="openTest(h.test_id)">{{ h.title }}</a>
              <span class="bt-muted"> · {{ h.created_at?.replace('T', ' ') }} · {{ statusText(h.status) }}</span>
            </li>
          </ul>
        </div>
      </section>

      <!-- ======================= Progress / results ======================= -->
      <section v-else class="bt-result">
        <a href="#" class="bt-back" @click.prevent="backToForm">← {{ $t('banner.newTest') }}</a>
        <h1 class="bt-title">{{ test?.title || $t('banner.title') }}</h1>
        <p class="bt-muted" v-if="test">
          <template v-if="isLanding">{{ $t('banner.modeLanding') }} · {{ $t(`banner.devices.${test.device}`) }} · </template>
          {{ $t(`banner.placements.${test.placement}`) }} · {{ $t('banner.panelOf', { count: test.panel_actual || test.panel_size }) }}
        </p>

        <div v-if="test && test.status !== 'completed' && test.status !== 'failed'" class="bt-progress">
          <div class="bt-bar"><div class="bt-bar-fill" :style="{ width: (test.progress || 0) + '%' }"></div></div>
          <p>{{ test.message || $t('banner.waiting') }} ({{ test.progress || 0 }}%)</p>
        </div>
        <p v-if="test?.status === 'failed'" class="bt-error">{{ $t('banner.failed') }}: {{ test.message }}</p>

        <template v-if="test?.status === 'completed' && test.stats">
          <p class="bt-note">{{ $t('banner.syntheticNote') }}</p>
          <p v-if="hasSsr" class="bt-note bt-note-info">{{ $t('banner.ssrNote') }}</p>

          <div class="bt-cards">
            <div
              v-for="b in test.banners"
              :key="b.label"
              class="bt-card"
              :class="{ winner: test.stats.ranking?.[0] === b.label }"
            >
              <div class="bt-card-head">
                <strong>{{ b.label }}</strong> {{ b.name }}
                <span v-if="test.stats.ranking?.[0] === b.label" class="bt-badge">{{ $t('banner.leader') }}</span>
              </div>
              <img :src="imageUrl(b.label)" :alt="b.name" />
              <table class="bt-metrics">
                <tr v-if="confidenceOf(b.label)" class="bt-main">
                  <td>{{ $t('banner.metrics.win_pct') }}</td>
                  <td class="num">{{ confidenceOf(b.label).win_pct }}%</td>
                </tr>
                <tr v-for="m in metricRows" :key="m.key" :class="{ 'bt-main': m.main }">
                  <td>{{ $t(`banner.metrics.${m.key}`) }}</td>
                  <td class="num">
                    {{ formatMetric(test.stats.banners[b.label]?.overall?.[m.key], m.pct) }}
                    <span v-if="m.key === primaryKey && confidenceOf(b.label)" class="bt-ci">
                      {{ confidenceOf(b.label).ci[0].toFixed(1) }}–{{ confidenceOf(b.label).ci[1].toFixed(1) }}
                    </span>
                  </td>
                </tr>
              </table>
              <div v-if="distOf(b.label)" class="bt-dist" :title="$t('banner.distHint')">
                <div v-for="(p, i) in distOf(b.label)" :key="i" class="bt-dist-col">
                  <div class="bt-dist-bar"><div :style="{ height: p + '%' }"></div></div>
                  <span>{{ i + 1 }}</span>
                </div>
              </div>
              <div v-if="test.stats.banners[b.label]?.top_objections?.length" class="bt-objections">
                <strong>{{ $t('banner.objections') }}</strong>
                <ul>
                  <li v-for="o in test.stats.banners[b.label].top_objections.slice(0, 4)" :key="o">{{ o }}</li>
                </ul>
              </div>
            </div>
          </div>

          <template v-if="isLanding">
            <h2>{{ $t('banner.funnel') }}</h2>
            <p class="bt-hint">{{ $t('banner.funnelHint') }}</p>
            <div class="bt-table-wrap">
              <table class="bt-funnel">
                <thead>
                  <tr>
                    <th>{{ $t('banner.screen') }}</th>
                    <th v-for="b in test.banners" :key="b.label">{{ b.label }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="n in maxScreens" :key="n">
                    <td class="num">{{ n }}</td>
                    <td v-for="b in test.banners" :key="b.label">
                      <div v-if="n <= (b.screens?.length || 0)" class="bt-funnel-cell">
                        <img :src="screenUrl(b.label, n)" :alt="`${b.label} ${n}`" />
                        <div class="bt-funnel-metrics">
                          <div class="bt-reach"><div :style="{ width: (reachAt(b.label, n) ?? 0) + '%' }"></div></div>
                          <span>{{ $t('banner.reached') }}: {{ formatMetric(reachAt(b.label, n), true) }}</span>
                          <span>{{ $t('banner.convincing') }}: {{ formatMetric(test.stats.banners[b.label]?.overall?.convincing?.[n - 1]) }}</span>
                        </div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>

          <h2>{{ $t('banner.bySegment') }}</h2>
          <div class="bt-table-wrap">
            <table class="bt-segments">
              <thead>
                <tr>
                  <th>{{ $t('banner.segment') }}</th>
                  <th v-for="b in test.banners" :key="b.label">{{ b.label }}: {{ $t(`banner.metrics.${primaryKey}`) }}</th>
                  <th v-for="b in test.banners" :key="b.label + 't'">{{ b.label }}: {{ $t('banner.metrics.trust') }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="seg in segmentNames" :key="seg">
                  <td>{{ seg }} <span class="bt-muted">({{ segmentSize(seg) }})</span></td>
                  <td v-for="b in test.banners" :key="b.label" class="num">
                    {{ formatMetric(test.stats.banners[b.label]?.segments?.[seg]?.[primaryKey]) }}
                  </td>
                  <td v-for="b in test.banners" :key="b.label + 't'" class="num">
                    {{ formatMetric(test.stats.banners[b.label]?.segments?.[seg]?.trust) }}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <h2>{{ $t('banner.report') }}</h2>
          <div class="bt-report" v-html="renderMarkdown(test.report || '')"></div>

          <h2>{{ $t('banner.quotes') }}</h2>
          <div class="bt-quotes">
            <div v-for="(a, i) in quotes" :key="i" class="bt-quote">
              <span class="bt-muted">{{ a.banner }} · {{ a.segment }} · {{ $t(`banner.metrics.${primaryKey}`) }} {{ formatMetric(a[primaryKey]) }}</span>
              <p>«{{ a.click_thoughts || a.apply_thoughts || a.reaction || a.decision_reason }}»</p>
            </div>
          </div>
        </template>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import LanguageSwitcher from '../components/LanguageSwitcher.vue'
import {
  createBannerTest,
  getBannerTest,
  listBannerTests,
  getBannerAnswers,
  bannerImageUrl,
  landingScreenUrl
} from '../api/banner'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()

const placements = ['vk_feed', 'telegram', 'website', 'rsya', 'outdoor', 'other']
const landingMetricRows = [
  { key: 'ssr_would_apply', main: true, ssr: true },
  { key: 'ssr_would_apply_pct', pct: true, ssr: true },
  { key: 'ssr_trust', ssr: true },
  { key: 'would_apply' },
  { key: 'apply_pct', pct: true },
  { key: 'understood_pct', pct: true },
  { key: 'want_to_scroll' },
  { key: 'read_to_end_pct', pct: true },
  { key: 'trust' }
]
const bannerMetricRows = [
  { key: 'ssr_click_intent', main: true, ssr: true },
  { key: 'ssr_click_intent_pct', pct: true, ssr: true },
  { key: 'ssr_trust', ssr: true },
  { key: 'ssr_relevance', ssr: true },
  { key: 'click_intent' },
  { key: 'would_click_pct', pct: true },
  { key: 'clarity' },
  { key: 'understood_pct', pct: true },
  { key: 'trust' },
  { key: 'relevance' }
]

// form state
const audienceFiles = ref([])
const audienceText = ref('')
const bannerFiles = ref([])
const bannerPreviews = ref([])
const placement = ref('vk_feed')
const panelSize = ref(100)
const goal = ref('')
const mode = ref('banner')
const device = ref('mobile')
const submitting = ref(false)
const error = ref('')
const history = ref([])

// result state
const testId = ref(route.params.testId || null)
const test = ref(null)
const answers = ref([])
let pollTimer = null

const canSubmit = computed(() =>
  bannerFiles.value.length > 0 && (audienceFiles.value.length > 0 || audienceText.value.trim() !== '')
)

const onAudienceFiles = (event) => {
  audienceFiles.value = Array.from(event.target.files || [])
}

const onBannerFiles = (event) => {
  bannerPreviews.value.forEach((p) => URL.revokeObjectURL(p.url))
  const files = Array.from(event.target.files || []).slice(0, 5)
  bannerFiles.value = files
  bannerPreviews.value = files.map((f) => ({ name: f.name, url: URL.createObjectURL(f) }))
}

const submit = async () => {
  error.value = ''
  submitting.value = true
  try {
    const form = new FormData()
    audienceFiles.value.forEach((f) => form.append('audience', f))
    if (audienceText.value.trim()) form.append('audience_text', audienceText.value.trim())
    bannerFiles.value.forEach((f) => form.append('banners', f))
    form.append('placement', placement.value)
    form.append('mode', mode.value)
    form.append('device', device.value)
    form.append('panel_size', String(panelSize.value || 100))
    form.append('goal', goal.value)
    form.append('title', goal.value.slice(0, 80))
    const res = await createBannerTest(form)
    openTest(res.data.test_id)
  } catch (err) {
    error.value = err.message
  } finally {
    submitting.value = false
  }
}

const openTest = (id) => {
  router.push({ name: 'BannerTestResult', params: { testId: id } })
}

const backToForm = () => {
  router.push({ name: 'BannerTest' })
}

const loadTest = async () => {
  if (!testId.value) return
  try {
    const res = await getBannerTest(testId.value)
    test.value = res.data
    if (['completed', 'failed'].includes(test.value.status)) {
      stopPolling()
      if (test.value.status === 'completed') {
        const ans = await getBannerAnswers(testId.value)
        answers.value = ans.data || []
      }
    }
  } catch (err) {
    error.value = err.message
    stopPolling()
  }
}

const startPolling = () => {
  stopPolling()
  loadTest()
  pollTimer = setInterval(loadTest, 3000)
}

const stopPolling = () => {
  if (pollTimer) {
    clearInterval(pollTimer)
    pollTimer = null
  }
}

const loadHistory = async () => {
  try {
    const res = await listBannerTests()
    history.value = res.data || []
  } catch {
    history.value = []
  }
}

watch(() => route.params.testId, (id) => {
  testId.value = id || null
  test.value = null
  answers.value = []
  if (id) startPolling()
  else {
    stopPolling()
    loadHistory()
  }
})

onMounted(() => {
  if (testId.value) startPolling()
  else loadHistory()
})

onUnmounted(stopPolling)

const imageUrl = (label) => bannerImageUrl(testId.value, label)
const screenUrl = (label, n) => landingScreenUrl(testId.value, label, n)

const isLanding = computed(() => test.value?.mode === 'landing')
// Tests made before text-based scales have only the direct 1-5 scores.
const hasSsr = computed(() => (test.value?.stats?.ranking_metric || '').startsWith('ssr_'))
const primaryKey = computed(() => {
  const base = isLanding.value ? 'would_apply' : 'click_intent'
  return hasSsr.value ? `ssr_${base}` : base
})
const metricRows = computed(() =>
  (isLanding.value ? landingMetricRows : bannerMetricRows).filter((m) => hasSsr.value || !m.ssr)
)
const confidenceOf = (label) => test.value?.stats?.confidence?.[label] || null
const distOf = (label) => test.value?.stats?.banners?.[label]?.overall?.[`${primaryKey.value}_dist`] || null
const maxScreens = computed(() => Math.max(0, ...(test.value?.banners || []).map((b) => b.screens?.length || 0)))
const reachAt = (label, n) => test.value?.stats?.banners?.[label]?.overall?.reach?.[n - 1]

const statusText = (status) => t(`banner.status.${status}`, status)

const formatMetric = (value, pct = false) => {
  if (value === null || value === undefined) return '—'
  return pct ? `${value}%` : Number(value).toFixed(1)
}

const segmentNames = computed(() => {
  const names = new Set()
  Object.values(test.value?.stats?.banners || {}).forEach((b) =>
    Object.keys(b.segments || {}).forEach((s) => names.add(s))
  )
  return Array.from(names)
})

const segmentSize = (seg) => {
  const first = test.value?.banners?.[0]?.label
  return test.value?.stats?.banners?.[first]?.segments?.[seg]?.n ?? ''
}

const quotes = computed(() => {
  // A spread of reactions: best and worst click intent per banner.
  const out = []
  const byBanner = {}
  answers.value.forEach((a) => {
    if (a.reaction || a.decision_reason) (byBanner[a.banner] = byBanner[a.banner] || []).push(a)
  })
  Object.values(byBanner).forEach((rows) => {
    const key = primaryKey.value
    const sorted = [...rows].sort((x, y) => (y[key] || 0) - (x[key] || 0))
    out.push(...sorted.slice(0, 3), ...sorted.slice(-3))
  })
  return out
})

// Minimal, safe Markdown: escape HTML first, then headings, bold, lists, tables.
const escapeHtml = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const inline = (s) => s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')

const renderMarkdown = (md) => {
  const lines = escapeHtml(md).split('\n')
  const html = []
  let list = null
  let table = []
  const flushList = () => {
    if (list) {
      html.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join('')}</${list.tag}>`)
      list = null
    }
  }
  const flushTable = () => {
    if (table.length) {
      const rows = table.filter((r) => !/^\|?\s*:?-{2,}/.test(r))
      const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => inline(c.trim()))
      const [head, ...body] = rows
      html.push(
        '<table><thead><tr>' + cells(head).map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>' +
        body.map((r) => '<tr>' + cells(r).map((c) => `<td>${c}</td>`).join('') + '</tr>').join('') +
        '</tbody></table>'
      )
      table = []
    }
  }
  for (const raw of lines) {
    const line = raw.trimEnd()
    if (line.trim().startsWith('|')) { flushList(); table.push(line.trim()); continue }
    flushTable()
    let m
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flushList(); const lvl = Math.min(m[1].length + 1, 5); html.push(`<h${lvl}>${inline(m[2])}</h${lvl}>`); continue }
    if ((m = line.match(/^\s*[-*]\s+(.*)$/))) { if (!list || list.tag !== 'ul') { flushList(); list = { tag: 'ul', items: [] } } list.items.push(m[1]); continue }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { if (!list || list.tag !== 'ol') { flushList(); list = { tag: 'ol', items: [] } } list.items.push(m[1]); continue }
    flushList()
    if (line.trim()) html.push(`<p>${inline(line)}</p>`)
  }
  flushList()
  flushTable()
  return html.join('\n')
}
</script>

<style scoped>
.bt-container { min-height: 100vh; background: #fff; color: #111; }
.navbar {
  height: 60px; background: #000; color: #fff; display: flex;
  justify-content: space-between; align-items: center; padding: 0 40px;
}
.nav-brand { font-family: 'JetBrains Mono', monospace; font-weight: 800; letter-spacing: 1px; cursor: pointer; }
.nav-links { display: flex; align-items: center; gap: 16px; }
.bt-content { max-width: 1200px; margin: 0 auto; padding: 40px 24px 80px; }
.bt-title { font-size: 2rem; margin: 0 0 8px; }
.bt-lead { color: #555; max-width: 760px; line-height: 1.6; margin-bottom: 32px; }
.bt-field { margin-bottom: 24px; flex: 1; }
.bt-row { display: flex; gap: 32px; flex-wrap: wrap; }
.bt-label { display: block; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; }
.bt-hint { color: #888; font-size: 0.82rem; margin: 0 0 8px; line-height: 1.5; }
.bt-textarea, .bt-select, .bt-number {
  width: 100%; box-sizing: border-box; padding: 10px 12px; border: 1px solid #ddd;
  background: #fafafa; font: inherit; margin-top: 8px;
}
.bt-number { width: 140px; }
.bt-files { margin: 8px 0 0; padding-left: 18px; color: #555; font-size: 0.9rem; }
.bt-previews { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
.bt-preview { margin: 0; width: 220px; }
.bt-preview img { width: 100%; border: 1px solid #eee; }
.bt-preview figcaption { font-size: 0.8rem; color: #666; margin-top: 4px; }
.bt-btn {
  background: #000; color: #fff; border: none; padding: 14px 28px; font-size: 1rem;
  font-family: 'JetBrains Mono', monospace; cursor: pointer;
}
.bt-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.bt-error { color: #c62828; }
.bt-muted { color: #888; font-size: 0.85rem; }
.bt-history { margin-top: 48px; }
.bt-history h2 { font-size: 1.1rem; }
.bt-history li { margin-bottom: 6px; }
.bt-back { color: #555; text-decoration: none; font-size: 0.9rem; }
.bt-progress { margin: 32px 0; }
.bt-bar { height: 8px; background: #eee; }
.bt-bar-fill { height: 100%; background: #FF4500; transition: width 0.5s; }
.bt-note { background: #fff8e1; border-left: 3px solid #f5b400; padding: 10px 14px; font-size: 0.88rem; color: #555; }
.bt-modes { display: flex; gap: 0; margin: 4px 0 16px; }
.bt-modes button {
  border: 1px solid #000; background: #fff; padding: 8px 18px; cursor: pointer;
  font-family: 'JetBrains Mono', monospace; font-size: 0.85rem;
}
.bt-modes button.active { background: #000; color: #fff; }
.bt-funnel { border-collapse: collapse; }
.bt-funnel th, .bt-funnel td { border-bottom: 1px solid #eee; padding: 8px 10px; vertical-align: top; text-align: left; }
.bt-funnel-cell { display: flex; gap: 10px; align-items: flex-start; min-width: 280px; }
.bt-funnel-cell img { width: 140px; border: 1px solid #eee; }
.bt-funnel-metrics { display: flex; flex-direction: column; gap: 4px; font-size: 0.82rem; min-width: 120px; }
.bt-reach { height: 6px; background: #eee; }
.bt-reach div { height: 100%; background: #FF4500; }
.bt-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 24px 0 40px; }
.bt-card { border: 1px solid #e5e5e5; padding: 16px; }
.bt-card.winner { border: 2px solid #FF4500; }
.bt-card img { width: 100%; border: 1px solid #eee; margin: 10px 0; }
.bt-card-head { font-size: 0.95rem; }
.bt-badge { background: #FF4500; color: #fff; font-size: 0.72rem; padding: 2px 8px; margin-left: 6px; }
.bt-metrics { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
.bt-metrics td { padding: 4px 0; border-bottom: 1px solid #f2f2f2; }
.bt-metrics tr.bt-main td { font-weight: 700; }
.bt-ci { display: block; font-size: 0.72rem; color: #999; font-weight: 400; }
.bt-note-info { background: #eef5ff; border-left-color: #3b82f6; }
.bt-dist { display: flex; gap: 6px; align-items: flex-end; margin-top: 10px; height: 60px; }
.bt-dist-col { flex: 1; display: flex; flex-direction: column; align-items: center; font-size: 0.7rem; color: #888; }
.bt-dist-bar { width: 100%; height: 44px; background: #f5f5f5; display: flex; align-items: flex-end; }
.bt-dist-bar div { width: 100%; background: #FF4500; }
.num { text-align: right; font-family: 'JetBrains Mono', monospace; }
.bt-objections { margin-top: 12px; font-size: 0.85rem; color: #444; }
.bt-objections ul { padding-left: 18px; margin: 6px 0 0; }
.bt-table-wrap { overflow-x: auto; }
.bt-segments { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
.bt-segments th, .bt-segments td { border-bottom: 1px solid #eee; padding: 8px 10px; text-align: left; }
.bt-segments th { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #666; }
.bt-report { line-height: 1.65; max-width: 900px; }
.bt-report :deep(table) { border-collapse: collapse; margin: 12px 0; font-size: 0.88rem; display: block; overflow-x: auto; }
.bt-report :deep(th), .bt-report :deep(td) { border: 1px solid #e5e5e5; padding: 6px 8px; text-align: left; }
.bt-quotes { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }
.bt-quote { border-left: 3px solid #ddd; padding: 6px 12px; }
.bt-quote p { margin: 4px 0 0; }
@media (max-width: 640px) {
  .navbar { padding: 0 16px; }
  .bt-content { padding: 24px 16px 60px; }
}
</style>
